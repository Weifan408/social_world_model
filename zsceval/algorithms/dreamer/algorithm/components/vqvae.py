import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings  # 码本大小 (K)
        self.embedding_dim = embedding_dim   # 码本向量维度
        self.commitment_cost = commitment_cost  # beta 值

        # 码本 (embedding) 是一种可学习的参数
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 初始化码本，通常使用均匀分布
        nn.init.uniform_(
            self.embeddings.weight, -1.0 / self.num_embeddings,
            1.0 / self.num_embeddings
        )

    def forward(self, inputs: torch.Tensor):
        # inputs 形状: (Batch_size, embedding_dim)

        # [1] 计算输入与码本中每个向量的距离
        # 扩展输入和码本，以便进行广播计算
        # inputs_expanded: (Batch_size, 1, embedding_dim)
        # embeddings_expanded: (1, num_embeddings, embedding_dim)
        # distances: (Batch_size, num_embeddings)
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embeddings.weight.T))

        # [2] 找到最近的码本向量索引
        # encoding_indices 形状: (Batch_size,)
        encoding_indices = torch.argmin(distances, dim=1)
        # 将索引转换为 one-hot 编码
        # encodings 形状: (Batch_size, num_embeddings)
        encodings = F.one_hot(
            encoding_indices, num_classes=self.num_embeddings).float()

        # [3] 从码本中获取量化后的向量
        # quantize 形状: (Batch_size, embedding_dim)
        quantized = torch.matmul(encodings, self.embeddings.weight)

        # [4] 计算 VQ 损失
        # 承诺损失 (commitment loss): 使编码器输出 (inputs) 接近码本向量 (quantized)
        # 这部分梯度会通过 (inputs - stop_gradient[quantized]) 反向传播给编码器
        # .detach() 停止梯度回传到 quantized
        e_latent_loss = F.mse_loss(inputs, quantized.detach())

        # 码本损失 (codebook loss): 使码本向量 (quantized) 接近编码器输出 (inputs)
        # 这部分梯度会通过 (quantized - stop_gradient[inputs]) 更新码本
        # 码本的更新通常直接通过优化器更新 self.embeddings.weight
        # .detach() 停止梯度回传到 inputs
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        # VQ Loss 是码本损失加上承诺损失
        # commitment_cost (beta) 是超参数，通常在0.1到0.25之间
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # [5] Straight-Through Estimator
        # 在前向传播时使用量化后的向量 (quantized)，但在反向传播时将梯度直接传递给输入 (inputs)。
        # 这使得编码器能够接收到有意义的梯度。
        quantized_st = inputs + (quantized - inputs).detach()

        # 记录每个码本向量的使用频率
        avg_probs = torch.mean(encodings, dim=0)  # (num_embeddings,)
        # +1e-10 防止 log(0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        return quantized_st, loss, perplexity, encoding_indices


class VQVAEPolicyIDPredictor(nn.Module):
    """
    基于 VQ-VAE 的策略 ID 推断网络。
    编码器将观测映射到连续潜在空间，然后通过 VQ 层量化到离散码本。
    量化后的向量用于预测两个智能体的 ID。
    """

    def __init__(
        self,
        args,
        input_dim: int,
        num_heads: int,
        num_classes: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float
    ):
        super().__init__()
        self.args = args
        self.num_heads = num_heads  # 通常为 2 (预测两个智能体ID)
        self.num_classes = num_classes  # 9 个策略ID
        self.num_embeddings = num_embeddings  # 码本大小 K
        self.embedding_dim = embedding_dim   # 码本向量维度
        self.commitment_cost = commitment_cost  # beta 值

        self.projection_layer = MLP(
            args=args,
            input_dim=input_dim,
            hidden_dim=embedding_dim,
            num_layers=2,
            activation='tanh',
        )

        # 向量量化模块
        self.vq_vae = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost)

        # ID 预测头：将量化后的向量映射到智能体 ID 的 logits
        # 两个独立的头，每个头预测一个智能体ID
        self.id_prediction_heads = nn.ModuleList(
            [nn.Linear(embedding_dim, num_classes) for _ in range(num_heads)]
        )

    def forward(self, h: torch.Tensor):
        # h 形状: (Batch_size, Input_dim) 或 (Batch_size, Sequence_length, Input_dim)
        # 如果 h 是序列，需要先通过一个适合序列的编码器（如 Transformer/LSTM）
        # 这里假设 h 已经被处理为 (Batch_size, Input_dim)，或者 MLP 内部处理序列的聚合

        # [1] 通过投影层调整维度
        # z_e 形状: (Batch_size, embedding_dim)
        z_e = self.projection_layer(h)

        # [2] 向量量化
        # quantized_st: 量化后的向量 (通过 Straight-Through Estimator)
        # vq_loss: 包含码本损失和承诺损失
        # perplexity: 衡量码本使用效率
        # encoding_indices: 每个样本对应的码本索引
        quantized_st, vq_loss, perplexity, encoding_indices = self.vq_vae(z_e)

        # [3] ID 预测头
        # 遍历每个预测头，将量化后的向量映射到 logits
        # logits_list 包含 num_heads 个 (Batch_size, num_classes) 张量
        logits_list = [head(quantized_st) for head in self.id_prediction_heads]
        # 将 logits 堆叠成 (Batch_size, num_heads, num_classes)
        policy_ids_logits = torch.stack(logits_list, dim=1)

        # dist = torch.distributions.OneHotCategorical(logits=policy_ids_logits)

        return {
            "policy_ids_logits": policy_ids_logits,  # 用于计算 ID 预测损失
            "quantized_st": quantized_st,
            "vq_loss": vq_loss,                     # VQ-VAE 核心损失
            "perplexity": perplexity,               # 码本使用效率指标
            "encoding_indices": encoding_indices    # 码本索引，可用于分析
        }

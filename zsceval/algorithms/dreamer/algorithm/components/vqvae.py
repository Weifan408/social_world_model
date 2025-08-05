import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim   
        self.commitment_cost = commitment_cost 

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(
            self.embeddings.weight, -1.0 / self.num_embeddings,
            1.0 / self.num_embeddings
        )

    def forward(self, inputs: torch.Tensor):
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embeddings.weight.T))

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(
            encoding_indices, num_classes=self.num_embeddings).float()

        quantized = torch.matmul(encodings, self.embeddings.weight)

        e_latent_loss = F.mse_loss(inputs, quantized.detach())

        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_st = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)  # (num_embeddings,)

        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        return quantized_st, loss, perplexity, encoding_indices


class VQVAEPolicyIDPredictor(nn.Module):

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
        self.num_heads = num_heads  
        self.num_classes = num_classes  
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim  
        self.commitment_cost = commitment_cost  

        self.projection_layer = MLP(
            args=args,
            input_dim=input_dim,
            hidden_dim=embedding_dim,
            num_layers=2,
            activation='tanh',
        )

        self.vq_vae = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost)

        self.id_prediction_heads = nn.ModuleList(
            [nn.Linear(embedding_dim, num_classes) for _ in range(num_heads)]
        )

    def forward(self, h: torch.Tensor):
        z_e = self.projection_layer(h)

        quantized_st, vq_loss, perplexity, encoding_indices = self.vq_vae(z_e)

        logits_list = [head(quantized_st) for head in self.id_prediction_heads]
        policy_ids_logits = torch.stack(logits_list, dim=1)

        return {
            "policy_ids_logits": policy_ids_logits, 
            "quantized_st": quantized_st,
            "vq_loss": vq_loss,                     
            "perplexity": perplexity,              
            "encoding_indices": encoding_indices  
        }

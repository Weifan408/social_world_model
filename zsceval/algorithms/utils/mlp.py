import torch
import torch.nn as nn
import torch.distributions as td

from .attention import Encoder
from .util import get_clones, init


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, activation_id):
        super().__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class CONVLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, use_orthogonal, activation_id):
        super().__init__()

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.conv = nn.Sequential(
            init_(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_size // 4,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                )
            ),
            active_func,  # nn.BatchNorm1d(hidden_size//4),
            init_(
                nn.Conv1d(
                    in_channels=hidden_size // 4,
                    out_channels=hidden_size // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
            active_func,  # nn.BatchNorm1d(hidden_size//2),
            init_(
                nn.Conv1d(
                    in_channels=hidden_size // 2,
                    out_channels=hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
            active_func,
        )  # , nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        x = self.conv(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, use_attn_internal=False, use_cat_self=True):
        super().__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_attn = args.use_attn
        self._use_attn_internal = use_attn_internal
        self._use_average_pool = args.use_average_pool
        self._use_conv1d = args.use_conv1d
        self._stacked_frames = args.stacked_frames
        self._layer_N = 0 if args.use_single_network else args.layer_N
        self._attn_size = args.attn_size
        self.hidden_size = args.hidden_size
        # self.use_agent_policy_id = args.use_agent_policy_id

        # logger.debug(
        #     f"use_agent_policy_id {self.use_agent_policy_id} obs_shape {obs_shape}"
        # )

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        if self._use_attn and self._use_attn_internal:
            if self._use_average_pool:
                if use_cat_self:
                    inputs_dim = self._attn_size + obs_shape[-1][1]
                else:
                    inputs_dim = self._attn_size
            else:
                split_inputs_dim = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    split_inputs_dim += split_shape[i][0]
                inputs_dim = split_inputs_dim * self._attn_size
            self.attn = Encoder(args, obs_shape, use_cat_self)
            self.attn_norm = nn.LayerNorm(inputs_dim)
        else:
            inputs_dim = obs_dim

        if self._use_conv1d:
            self.conv = CONVLayer(
                self._stacked_frames,
                self.hidden_size,
                self._use_orthogonal,
                self._activation_id,
            )
            random_x = torch.FloatTensor(
                1, self._stacked_frames, inputs_dim // self._stacked_frames)
            random_out = self.conv(random_x)
            assert len(random_out.shape) == 3
            inputs_dim = random_out.size(-1) * random_out.size(-2)

        self.mlp = MLPLayer(
            inputs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._activation_id,
        )

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_attn and self._use_attn_internal:
            x = self.attn(x, self_idx=-1)
            x = self.attn_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        return x

    @property
    def output_size(self):
        return self.hidden_size


class MLP(nn.Module):
    def __init__(
        self, in_dim, out_dim, layers, hidden, use_orthogonal, activation_id
    ):
        super().__init__()

        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.base = MLPLayer(in_dim, hidden, layers,
                             use_orthogonal, activation_id)
        self.out = init_(nn.Linear(hidden, out_dim))

    def forward(self, x):
        x = self.base(x)
        x = self.out(x)
        return x


class MLPBinary(nn.Module):
    def __init__(
        self, in_dim, out_dim, layers, hidden, use_orthogonal, activation_id
    ):
        super().__init__()

        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.base = MLPLayer(in_dim, hidden, layers,
                             use_orthogonal, activation_id)
        self.out = init_(nn.Linear(hidden, out_dim))

    def forward(self, x):
        x = self.base(x)
        dist_inputs = self.out(x)
        return td.independent.Independent(td.Bernoulli(logits=dist_inputs), 1)


class MLPDecoder(nn.Module):
    def __init__(self, args, obs_shape, use_attn_internal=False, use_cat_self=True):
        super().__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_attn = args.use_attn
        self._use_attn_internal = use_attn_internal
        self._use_average_pool = args.use_average_pool
        self._use_conv1d = args.use_conv1d
        self._stacked_frames = args.stacked_frames
        self._layer_N = 0 if args.use_single_network else args.layer_N
        self._attn_size = args.attn_size
        self.hidden_size = args.hidden_size

        # 确定输出维度（原始观测的维度）
        self.obs_dim = obs_shape[0]

        # 计算中间层维度（与MLPBase中的inputs_dim对应）
        if self._use_attn and self._use_attn_internal:
            if self._use_average_pool:
                if use_cat_self:
                    intermediate_dim = self._attn_size + obs_shape[-1][1]
                else:
                    intermediate_dim = self._attn_size
            else:
                split_inputs_dim = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    split_inputs_dim += split_shape[i][0]
                intermediate_dim = split_inputs_dim * self._attn_size
        else:
            intermediate_dim = self.obs_dim

        # 如果使用conv1d，计算conv解码前的维度
        if self._use_conv1d:
            # 创建随机输入模拟MLPBase中的conv输出，以确定维度
            random_x = torch.FloatTensor(
                1, self._stacked_frames, intermediate_dim // self._stacked_frames
            )
            random_out = MLPBase.conv(random_x)  # 使用encoder的conv测试
            conv_output_dim = random_out.size(-1) * random_out.size(-2)
            intermediate_dim = conv_output_dim

        # 主MLP解码器 - 从hidden_size到intermediate_dim
        self.mlp_decoder = self._build_mlp_decoder(
            self.hidden_size,
            intermediate_dim,
            self._layer_N,
            self._use_orthogonal,
            self._activation_id,
        )

        # 如果使用Conv1D，添加反Conv1D层
        if self._use_conv1d:
            self.conv_decoder = self._build_conv_decoder(
                self.hidden_size,
                self._stacked_frames,
                intermediate_dim // self._stacked_frames,
                self._use_orthogonal,
                self._activation_id,
            )

        # 如果使用注意力，添加反注意力层（简化版）
        if self._use_attn and self._use_attn_internal:
            # 注意：反向注意力机制通常很复杂，这里简化为线性层
            self.attn_decoder = nn.Linear(intermediate_dim, self.obs_dim)
            self.attn_denorm = nn.LayerNorm(self.obs_dim)

        # 如果使用特征标准化，添加反标准化层
        if self._use_feature_normalization:
            self.feature_denorm = nn.LayerNorm(self.obs_dim)

    def _build_mlp_decoder(
        self, input_dim, output_dim, layer_N, use_orthogonal, activation_id
    ):
        """构建MLP解码器"""
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 构建与MLPLayer类似但反向的结构
        layers = []

        # 如果有多个隐藏层
        if layer_N > 0:
            # 输入映射
            layers.append(
                nn.Sequential(
                    init_(nn.Linear(input_dim, self.hidden_size)),
                    active_func,
                    nn.LayerNorm(self.hidden_size),
                )
            )

            # 中间隐藏层
            for _ in range(layer_N):
                layers.append(
                    nn.Sequential(
                        init_(nn.Linear(self.hidden_size, self.hidden_size)),
                        active_func,
                        nn.LayerNorm(self.hidden_size),
                    )
                )

            # 输出层
            layers.append(
                nn.Sequential(
                    init_(nn.Linear(self.hidden_size, output_dim)),
                    active_func,
                    nn.LayerNorm(output_dim),
                )
            )
        else:
            # 单层直接映射
            layers.append(
                nn.Sequential(
                    init_(nn.Linear(input_dim, output_dim)),
                    active_func,
                    nn.LayerNorm(output_dim),
                )
            )

        return nn.Sequential(*layers)

    def _build_conv_decoder(
        self,
        hidden_size,
        stacked_frames,
        output_channels,
        use_orthogonal,
        activation_id,
    ):
        """构建反卷积解码器"""
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 反卷积层 - 与原始CONVLayer逆序
        conv_decoder = nn.Sequential(
            init_(
                nn.ConvTranspose1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
            active_func,
            init_(
                nn.ConvTranspose1d(
                    in_channels=hidden_size // 2,
                    out_channels=hidden_size // 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
            active_func,
            init_(
                nn.ConvTranspose1d(
                    in_channels=hidden_size // 4,
                    out_channels=output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    output_padding=1,  # 确保尺寸匹配
                )
            ),
            active_func,
        )

        return conv_decoder

    def forward(self, x):
        """
        将编码特征解码回原始观测
        Args:
            x: 形状为[batch_size, hidden_size]的编码特征

        Returns:
            解码后的观测，形状为原始观测的shape
        """
        # 主MLP解码
        x = self.mlp_decoder(x)

        # 如果有Conv1D，应用反卷积
        if self._use_conv1d:
            batch_size = x.size(0)
            # 重塑为卷积输入格式
            x = x.view(batch_size, -1, 1)  # 临时reshape以匹配ConvTranspose1d的输入
            x = self.conv_decoder(x)
            x = x.view(batch_size, self._stacked_frames, -1)

        # 如果有注意力，应用反注意力（简化）
        if self._use_attn and self._use_attn_internal:
            x = self.attn_decoder(x)
            x = self.attn_denorm(x)

        # 如果有特征标准化，应用反标准化
        if self._use_feature_normalization:
            x = self.feature_denorm(x)

        return x

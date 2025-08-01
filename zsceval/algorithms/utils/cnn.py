import numpy as np
import torch
import torch.nn as nn

from .util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None, hidden=None):
        super().__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self.hidden_size = args.hidden_size if hidden is None else hidden
        self.cnn_keys = ["rgb"]

        self.cnn = self._build_cnn_model(
            obs_shape,
            self.cnn_keys,
            cnn_layers_params,
            self.hidden_size,
            self._use_orthogonal,
            self._activation_id,
        )

    def _build_cnn_model(
        self,
        obs_shape,
        cnn_keys,
        cnn_layers_params,
        hidden_size,
        use_orthogonal,
        activation_id,
    ):
        if cnn_layers_params is None:
            cnn_layers_params = [(16, 5, 1, 0), (32, 3, 1, 0), (16, 3, 1, 0)]
        else:

            def _convert(params):
                output = []
                for l in params.split(" "):
                    output.append(tuple(map(int, l.split(","))))
                return output

            cnn_layers_params = _convert(cnn_layers_params)

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        n_cnn_input = 0
        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                n_cnn_input += obs_shape[2]
                cnn_dims = np.array(obs_shape[:2], dtype=np.float32)
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "local_merge_obs",
                "trace_image",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                n_cnn_input += obs_shape.shape[0]
                cnn_dims = np.array(obs_shape[1:3], dtype=np.float32)
            else:
                raise NotImplementedError

        cnn_layers = []
        prev_out_channels = None
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_layers.append(nn.MaxPool2d(2))

            if i == 0:
                in_channels = n_cnn_input
            else:
                in_channels = prev_out_channels

            cnn_layers.append(
                init_(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            )

            cnn_layers.append(active_func)
            prev_out_channels = out_channels

        for i, (_, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_dims = self._maxpool_output_dim(
                    dimension=cnn_dims,
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array([2, 2], dtype=np.float32),
                    stride=np.array([2, 2], dtype=np.float32),
                )
            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(
                    [kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )

        self.cnn_output_dim = (prev_out_channels, cnn_dims[0], cnn_dims[1])
        if (cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1]) > 20000:
            cnn_layers += [
                Flatten(),
                init_(
                    nn.Linear(cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1], 2048)),
                active_func,
                nn.LayerNorm(2048),
                init_(nn.Linear(2048, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
        else:
            cnn_layers += [
                Flatten(),
                init_(
                    nn.Linear(
                        cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                        hidden_size,
                    )
                ),
                active_func,
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
        return nn.Sequential(*cnn_layers)

    def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(((dimension[i] + 2 * padding[i] - dilation[i]
                             * (kernel_size[i] - 1) - 1) / stride[i]) + 1)
                )
            )
        return tuple(out_dimension)

    def _maxpool_output_dim(self, dimension, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1))
            )
        return tuple(out_dimension)

    def _build_cnn_input(self, obs, cnn_keys):
        cnn_input = []

        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                cnn_input.append(obs.permute(0, 3, 1, 2) / 255.0)
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "trace_image",
                "local_merge_obs",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                cnn_input.append(obs)
            else:
                raise NotImplementedError

        cnn_input = torch.cat(cnn_input, dim=1)
        return cnn_input

    def forward(self, x):
        cnn_input = self._build_cnn_input(x, self.cnn_keys)
        cnn_x = self.cnn(cnn_input)
        return cnn_x

    @property
    def output_size(self):
        return self.hidden_size


class CNNDecoder(nn.Module):
    def __init__(
        self,
        args,
        obs_shape,
        cnn_output_dim,
        hidden=None,
        cnn_layers_params=None
    ):
        super().__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self.hidden_size = args.hidden_size if hidden is None else hidden
        self.output_shape = obs_shape  # 原始观测的形状
        self.cnn_keys = ["rgb"]  # 与CNNBase保持一致

        # 构建解码器网络
        self.pre_cnn_decoder, self.cnn_decoder = self._build_decoder_model(
            obs_shape=obs_shape,
            cnn_keys=self.cnn_keys,
            input_dim=args.wm_feat_size,
            cnn_output_dim=cnn_output_dim,
            cnn_layers_params=cnn_layers_params,
            hidden_size=self.hidden_size,
            use_orthogonal=self._use_orthogonal,
            activation_id=self._activation_id,
        )

    def _build_decoder_model(
        self,
        obs_shape,
        cnn_keys,
        input_dim,
        cnn_output_dim,
        cnn_layers_params,
        hidden_size,
        use_orthogonal,
        activation_id,
    ):
        n_cnn_input = 0
        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                n_cnn_input += obs_shape[2]
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "local_merge_obs",
                "trace_image",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                n_cnn_input += obs_shape.shape[0]
            else:
                raise NotImplementedError

        if cnn_layers_params is None:
            cnn_layers_params = [(16, 5, 1, 0), (32, 3, 1, 0), (16, 3, 1, 0)]
        else:

            def _convert(params):
                output = []
                for l in params.split(" "):
                    output.append(tuple(map(int, l.split(","))))
                return output

            cnn_layers_params = _convert(cnn_layers_params)
        deconv_params = list(reversed(cnn_layers_params))[1:]

        # 激活函数和初始化方法
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        mlp_layers = [
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        ]

        cnn_output_size = np.prod(cnn_output_dim)
        cnn_layers = []
        if cnn_output_size > 20000:
            cnn_layers = []
        else:
            cnn_layers = [
                init_(nn.Linear(hidden_size, cnn_output_size)),
                active_func,
                nn.Unflatten(
                    -1,
                    (
                        cnn_output_dim[0],
                        cnn_output_dim[1],
                        cnn_output_dim[2],
                    ),
                )
            ]
        for i, (out_channels, kernel_size, stride, padding) in enumerate(deconv_params):
            in_channels = out_channels
            out_channels_next = deconv_params[i + 1][0] if i + \
                1 < len(deconv_params) else n_cnn_input
            cnn_layers.append(
                init_(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels_next,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            )
            if i + 1 < len(deconv_params):
                cnn_layers.append(active_func)

            if self._use_maxpool2d and i < len(deconv_params) - 1:
                cnn_layers.append(
                    nn.Upsample(scale_factor=2, mode="nearest"))

        return nn.Sequential(*mlp_layers), nn.Sequential(*cnn_layers)

    def _build_decoder_output(self, obs, cnn_keys):
        decoder_output = []

        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                # [B*T, C, H, W] -> [B*T, H, W, C]
                decoder_output.append(obs.permute(0, 2, 3, 1))
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "trace_image",
                "local_merge_obs",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                decoder_output.append(obs)
            else:
                raise NotImplementedError

        decoder_output = torch.cat(decoder_output, dim=1)
        return decoder_output

    def forward(self, x):
        """
        接收编码后的向量，解码回原始观测空间

        Args:
            x: 形状为[batch_size, hidden_size]的特征向量

        Returns:
            重构的观测，形状为原始观测的形状
        """
        pre_cnn_out = self.pre_cnn_decoder(x)  # [B, C, H, W]
        out = self.cnn_decoder(pre_cnn_out)
        output = self._build_decoder_output(out, self.cnn_keys)
        return output, pre_cnn_out

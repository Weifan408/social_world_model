import numpy as np

import torch
import torch.nn as nn

from .util import init


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
            input_dim=args.wm_model_hidden,
            cnn_output_dim=cnn_output_dim,
            cnn_layers_params=cnn_layers_params,
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

        deconv_params = list(reversed(cnn_layers_params))

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),
                       nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        pre_cnn_decoder = nn.Sequential(
            init_(
                nn.Linear(
                    input_dim, np.prod(cnn_output_dim)
                )
            ),
            nn.Unflatten(
                -1,
                (
                    cnn_output_dim[0],
                    cnn_output_dim[1],
                    cnn_output_dim[2],
                ),
            )
        )
        cnn_transpose_layers = []
        for i, (out_channels, kernel_size, stride, padding) in enumerate(deconv_params):
            in_channels = out_channels
            out_channels_next = deconv_params[i + 1][0] if i + \
                1 < len(deconv_params) else n_cnn_input
            cnn_transpose_layers.append(
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
                cnn_transpose_layers.append(active_func)

            if self._use_maxpool2d and i < len(deconv_params) - 1:
                cnn_transpose_layers.append(
                    nn.Upsample(scale_factor=2, mode="nearest"))

        return pre_cnn_decoder, nn.Sequential(*cnn_transpose_layers)

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

    def forward(self, feat):
        # feat: [batch_size * num_agents, hidden_size + n_classes * n_categories]
        assert len(feat.shape) == 2

        out = self.pre_cnn_decoder(feat)
        out = self.cnn_decoder(out)
        # out += 0.5  # dreamer V3
        # loc = out.reshape(
        #     out.shape[0], -1
        # )
        output = self._build_decoder_output(out, self.cnn_keys)
        return output  # diag-Gaussian with std=1.0

from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from zsceval.algorithms.utils.util import init
from zsceval.algorithms.utils.cnn import Flatten

from .mlp import MLP


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
        obs_shape,
        encoder_output_dims=None,
        cnn_layers_params=None
    ):
        super().__init__()

        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.net = MIXDecoder(
                args,
                input_dim,
                obs_shape,
                encoder_output_dims,
                cnn_layers_params
            )
        elif len(obs_shape) == 1:
            self._mixed_obs = False
            self.net = MLP(
                args,
                input_dim=input_dim,
                hidden_dim=args.obs_hidden,
                num_layers=args.obs_layers,
                output_dim=obs_shape[0],
            )
        else:
            self._mixed_obs = False
            self.net = CNNDecoder(
                args,
                input_dim,
                obs_shape,
                encoder_output_dims['cnn'],
                cnn_layers_params=cnn_layers_params
            )

    def forward(self, h, z):
        assert len(z.shape) == 3
        z = z.reshape(z.shape[0], -1)
        input_ = torch.cat([h, z], dim=-1)
        return self.net(input_)


class MIXDecoder(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
        obs_shape,
        encoder_output_dims,
        cnn_layers_params=None
    ):
        super().__init__()
        self.cnn_keys = []
        self.local_cnn_keys = []
        self.embed_keys = []
        self.mlp_keys = []
        self.mlp_key_sizes = 0

        for key in obs_shape:
            if obs_shape[key].__class__.__name__ == "Box":
                key_obs_shape = obs_shape[key].shape
                if len(key_obs_shape) == 3:
                    if key in ["local_obs", "local_merge_obs"]:
                        self.local_cnn_keys.append(key)
                    else:
                        self.cnn_keys.append(key)
                else:
                    if "orientation" in key:
                        self.embed_keys.append(key)
                    else:
                        self.mlp_keys.append(key)
                        self.mlp_key_sizes += key_obs_shape[0]
            else:
                raise NotImplementedError

        if len(self.cnn_keys):
            self.cnn_decoder = CNNDecoder(
                args,
                input_dim,
                obs_shape['rgb'].shape,
                encoder_output_dims['cnn'],
                cnn_layers_params,
            )

        if len(self.mlp_keys) > 0:
            self.mlp_decoder = MLP(
                args,
                input_dim,
                hidden_dim=args.obs_hidden,
                num_layers=1,
                output_dim=self.mlp_key_sizes
            )

    def forward(self, x):
        cnn = self.cnn_decoder(x)
        mlp = self.mlp_decoder(x)

        return {
            'rgb': cnn,
            'mlp': mlp,
        }


class CNNDecoder(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
        obs_shape,
        cnn_output_dim,
        cnn_layers_params=None
    ):
        super().__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_maxpool2d = args.use_maxpool2d
        self.output_shape = obs_shape  # 原始观测的形状
        self.cnn_keys = ["rgb"]  # 与CNNBase保持一致

        self.input_img_shape = cnn_output_dim
        self.pre_cnn_mlp = nn.Linear(
            input_dim,
            int(np.prod(self.input_img_shape)),
        )
        # 构建解码器网络
        self.cnn_decoder = self._build_decoder_model(
            obs_shape=obs_shape,
            cnn_keys=self.cnn_keys,
            cnn_layers_params=cnn_layers_params,
            use_orthogonal=self._use_orthogonal,
            activation_id=args.activation_id
        )

    def _build_decoder_model(
        self,
        obs_shape,
        cnn_keys,
        cnn_layers_params,
        use_orthogonal,
        activation_id
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

        active_func = nn.SiLU()
        init_method = [nn.init.xavier_uniform_,
                       nn.init.orthogonal_][use_orthogonal]

        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

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
                cnn_transpose_layers.append(
                    nn.GroupNorm(
                        num_groups=1, num_channels=out_channels_next, affine=True)
                )
                cnn_transpose_layers.append(active_func)

            if self._use_maxpool2d and i < len(deconv_params) - 1:
                cnn_transpose_layers.append(
                    nn.Upsample(scale_factor=2, mode="nearest"))

        cnn_transpose_layers.append(nn.Sigmoid())  # scale to [0, 1]
        return nn.Sequential(*cnn_transpose_layers)

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
        out = self.pre_cnn_mlp(x)
        out = out.reshape(
            -1,
            *self.input_img_shape,
        )
        out = self.cnn_decoder(out)
        out = self._build_decoder_output(out, self.cnn_keys)

        loc = out.reshape([out.shape[0], -1])
        return loc  # diag-Gaussian with std=1.0

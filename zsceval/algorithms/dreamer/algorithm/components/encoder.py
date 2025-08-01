import numpy as np
import torch
import torch.nn as nn

from zsceval.algorithms.utils.util import init
from zsceval.algorithms.utils.cnn import Flatten

from .mlp import MLP


def init_(m):
    return init(
        m,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        gain=nn.init.calculate_gain("relu"),
    )


class Encoder(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None):
        super().__init__()
        self.obs_out_size = args.obs_hidden

        if "Dict" in obs_shape.__class__.__name__:
            self.net = MIXEncoder(
                args, obs_shape=obs_shape, cnn_layers_params=cnn_layers_params
            )
        elif len(obs_shape) == 1:
            self.net = MLP(
                args,
                input_dim=obs_shape[0],
                hidden_dim=args.obs_hidden,
                num_layers=args.obs_layers,
                output_dim=None,
            )
        else:
            # CNN obs only now
            self.net = CNNEncoder(args, obs_shape, cnn_layers_params)

        self.out_layer = nn.Sequential(
            init_(nn.Linear(self.net.output_size, self.obs_out_size)),
            nn.LayerNorm(self.obs_out_size),
            nn.SiLU(),
        )

    def forward(self, x):
        out = self.net(x)
        return self.out_layer(out)

    @property
    def encoder_output_dims(self):
        return self.net.output_dims


class MIXEncoder(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params):
        super().__init__()
        self.mlp_hidden_size = args.obs_hidden
        self.cnn_keys = []
        self.local_cnn_keys = []
        self.embed_keys = []
        self.mlp_keys = []

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
            else:
                raise NotImplementedError

        if len(self.cnn_keys):
            self.cnn = CNNEncoder(
                args,
                obs_shape["rgb"].shape,
                cnn_layers_params,
            )
            self.cnn_output_dim = self.cnn.cnn_output_dim

        if len(self.local_cnn_keys) > 0:
            raise NotImplementedError

        if len(self.embed_keys) > 0:
            raise NotImplementedError

        if len(self.mlp_keys) > 0:
            self.n_mlp_input = 0
            self.mlp = self._build_mlp_model(obs_shape, self.mlp_hidden_size)
            self.mlp_output_dim = self.mlp_hidden_size

    @property
    def output_size(self):
        return self.mlp_hidden_size + np.prod(self.cnn_output_dim)

    @property
    def output_dims(self):
        return {
            "mlp": self.mlp_output_dim,
            "cnn": self.cnn_output_dim,
        }

    def forward(self, x):
        if len(self.cnn_keys) > 0:
            cnn_in = x["rgb"]
            cnn_x = self.cnn(cnn_in)
            out_x = cnn_x

        if len(self.mlp_keys) > 0:
            mlp_input = self._build_mlp_input(x)
            mlp_x = self.mlp(mlp_input).view(mlp_input.size(0), -1)
            out_x = torch.cat([out_x, mlp_x], dim=-1)  # ! wrong

        return out_x

    def _build_mlp_model(self, obs_shape, hidden_size):
        active_func = nn.SiLU()

        for key in self.mlp_keys:
            self.n_mlp_input += np.prod(obs_shape[key].shape[0])

        return nn.Sequential(
            init_(nn.Linear(self.n_mlp_input, hidden_size)),
            nn.LayerNorm(hidden_size),
            active_func,
        )

    def _build_mlp_input(self, obs):
        mlp_input = []
        for key in self.mlp_keys:
            mlp_input.append(obs[key].view(obs[key].size(0), -1))

        mlp_input = torch.cat(mlp_input, dim=1)
        return mlp_input


class CNNEncoder(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params):
        super().__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self.cnn_keys = ["rgb"]
        self.obs_out_size = args.obs_hidden

        self.cnn = self._build_cnn_model(
            obs_shape,
            self.cnn_keys,
            cnn_layers_params,
            self._use_orthogonal,
            self._activation_id,
        )

    def _build_cnn_model(
        self, obs_shape, cnn_keys, cnn_layers_params, use_orthogonal, activation_id
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

        active_func = nn.SiLU()

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
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
            cnn_layers_params
        ):
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
            cnn_layers.append(
                nn.GroupNorm(
                    num_groups=1, num_channels=out_channels, affine=True)
            )
            cnn_layers.append(active_func)
            prev_out_channels = out_channels

            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(
                    [kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )

        self.cnn_output_dim = (prev_out_channels, cnn_dims[0], cnn_dims[1])
        cnn_layers += [Flatten()]
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
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def _build_cnn_input(self, obs, cnn_keys):
        cnn_input = []

        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                cnn_input.append(obs.permute(0, 3, 1, 2))
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
        return np.prod(self.cnn_output_dim)

    @property
    def output_dims(self):
        return {"cnn": self.cnn_output_dim}

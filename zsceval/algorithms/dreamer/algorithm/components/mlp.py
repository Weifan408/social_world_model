import torch.nn as nn

from zsceval.algorithms.utils.util import init
from .utils import ACTIVATION_MAP


class MLP(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim=None,
        activation=None
    ):
        super().__init__()

        init_method = nn.init.orthogonal_
        activation_id = args.activation_id
        if activation is not None:
            active_func = ACTIVATION_MAP.get(activation.lower(), nn.ReLU())
        else:
            active_func = nn.SiLU()

        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=0.01)

        layers = []
        for _ in range(num_layers):
            layers.append(
                init_(nn.Linear(input_dim, hidden_dim))
            )
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(active_func)
            input_dim = hidden_dim

        self.output_size = hidden_dim
        if output_dim is not None:
            layers.append(
                init_(nn.Linear(input_dim, output_dim))
            )
            self.output_size = output_dim
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

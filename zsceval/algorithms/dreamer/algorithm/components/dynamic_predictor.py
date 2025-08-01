import torch.nn as nn

from .mlp import MLP
from .representation_layer import RepresentationLayer


class DynamicPredictor(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.mlp = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=args.num_gru_units,
            num_layers=1,
        )

        self.representation_layer = RepresentationLayer(
            args,
            input_dim=args.num_gru_units,
        )

    def forward(self, h):
        h = self.mlp(h)
        return self.representation_layer(h)

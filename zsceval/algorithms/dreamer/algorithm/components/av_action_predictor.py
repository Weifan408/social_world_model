import torch
import torch.nn as nn
import torch.distributions as td

from .mlp import MLP


class AvailableActionPredictor(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, hidden_layers, action_size):
        super().__init__()

        self.mlp = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=hidden_layers,
            output_dim=action_size,
        )
        self.dist = lambda x: td.independent.Independent(
            td.Bernoulli(logits=x), 1)

    def forward(self, h, z):
        assert len(z.shape) == 3
        z_shape = z.shape
        z = torch.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = torch.cat([h, z], dim=-1)

        out = self.mlp(out)

        logits = out.squeeze(-1)
        dist = self.dist(logits)
        available_actions = dist.sample()

        return available_actions, dist

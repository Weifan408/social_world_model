import gym

import torch
import torch.nn as nn

from .mlp import MLP


class SequenceModel(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
        num_gru_unit,
    ):
        super().__init__()

        self.pre_gru_layer = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=num_gru_unit,
            num_layers=1,
        )

        self.gru = nn.GRU(
            input_size=num_gru_unit,
            hidden_size=num_gru_unit,
            num_layers=1,
            batch_first=True,
            bias=True
        )

    def forward(self, a, h, z):
        # Flatten last two dims of z.
        z_shape = z.shape

        z = torch.reshape(z, shape=(z_shape[0], -1))
        out = torch.cat([z, a], dim=-1)

        # Pass through pre-GRU layer.
        out = self.pre_gru_layer(out)
        # Pass through (batch-major) GRU (expand axis=1 as the time axis).
        # h: (num_layers * num_directions, B, H)
        _, h_next = self.gru(out.unsqueeze(1), h.unsqueeze(0))
        # Return the GRU's output (the next h-state). return (B, H)
        return h_next.squeeze(0)

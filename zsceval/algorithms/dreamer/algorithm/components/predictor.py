import torch
import torch.nn as nn
import torch.nn.functional as F

from zsceval.algorithms.utils.util import init
from .mlp import MLP


class MultiHeadCategoricalPredictor(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, num_heads, num_classes):
        super().__init__()
        self.args = args

        self.feature_net = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            output_dim=hidden_dim,
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_heads)]
        )

    def forward(self, h, z=None, is_first=None):
        feat = self.feature_net(h)
        logits = torch.stack([head(feat) for head in self.heads], dim=1)

        dist = torch.distributions.OneHotCategorical(logits=logits)

        return dist.sample(), logits

    # def forward_inference(self, h):
    #     inpt = torch.cat([h], dim=-1)

    #     feat = self.feature_net(inpt)
    #     logits = [head(feat) for head in self.heads]
    #     logits = torch.stack(logits, dim=1)
    #     dist = torch.distributions.OneHotCategorical(logits=logits)
    #     return dist.sample()

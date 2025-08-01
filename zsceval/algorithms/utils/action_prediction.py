import torch
import torch.nn as nn

from zsceval.algorithms.utils.act import ACTLayer
from zsceval.algorithms.utils.mlp import MLPLayer


class ActionPrediction(nn.Module):
    def __init__(self, args, input_dim, action_space, num_agents):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id

        self.backbone = MLPLayer(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            layer_N=2,
            use_orthogonal=self._use_orthogonal,
            activation_id=self._activation_id,
        )

        self.predition_head_list = nn.ModuleList(
            [
                ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain) for _ in range(num_agents)
            ]
        )

    def forward(self, feat, one_hot=False):
        feat = self.backbone(feat)
        actions = [
            head(feat, one_hot=one_hot)[0] for head in self.predition_head_list
        ]
        return torch.cat(actions, dim=-1)

    def get_action_logits(self, feat):
        feat = self.backbone(feat)
        action_logits = [
            head.get_action_logits(feat) for head in self.predition_head_list
        ]
        return torch.cat(action_logits, dim=-1)

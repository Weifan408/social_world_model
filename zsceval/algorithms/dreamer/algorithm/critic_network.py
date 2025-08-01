import torch
import torch.nn as nn

from .components.mlp import MLP
from .components.reward_predictor import RewardPredictorLayer


class CriticNetwork(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args

        self.ema_decay = args.ema_decay
        self.mlp = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=args.critic_hidden,
            num_layers=args.critic_layers,
            output_dim=None
        )
        self.return_layer = RewardPredictorLayer(
            args, input_dim=args.critic_hidden
        )

        self.mlp_ema = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=args.critic_hidden,
            num_layers=args.critic_layers,
            output_dim=None
        )
        self.return_layer_ema = RewardPredictorLayer(
            args, input_dim=args.critic_hidden
        )

        for param in self.mlp_ema.parameters():
            param.requires_grad = False
        for param in self.return_layer_ema.parameters():
            param.requires_grad = False

        self._comp_dtype = torch.get_autocast_gpu_dtype(
        ) if torch.is_autocast_enabled() else torch.float32

    def forward(self, h, z, use_ema, ids=None):
        assert len(z.shape) == 3
        z_shape = z.shape
        z = torch.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2

        if ids is not None:
            out = torch.cat([h, z, ids], dim=-1)
        else:
            out = torch.cat([h, z], dim=-1)

        if not use_ema:
            # Send h-cat-z through MLP.
            out = self.mlp(out)
            # Return expected return OR (expected return, probs of bucket values).
            return self.return_layer(out)
        else:
            out = self.mlp_ema(out)
            return self.return_layer_ema(out)

    def init_ema(self) -> None:
        vars = list(self.mlp.parameters()) + \
            list(self.return_layer.parameters())
        vars_ema = list(self.mlp_ema.parameters()) + \
            list(self.return_layer_ema.parameters())

        assert len(vars) == len(vars_ema) and len(
            vars) > 0, "Mismatch in parameter lengths"

        for var, var_ema in zip(vars, vars_ema):
            assert var is not var_ema, "Variable and EMA should not be the same object"
            var_ema.data.copy_(var.data)

    def update_ema(self) -> None:
        """Updates the EMA-copy of the critic according to the update formula:

        ema_net=(`ema_decay`*ema_net) + (1.0-`ema_decay`)*critic_net
        """
        vars = list(self.mlp.parameters()) + \
            list(self.return_layer.parameters())
        vars_ema = list(self.mlp_ema.parameters()) + \
            list(self.return_layer_ema.parameters())

        assert len(vars) == len(vars_ema) and len(
            vars) > 0, "Mismatch in parameter lengths"

        for var, var_ema in zip(vars, vars_ema):
            var_ema.data.copy_(self.ema_decay * var_ema.data +
                               (1.0 - self.ema_decay) * var.data)

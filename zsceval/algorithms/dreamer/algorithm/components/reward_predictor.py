import torch
import torch.nn as nn
import torch.nn.functional as F

from zsceval.algorithms.utils.util import init
from .mlp import MLP


class RewardPredictor(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args

        self.mlp = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=args.reward_hidden,
            num_layers=args.reward_layers,
            output_dim=None
        )
        self.reward_layer = RewardPredictorLayer(args, args.reward_hidden)
        self._comp_dtype = torch.get_autocast_gpu_dtype(
        ) if torch.is_autocast_enabled() else torch.float32

    def forward(self, h, z):
        assert len(z.shape) == 3
        z_shape = z.shape
        z = torch.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = torch.cat([h, z], dim=-1)

        # Send h-cat-z through MLP.
        out = self.mlp(out)
        # Return a) mean reward OR b) a tuple: (mean reward, logits over the reward
        # buckets).
        return self.reward_layer(out)


class RewardPredictorLayer(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args

        self.lower_bound = args.lower_bound
        self.upper_bound = args.upper_bound
        self.num_buckets = args.num_buckets

        init_method = nn.init.orthogonal_
        activation_id = args.activation_id
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.reward_buckets_layer = init_(
            nn.Linear(input_dim, self.num_buckets)
        )

    def forward(self, x):
        assert len(x.shape) == 2
        logits = self.reward_buckets_layer(x)
        probs = F.softmax(logits, dim=-1)
        possible_outcomes = torch.linspace(
            self.lower_bound, self.upper_bound, self.num_buckets
        ).to(probs.device)
        expected_rewards = torch.sum(probs*possible_outcomes, dim=-1)

        return expected_rewards, logits

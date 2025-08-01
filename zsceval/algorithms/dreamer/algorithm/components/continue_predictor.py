import torch
import torch.nn as nn

from .mlp import MLP


class ContinuePredictor(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.mlp = MLP(
            args,
            input_dim=input_dim,
            hidden_dim=args.continue_hidden,
            num_layers=args.continue_layers,
            output_dim=1,
        )

    def forward(self, h, z):
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z_shape = z.shape
        z = torch.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = torch.cat([h, z], dim=-1)
        # Send h-cat-z through MLP.
        out = self.mlp(out)
        # Remove the extra [B, 1] dimension at the end to get a proper Bernoulli
        # distribution. Otherwise, tfp will think that the batch dims are [B, 1]
        # where they should be just [B].
        logits = out.squeeze(-1)

        # Take the mode (greedy, deterministic "sample").
        continue_ = (logits > 0).to(torch.float32)

        # Return Bernoulli sample (whether to continue) OR (continue?, Bernoulli prob).
        return continue_, logits

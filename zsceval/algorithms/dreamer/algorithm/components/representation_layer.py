import torch
import torch.nn as nn
import torch.nn.functional as F

from zsceval.algorithms.utils.util import init


class RepresentationLayer(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.num_categoricals = args.n_categoricals
        self.num_classes_per_categorical = args.n_classes

        init_method = nn.init.orthogonal_
        activation_id = args.activation_id

        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.z_generating_layer = init_(
            nn.Linear(
                input_dim,
                self.num_categoricals * self.num_classes_per_categorical
            )
        )

        self._comp_dtype = torch.get_autocast_gpu_dtype(
        ) if torch.is_autocast_enabled() else torch.float32

    def forward(self, x):
        logits = self.z_generating_layer(x)
        logits = torch.reshape(
            logits,
            shape=(
                -1,
                self.num_categoricals,
                self.num_classes_per_categorical
            ),
        )
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        probs = 0.99 * probs + 0.01 * (1.0 / self.num_classes_per_categorical)
        logits = torch.log(probs)

        distribution = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(logits=logits),
            reinterpreted_batch_ndims=1,
        )
        sample = distribution.sample()

        differentiable_sample = sample.detach() + probs - probs.detach()
        return differentiable_sample.to(self._comp_dtype), probs

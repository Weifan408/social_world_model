from gym.spaces import Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.mlp import MLP


class ActorNetwork(nn.Module):
    def __init__(
        self,
        args,
        input_dim,
        action_space,
    ):
        super().__init__()

        self.action_space = action_space
        self.ema_value_target_pct5 = nn.Parameter(
            torch.tensor(float("nan")), requires_grad=False
        )
        self.ema_value_target_pct95 = nn.Parameter(
            torch.tensor(float("nan")), requires_grad=False
        )

        if isinstance(self.action_space, Discrete):
            self.mlp = MLP(
                args,
                input_dim=input_dim,
                hidden_dim=args.actor_hidden,
                num_layers=args.actor_layers,
                output_dim=action_space.n,
            )
        else:
            raise NotImplementedError()

        self._comp_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else torch.float32
        )

    def forward(self, h, z, available_actions, ids):
        assert len(z.shape) == 3
        z_shape = z.shape
        z = torch.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2

        if ids is not None:
            out = torch.cat([h, z, ids], dim=-1)
        else:
            out = torch.cat([h, z], dim=-1)

        # Send h-cat-z through MLP.
        action_logits = self.mlp(out).to(self._comp_dtype)

        if available_actions is not None:
            action_logits[available_actions == 0] = torch.finfo(
                action_logits.dtype).min

        if isinstance(self.action_space, Discrete):
            action_probs = F.softmax(action_logits, dim=-1)

            # Add the unimix weighting (1% uniform) to the probs.
            # See [1]: "Unimix categoricals: We parameterize the categorical
            # distributions for the world model representations and dynamics, as well as
            # for the actor network, as mixtures of 1% uniform and 99% neural network
            # output to ensure a minimal amount of probability mass on every class and
            # thus keep log probabilities and KL divergences well behaved."
            action_probs = 0.99 * action_probs + \
                0.01 * (1.0 / self.action_space.n)

            # Danijar's code does: distr = [Distr class](logits=tf.log(probs)).
            # Not sure why we don't directly use the already available probs instead.
            action_logits = torch.log(action_probs)

            # Distribution parameters are the log(probs) directly.
            distr_params = action_logits
            distr = self.get_action_dist_object(distr_params)

            action = distr.sample() + (action_probs - action_probs.detach())

        else:
            raise NotImplementedError()

        return action, distr_params

    def get_action_dist_object(self, action_dist_params_T_B):
        """Helper method to create an action distribution object from (T, B, ..) params.

        Args:
            action_dist_params_T_B: The time-major action distribution parameters.
                This could be simply the logits (discrete) or a to-be-split-in-2
                tensor for mean and stddev (continuous).

        Returns:
            The tfp action distribution object, from which one can sample, compute
            log probs, entropy, etc..
        """
        if isinstance(self.action_space, Discrete):
            # Create the distribution object using the unimix'd logits.
            distr = torch.distributions.OneHotCategorical(
                logits=action_dist_params_T_B,
            )
        else:
            raise ValueError(
                f"Action space {self.action_space} not supported!")

        return distr

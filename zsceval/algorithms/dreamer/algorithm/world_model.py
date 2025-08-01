import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.utils import symlog
from .components.mlp import MLP
from .components.reward_predictor import RewardPredictor
from .components.representation_layer import RepresentationLayer
from .components.dynamic_predictor import DynamicPredictor
from .components.continue_predictor import ContinuePredictor
from .components.sequence_model import SequenceModel
from .components.predictor import MultiHeadCategoricalPredictor
from .components.vqvae import VQVAEPolicyIDPredictor
from .components.av_action_predictor import AvailableActionPredictor


from zsceval.utils.util import check


class WorldModel(nn.Module):
    def __init__(
        self,
        args,
        action_space,
        encoder,
        decoder,
        symlog_obs=True,
    ):
        super().__init__()

        # self.args = args

        self.img_obs = args.image_obs
        self.batch_length_T = args.batch_length_T
        self.symlog_obs = symlog_obs
        self.data_parallel = getattr(args, "data_parallel", False)
        self.use_vqvae = args.use_vqvae

        self._comp_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else torch.float32
        )

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_size = action_space.n
        else:
            raise NotImplementedError()

        # xt -> lt
        self.encoder = encoder

        # [ht, lt] -> zt.
        self.posterior_mlp = MLP(
            args,
            input_dim=args.num_gru_units + args.obs_hidden,
            hidden_dim=args.num_gru_units,
            num_layers=1,
        )

        self.posterior_representation_layer = RepresentationLayer(
            args, input_dim=args.num_gru_units
        )

        # ht -> z^t
        self.dynamics_predictor = DynamicPredictor(
            args, input_dim=args.num_gru_units)

        self.initial_h = nn.Parameter(
            torch.zeros(args.num_gru_units), requires_grad=True
        )
        self.sequence_model = SequenceModel(
            args,
            input_dim=args.feat_size + self.action_size,
            num_gru_unit=args.num_gru_units,
        )

        self.reward_predictor = RewardPredictor(
            args, args.feat_size + args.num_gru_units
        )
        self.continue_predictor = ContinuePredictor(
            args, args.feat_size + args.num_gru_units
        )

        self.av_predictor = AvailableActionPredictor(
            args,
            input_dim=args.feat_size + args.num_gru_units,
            hidden_dim=args.continue_hidden,
            hidden_layers=args.continue_layers,
            action_size=self.action_size,
        )

        # h -> ids
        if self.use_vqvae:
            self.policy_id_predictor = VQVAEPolicyIDPredictor(
                args=args,
                input_dim=args.num_gru_units,
                num_heads=args.num_agents-1,
                num_classes=args.population_size + 1,
                num_embeddings=16,
                embedding_dim=args.num_gru_units,
                commitment_cost=0.1,
            )
        else:
            self.policy_id_predictor = MultiHeadCategoricalPredictor(
                args,
                input_dim=args.num_gru_units,
                hidden_dim=args.obs_hidden,
                num_heads=args.num_agents - 1,
                num_classes=args.population_size, # +1  grf
            )

        self.decoder = decoder

    def get_initial_state(self):
        # [1, num_gru_units]
        h = F.tanh(self.initial_h.to(self._comp_dtype)).unsqueeze(0)
        _, z_probs = self.dynamics_predictor(h)
        z = torch.argmax(z_probs, dim=-1)
        z = F.one_hot(z, num_classes=z_probs.shape[-1]).to(self._comp_dtype)

        return {"h": h, "z": z}

    def forward_inference(self, observations, previous_states, is_first):

        initial_states = {
            k: torch.repeat_interleave(v, is_first.shape[0], dim=0)
            for k, v in self.get_initial_state().items()
        }

        # If first, mask it with initial state/actions.
        previous_h = self._mask(
            previous_states["h"], 1.0 - is_first)  # zero out
        previous_h = previous_h + \
            self._mask(initial_states["h"], is_first)  # add init

        previous_z = self._mask(
            previous_states["z"], 1.0 - is_first)  # zero out
        previous_z = previous_z + \
            self._mask(initial_states["z"], is_first)  # add init

        # Zero out actions (no special learnt initial state).
        previous_a = self._mask(previous_states["a"], 1.0 - is_first)

        # Compute new states.
        h = self.sequence_model(a=previous_a, h=previous_h, z=previous_z)
        z = self.compute_posterior_z(observations=observations, initial_h=h)
        if self.use_vqvae:
            out_dict = self.policy_id_predictor(h)
            ids = out_dict['quantized_st']
        else:
            ids, _ = self.policy_id_predictor(h)

        return {"h": h, "z": z, "ids": ids}

    def forward_train(self, observations, actions, is_first):
        if self.symlog_obs:
            observations = symlog(observations)
        elif self.img_obs:
            observations = observations / 255.0

        B, T = actions.shape[:2]

        if isinstance(observations, dict):
            observations = {
                k: v.reshape(-1, *v.shape[2:]) for k, v in observations.items()
            }
        else:
            observations = observations.reshape(-1, *observations.shape[2:])

        encoder_out = self.encoder(observations)
        encoder_out = encoder_out.reshape(B, T, *encoder_out.shape[1:])

        # time major
        encoder_out = encoder_out.permute(1, 0, *range(2, encoder_out.dim()))
        initial_states = {
            k: v.repeat_interleave(B, dim=0)
            for k, v in self.get_initial_state().items()
        }

        # time major
        actions = actions.permute(1, 0, *range(2, actions.dim()))
        actions = F.one_hot(
            actions.long(), num_classes=self.action_size).squeeze(-2)
        is_first = is_first.permute(1, 0).to(self._comp_dtype)

        z_t0_to_T = [initial_states["z"]]
        z_posterior_probs = []
        z_prior_probs = []
        h_t0_to_T = [initial_states["h"]]
        for t in range(self.batch_length_T):
            # If first, mask it with initial state/actions.
            h_tm1 = self._mask(h_t0_to_T[-1], 1.0 - is_first[t])  # zero out
            h_tm1 = h_tm1 + \
                self._mask(initial_states["h"], is_first[t])  # add init

            z_tm1 = self._mask(z_t0_to_T[-1], 1.0 - is_first[t])  # zero out
            z_tm1 = z_tm1 + \
                self._mask(initial_states["z"], is_first[t])  # add init

            # Zero out actions (no special learnt initial state).
            a_tm1 = self._mask(actions[t - 1], 1.0 - is_first[t])

            # Perform one RSSM (sequence model) step to get the current h.
            h_t = self.sequence_model(a=a_tm1, h=h_tm1, z=z_tm1)
            h_t0_to_T.append(h_t)

            posterior_mlp_input = torch.cat([encoder_out[t], h_t], dim=-1)
            repr_input = self.posterior_mlp(posterior_mlp_input)
            # Draw one z-sample (z(t)) and also get the z-distribution for dynamics and
            # representation loss computations.
            z_t, z_probs = self.posterior_representation_layer(repr_input)
            # z_t=[B, num_categoricals, num_classes]
            z_posterior_probs.append(z_probs)
            z_t0_to_T.append(z_t)

            # Compute the predicted z_t (z^) using the dynamics model.
            _, z_probs = self.dynamics_predictor(h_t)
            z_prior_probs.append(z_probs)

        # Stack at time dimension to yield: [B, T, ...].
        h_t1_to_T = torch.stack(h_t0_to_T[1:], dim=1)
        z_t1_to_T = torch.stack(z_t0_to_T[1:], dim=1)

        # Fold time axis to retrieve the final (loss ready) Independent distribution
        # (over `num_categoricals` Categoricals).
        z_posterior_probs = torch.stack(z_posterior_probs, dim=1)
        z_posterior_probs = z_posterior_probs.reshape(
            [-1, *z_posterior_probs.shape[2:]]
        )
        # Fold time axis to retrieve the final (loss ready) Independent distribution
        # (over `num_categoricals` Categoricals).
        z_prior_probs = torch.stack(z_prior_probs, dim=1)
        z_prior_probs = z_prior_probs.reshape([-1, *z_prior_probs.shape[2:]])

        # Fold time dimension for parallelization of all dependent predictions:
        # observations (reproduction via decoder), rewards, continues.
        h_BxT = h_t1_to_T.reshape([-1, *h_t1_to_T.shape[2:]])
        z_BxT = z_t1_to_T.reshape([-1, *z_t1_to_T.shape[2:]])

        obs_distribution_means = self.decoder(h=h_BxT, z=z_BxT)

        # Compute (predicted) reward distributions.
        rewards, reward_logits = self.reward_predictor(h=h_BxT, z=z_BxT)

        avas, av_dist = self.av_predictor(h=h_BxT, z=z_BxT)
        if self.use_vqvae:
            vq_dict = self.policy_id_predictor(h=h_BxT)
            policy_ids_BxT = vq_dict['quantized_st']
            policy_ids_logits_BxT = vq_dict['policy_ids_logits']
            vq_loss = vq_dict['vq_loss']
            perplexity = vq_dict['perplexity']
        else:
            policy_ids_BxT, policy_ids_logits_BxT = self.policy_id_predictor(
                h=h_BxT)
            perplexity = torch.tensor(0)
            vq_loss = torch.tensor(0)

        # Compute (predicted) continue distributions.
        continues, continue_logits = self.continue_predictor(h=h_BxT, z=z_BxT)

        # Return outputs for loss computation.
        # Note that all shapes are [BxT, ...] (time axis already folded).
        return {
            # Obs.
            "sampled_obs_symlog_BxT": observations,
            "obs_distribution_means_BxT": obs_distribution_means,
            # Rewards.
            "reward_logits_BxT": reward_logits,
            "rewards_BxT": rewards,
            # Continues.
            "continue_logits_BxT": continue_logits,
            "continues_BxT": continues,
            # Deterministic, continuous h-states (t1 to T).
            "h_states_BxT": h_BxT,
            # Sampled, discrete posterior z-states and their probs (t1 to T).
            "z_posterior_states_BxT": z_BxT,
            "z_posterior_probs_BxT": z_posterior_probs,
            # Probs of the prior z-states (t1 to T).
            "z_prior_probs_BxT": z_prior_probs,
            "policy_ids_BxT": policy_ids_BxT,
            "policy_ids_logits_BxT": policy_ids_logits_BxT,
            "perplexity": perplexity,
            "vq_loss": vq_loss,
            "available_actions_BxT": avas,
            "av_dist_BxT": av_dist,
        }

    def compute_posterior_z(self, observations, initial_h):
        # Compute bare encoder outputs (not including z, which is computed in next step
        # with involvement of the previous output (initial_h) of the sequence model).
        # encoder_outs=[B, ...]
        if self.symlog_obs:
            observations = symlog(observations)
        elif self.img_obs:
            observations = observations / 255.0
        encoder_out = self.encoder(observations)
        # Concat encoder outs with the h-states.
        posterior_mlp_input = torch.cat([encoder_out, initial_h], dim=-1)
        # Compute z.
        repr_input = self.posterior_mlp(posterior_mlp_input)
        # Draw a z-sample.
        z_t, _ = self.posterior_representation_layer(repr_input)
        return z_t

    def _mask(self, value, mask):
        return torch.einsum("b...,b->b...", value, mask)

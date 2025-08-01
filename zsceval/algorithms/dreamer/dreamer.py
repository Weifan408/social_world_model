from collections import defaultdict, Counter
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from zsceval.utils.util import get_gard_norm, mse_loss
from .algorithm.components.utils import symlog, two_hot, to_torch


class ExDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Dreamer:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device

        if args.data_parallel:
            self.policy = ExDataParallel(policy)
        else:
            self.policy = policy

        self.args = args
        self._use_max_grad_norm = args.use_max_grad_norm

        self.batch_size_B = args.batch_size_B
        self.batch_length_T = args.batch_length_T
        self.batch_data_num = self.batch_size_B * self.batch_length_T

        self.entropy_scale = args.entropy_scale
        self.entropy_annealing = args.entropy_annealing

        self.gae_lambda = args.gae_lambda
        self.gamma = args.gamma
        self.return_normalization_decay = args.return_normalization_decay

        self.world_model_grad_clip_by_global_norm = (
            args.world_model_grad_clip_by_global_norm
        )
        self.critic_grad_clip_by_global_norm = args.critic_grad_clip_by_global_norm
        self.actor_grad_clip_by_global_norm = args.actor_grad_clip_by_global_norm

        self.trianing_ratio = args.training_ratio

    def train(self, buffer):
        train_info = dict()

        replayed_steps_this_iter = 0
        total_sample_cost_time = 0

        s_time = time.time()
        total_sampled_steps = buffer.last_sampled_timesteps

        cnt = 0
        while replayed_steps_this_iter / total_sampled_steps < self.trianing_ratio:
            cnt += 1
            sample_st = time.time()
            sample = buffer.sample(
                batch_size_B=self.batch_size_B,
                batch_length_T=self.batch_length_T,
            )
            total_sample_cost_time += time.time() - sample_st

            replayed_steps_this_iter += self.batch_data_num

            metrics = self.train_model(sample)
            for k, v in metrics.items():
                if k not in train_info:
                    train_info[k] = v
                else:
                    train_info[k] += v

        for k, v in train_info.items():
            if isinstance(train_info[k], Counter):
                for key, val in train_info[k].items():
                    train_info[k][key] = val / cnt
            else:
                train_info[k] = v / cnt

        e_time = time.time()
        logger.info(
            f"Sample {replayed_steps_this_iter} steps cost {total_sample_cost_time:.2f}s"
        )
        logger.info(f"Training {cnt} iterations cost {e_time - s_time:.2f}s")
        return train_info

    def train_model(self, sample):
        sample = to_torch(sample, device=self.device)

        fwd_out = self.policy(sample)
        prediction_losses = self._compute_world_model_prediction_losses(
            rewards_B_T=sample["rewards"],
            available_actions=sample["available_actions"],
            continues_B_T=1.0 - sample["is_terminated"].to(torch.float32),
            policy_ids_B_T=sample["others_policy_ids"],
            fwd_out=fwd_out,
        )

        (
            L_dyn_B_T,
            L_rep_B_T,
        ) = self._compute_world_model_dynamics_and_representation_loss(fwd_out=fwd_out)

        L_dyn = L_dyn_B_T.mean()
        L_rep = L_rep_B_T.mean()

        # Make sure values for L_rep and L_dyn are the same (they only differ in their
        # gradients).
        assert torch.equal(L_dyn, L_rep)

        L_world_model_total_B_T = (
            1.0 * prediction_losses["L_prediction_B_T"]
            + 0.5 * L_dyn_B_T
            + 0.1 * L_rep_B_T
        )
        L_world_model_total = L_world_model_total_B_T.mean() + \
            fwd_out['vq_loss']
        wm_grad_norm = self.apply_optimizer(
            self.policy.world_model_optimizer,
            self.policy.world_model,
            L_world_model_total,
            self.world_model_grad_clip_by_global_norm,
        )

        wm_metrics = {
            "WM/L_world_model_total": L_world_model_total.item(),
            "WM/L_decoder": prediction_losses["L_decoder"].item(),
            "WM/L_reward": prediction_losses["L_reward"].item(),
            "WM/L_continue": prediction_losses["L_continue"].item(),
            "WM/L_vq": fwd_out['vq_loss'].item(),
            "WM/L_prediction_B_T": prediction_losses["L_prediction_B_T"].mean().item(),
            "WM/L_ids": prediction_losses["L_ids"].item(),
            "WM/L_av": prediction_losses["L_av"].item(),
            "WM/L_dyn_B_T": L_dyn_B_T.mean().item(),
            "WM/L_rep_B_T": L_rep_B_T.mean().item(),
        }

        dream_data = self.policy.dreamer_model.dream_trajectory(
            start_states={
                "h": fwd_out["h_states_BxT"],
                "z": fwd_out["z_posterior_states_BxT"],
                "ava": fwd_out["available_actions_BxT"],
                "ids": fwd_out["policy_ids_BxT"],
            },
            start_is_terminated=torch.reshape(sample["is_terminated"], [-1]),
        )

        value_targets_t0_to_Hm1_BxT = self._compute_value_targets(
            # Learn critic in symlog'd space.
            rewards_t0_to_H_BxT=dream_data["rewards_dreamed_t0_to_H_BxT"].detach(
            ),
            continues_t0_to_H_BxT=dream_data["continues_dreamed_t0_to_H_BxT"].detach(
            ),
            value_predictions_t0_to_H_BxT=dream_data["values_dreamed_t0_to_H_BxT"],
        )
        CRITIC_L_total, critic_metrics = self._compute_critic_loss(
            dream_data=dream_data,
            value_targets_t0_to_Hm1_BxT=value_targets_t0_to_Hm1_BxT,
        )
        critic_grad_norm = self.apply_optimizer(
            self.policy.critic_optimizer,
            self.policy.critic,
            CRITIC_L_total,
            self.critic_grad_clip_by_global_norm,
        )

        ACTOR_L_total, actor_metrics = self._compute_actor_loss(
            dream_data=dream_data,
            value_targets_t0_to_Hm1_BxT=value_targets_t0_to_Hm1_BxT,
        )
        actor_grad_norm = self.apply_optimizer(
            self.policy.actor_optimizer,
            self.policy.actor,
            ACTOR_L_total,
            self.actor_grad_clip_by_global_norm,
        )

        self.policy.critic.update_ema()
        self.entropy_scale = max(
            self.entropy_scale * self.entropy_annealing, 5e-4)
        # print("perplexity:", fwd_out["perplexity"].item())
        # print("Counter: ", Counter(fwd_out["encoding_indices_BxT"].tolist()))
        return {
            "WM/WM_grad_norm": wm_grad_norm.item(),
            "CRITIC/CRITIC_grad_norm": critic_grad_norm.item(),
            "perplexity": fwd_out["perplexity"].item(),
            # "encoding_indices": Counter(fwd_out["encoding_indices_BxT"].tolist()),
            "ACTOR/ACTOR_grad_norm": actor_grad_norm.item(),
            "ACTOR/entropy_scale": self.entropy_scale,
            **wm_metrics,
            **actor_metrics,
            **critic_metrics,
        }

    def _compute_world_model_prediction_losses(
        self,
        *,
        rewards_B_T,
        available_actions,
        continues_B_T,
        policy_ids_B_T,
        fwd_out,
    ):
        """Helper method computing all world-model related prediction losses.

        Prediction losses are used to train the predictors of the world model, which
        are: Reward predictor, continue predictor, and the decoder (which predicts
        observations).

        Args:
            config: The DreamerV3Config to use.
            rewards_B_T: The rewards batch in the shape (B, T) and of type float32.
            continues_B_T: The continues batch in the shape (B, T) and of type float32
                (1.0 -> continue; 0.0 -> end of episode).
            fwd_out: The `forward_train` outputs of the DreamerV3RLModule.
        """

        # Learn to produce symlog'd observation predictions.
        # If symlog is disabled (e.g. for uint8 image inputs), `obs_symlog_BxT` is the
        # same as `obs_BxT`.
        obs_BxT = fwd_out["sampled_obs_symlog_BxT"]
        obs_distr_means = fwd_out["obs_distribution_means_BxT"]
        # In case we wanted to construct a distribution object from the fwd out data,
        # we would have to do it like this:
        # obs_distr = tfp.distributions.MultivariateNormalDiag(
        #    loc=obs_distr_means,
        #    # Scale == 1.0.
        #    # [2]: "Distributions The image predictor outputs the mean of a diagonal
        #    # Gaussian likelihood with **unit variance** ..."
        #    scale_diag=tf.ones_like(obs_distr_means),
        # )

        # Leave time dim folded (BxT) and flatten all other (e.g. image) dims.
        if isinstance(obs_BxT, dict):
            obs_BxT = {
                k: v.reshape(-1, np.prod(v.shape[1:])) for k, v in obs_BxT.items()
            }
            image_loss_BxT = F.binary_cross_entropy_with_logits(
                input=obs_distr_means["rgb"], target=obs_BxT["rgb"], reduction="none"
            ).mean(-1)
            mlp_loss_BxT = F.mse_loss(
                input=obs_distr_means["mlp"],
                target=obs_BxT["timestep"],
                reduction="none",
            ).mean(-1)
            decoder_loss_BxT = image_loss_BxT + mlp_loss_BxT
        else:
            obs_BxT = torch.reshape(
                obs_BxT, shape=[-1, np.prod(obs_BxT.shape[1:])])

            # Squared diff loss w/ sum(!) over all (already folded) obs dims.
            # decoder_loss_BxT = SUM[ (obs_distr.loc - observations)^2 ]
            # Note: This is described strangely in the paper (stating a neglogp loss here),
            # but the author's own implementation actually uses simple MSE with the loc
            # of the Gaussian.
            decoder_loss_BxT = F.mse_loss(
                input=obs_distr_means, target=obs_BxT, reduction="none"
            ).sum(-1)

        # Unfold time rank back in.
        decoder_loss_B_T = decoder_loss_BxT.reshape(
            [self.batch_size_B, self.batch_length_T]
        )

        L_decoder = decoder_loss_B_T.mean()

        # The FiniteDiscrete reward bucket distribution computed by our reward
        # predictor.
        # [B x num_buckets].
        reward_logits_BxT = fwd_out["reward_logits_BxT"]
        # Learn to produce symlog'd reward predictions.
        rewards_symlog_B_T = symlog(rewards_B_T)
        # Fold time dim.
        rewards_symlog_BxT = rewards_symlog_B_T.view(-1)

        # Two-hot encode.
        two_hot_rewards_symlog_BxT = two_hot(
            rewards_symlog_BxT,
            num_buckets=self.args.num_buckets,
            lower_bound=self.args.lower_bound,
            upper_bound=self.args.upper_bound,
        )
        # two_hot_rewards_symlog_BxT=[B*T, num_buckets]
        reward_log_pred_BxT = reward_logits_BxT - torch.logsumexp(
            reward_logits_BxT, dim=-1, keepdims=True
        )
        # Multiply with two-hot targets and neg.
        reward_loss_two_hot_BxT = -(
            reward_log_pred_BxT * two_hot_rewards_symlog_BxT
        ).sum(-1)
        # Unfold time rank back in.
        reward_loss_two_hot_B_T = torch.reshape(
            reward_loss_two_hot_BxT,
            (self.batch_size_B, self.batch_length_T),
        )
        L_reward_two_hot = reward_loss_two_hot_B_T.mean()

        # Probabilities that episode continues, computed by our continue predictor.
        # [B]
        continue_distr = torch.distributions.Bernoulli(
            logits=fwd_out["continue_logits_BxT"]
        )
        # -log(p) loss
        # Fold time dim.
        continues_BxT = continues_B_T.view(-1)
        continue_loss_BxT = -continue_distr.log_prob(continues_BxT)
        # Unfold time rank back in.
        continue_loss_B_T = continue_loss_BxT.reshape(
            self.batch_size_B, self.batch_length_T
        )
        L_continue = continue_loss_B_T.mean()

        if available_actions is not None:
            available_actions_BxT = available_actions.reshape(
                self.batch_size_B * self.batch_length_T, -1
            )
            av_dist = fwd_out["av_dist_BxT"]
            av_loss = -av_dist.log_prob(available_actions_BxT).reshape(
                self.batch_size_B, self.batch_length_T
            )
        else:
            av_loss = torch.tensor(0)
        L_av = av_loss.mean()

        policy_ids_logits_BxT = fwd_out["policy_ids_logits_BxT"]
        ids_dist = torch.distributions.Categorical(
            logits=policy_ids_logits_BxT.view(-1,
                                              policy_ids_logits_BxT.shape[-1])
        )
        policy_ids_targets = policy_ids_B_T.view(-1)
        ids_loss_BxT = -ids_dist.log_prob(policy_ids_targets)
        ids_loss_B_T = ids_loss_BxT.reshape(
            self.batch_size_B, self.batch_length_T, -1
        ).mean(dim=-1)
        L_ids = ids_loss_B_T.mean()

        # Sum all losses together as the "prediction" loss.
        L_pred_B_T = (
            decoder_loss_B_T
            + reward_loss_two_hot_B_T
            + continue_loss_B_T
            + ids_loss_B_T
            + av_loss
        )
        L_pred = L_pred_B_T.mean()

        return {
            # "L_decoder_B_T": decoder_loss_B_T,
            "L_decoder": L_decoder,
            "L_reward": L_reward_two_hot,
            # "L_reward_B_T": reward_loss_two_hot_B_T,
            "L_continue": L_continue,
            # "L_continue_B_T": continue_loss_B_T,
            "L_prediction": L_pred,
            "L_prediction_B_T": L_pred_B_T,
            "L_ids": L_ids,
            "L_av": L_av,
        }

    def _compute_world_model_dynamics_and_representation_loss(self, *, fwd_out):
        """Helper method computing the world-model's dynamics and representation losses.

        Args:
            config: The DreamerV3Config to use.
            fwd_out: The `forward_train` outputs of the DreamerV3RLModule.

        Returns:
            Tuple consisting of a) dynamics loss: Trains the prior network, predicting
            z^ prior states from h-states and b) representation loss: Trains posterior
            network, predicting z posterior states from h-states and (encoded)
            observations.
        """

        # Actual distribution over stochastic internal states (z) produced by the
        # encoder.
        z_posterior_probs_BxT = fwd_out["z_posterior_probs_BxT"]
        z_posterior_distr_BxT = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(probs=z_posterior_probs_BxT),
            reinterpreted_batch_ndims=1,
        )

        # Actual distribution over stochastic internal states (z) produced by the
        # dynamics network.
        z_prior_probs_BxT = fwd_out["z_prior_probs_BxT"]
        z_prior_distr_BxT = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(probs=z_prior_probs_BxT),
            reinterpreted_batch_ndims=1,
        )

        # Stop gradient for encoder's z-outputs:
        sg_z_posterior_distr_BxT = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(
                probs=z_posterior_probs_BxT.detach()),
            reinterpreted_batch_ndims=1,
        )
        # Stop gradient for dynamics model's z-outputs:
        sg_z_prior_distr_BxT = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(
                probs=z_prior_probs_BxT.detach()),
            reinterpreted_batch_ndims=1,
        )

        # Implement free bits. According to [1]:
        # "To avoid a degenerate solution where the dynamics are trivial to predict but
        # contain not enough information about the inputs, we employ free bits by
        # clipping the dynamics and representation losses below the value of
        # 1 nat ≈ 1.44 bits. This disables them while they are already minimized well to
        # focus the world model on its prediction loss"

        L_dyn_BxT = self._safe_kl_divergence(
            sg_z_posterior_distr_BxT, z_prior_distr_BxT
        )

        # Unfold time rank back in.
        L_dyn_B_T = L_dyn_BxT.reshape(self.batch_size_B, self.batch_length_T)

        L_rep_BxT = self._safe_kl_divergence(
            z_posterior_distr_BxT, sg_z_prior_distr_BxT
        )

        # Unfold time rank back in.
        L_rep_B_T = L_rep_BxT.reshape(self.batch_size_B, self.batch_length_T)

        return L_dyn_B_T, L_rep_B_T

    def _safe_kl_divergence(self, posterior, prior):
        kl_values = torch.distributions.kl_divergence(posterior, prior)
        kl_values = torch.where(
            torch.isfinite(kl_values),
            kl_values,
            torch.tensor(1.0, device=kl_values.device),
        )
        return torch.clamp(kl_values, min=1.0)

    def _compute_actor_loss(
        self,
        *,
        dream_data,
        value_targets_t0_to_Hm1_BxT,
    ):
        """Helper method computing the actor's loss terms.

        Args:
            module_id: The module_id for which to compute the actor loss.
            config: The DreamerV3Config to use.
            dream_data: The data generated by dreaming for H steps (horizon) starting
                from any BxT state (sampled from the buffer for the train batch).
            value_targets_t0_to_Hm1_BxT: The computed value function targets of the
                shape (t0 to H-1, BxT).

        Returns:
            The total actor loss tensor.
        """
        actor = self.policy.actor

        # Note: `scaled_value_targets_t0_to_Hm1_B` are NOT stop_gradient'd yet.
        scaled_value_targets_t0_to_Hm1_B = self._compute_scaled_value_targets(
            value_targets_t0_to_Hm1_BxT=value_targets_t0_to_Hm1_BxT,
            value_predictions_t0_to_Hm1_BxT=dream_data["values_dreamed_t0_to_H_BxT"][
                :-1
            ],
        )

        # Actions actually taken in the dream.
        actions_dreamed = dream_data["actions_dreamed_t0_to_H_BxT"].detach()[
            :-1]
        actions_dreamed_dist_params_t0_to_Hm1_B = dream_data[
            "actions_dreamed_dist_params_t0_to_H_BxT"
        ][:-1]

        dist_t0_to_Hm1_B = actor.get_action_dist_object(
            actions_dreamed_dist_params_t0_to_Hm1_B
        )

        # Compute log(p)s of all possible actions in the dream.
        if isinstance(actor.action_space, gym.spaces.Discrete):
            # Note that when we create the Categorical action distributions, we compute
            # unimix probs, then math.log these and provide these log(p) as "logits" to
            # the Categorical. So here, we'll continue to work with log(p)s (not
            # really "logits")!
            logp_actions_t0_to_Hm1_B = actions_dreamed_dist_params_t0_to_Hm1_B

            # Log probs of actions actually taken in the dream.
            logp_actions_dreamed_t0_to_Hm1_B = (
                actions_dreamed * logp_actions_t0_to_Hm1_B
            ).sum(-1)
            # First term of loss function. [1] eq. 11.
            logp_loss_H_B = (
                logp_actions_dreamed_t0_to_Hm1_B
                * scaled_value_targets_t0_to_Hm1_B.detach()
            )

        # Box space.
        else:
            pass
            # logp_actions_dreamed_t0_to_Hm1_B = dist_t0_to_Hm1_B.log_prob(
            #     actions_dreamed
            # )
            # # First term of loss function. [1] eq. 11.
            # logp_loss_H_B = scaled_value_targets_t0_to_Hm1_B

        assert len(logp_loss_H_B.shape) == 2

        # Add entropy loss term (second term [1] eq. 11).
        entropy_H_B = dist_t0_to_Hm1_B.entropy()
        assert len(entropy_H_B.shape) == 2
        entropy = entropy_H_B.mean()

        L_actor_reinforce_term_H_B = -logp_loss_H_B
        L_actor_action_entropy_term_H_B = -self.entropy_scale * entropy_H_B

        L_actor_H_B = L_actor_reinforce_term_H_B + L_actor_action_entropy_term_H_B
        # Mask out everything that goes beyond a predicted continue=False boundary.
        L_actor_H_B *= dream_data["dream_loss_weights_t0_to_H_BxT"].detach()[
            :-1]
        L_actor = L_actor_H_B.mean()

        return L_actor, {
            "ACTOR/ACTOR_L_total": L_actor.item(),
            "ACTOR/ACTOR_value_targets_pct95_ema": actor.ema_value_target_pct95.item(),
            "ACTOR/ACTOR_value_targets_pct5_ema": actor.ema_value_target_pct5.item(),
            "ACTOR/ACTOR_action_entropy": entropy.item(),
            # Individual loss terms.
            "ACTOR/ACTOR_L_neglogp_reinforce_term": L_actor_reinforce_term_H_B.mean().item(),
            "ACTOR/ACTOR_L_neg_entropy_term": L_actor_action_entropy_term_H_B.mean().item(),
        }

    def _compute_critic_loss(
        self,
        *,
        dream_data,
        value_targets_t0_to_Hm1_BxT,
    ):
        """Helper method computing the critic's loss terms.

        Args:
            module_id: The ModuleID for which to compute the critic loss.
            config: The DreamerV3Config to use.
            dream_data: The data generated by dreaming for H steps (horizon) starting
                from any BxT state (sampled from the buffer for the train batch).
            value_targets_t0_to_Hm1_BxT: The computed value function targets of the
                shape (t0 to H-1, BxT).

        Returns:
            The total critic loss tensor.
        """
        # B=BxT
        H, B = dream_data["rewards_dreamed_t0_to_H_BxT"].shape[:2]
        Hm1 = H - 1

        # Note that value targets are NOT symlog'd and go from t0 to H-1, not H, like
        # all the other dream data.

        # From here on: B=BxT
        value_targets_t0_to_Hm1_B = value_targets_t0_to_Hm1_BxT.detach()
        value_symlog_targets_t0_to_Hm1_B = symlog(value_targets_t0_to_Hm1_B)
        # Fold time rank (for two_hot'ing).
        value_symlog_targets_HxB = value_symlog_targets_t0_to_Hm1_B.view(-1)
        value_symlog_targets_two_hot_HxB = two_hot(
            value_symlog_targets_HxB,
            num_buckets=self.args.num_buckets,
            lower_bound=self.args.lower_bound,
            upper_bound=self.args.upper_bound,
        )
        # Unfold time rank.
        value_symlog_targets_two_hot_t0_to_Hm1_B = (
            value_symlog_targets_two_hot_HxB.reshape(
                Hm1, B, value_symlog_targets_two_hot_HxB.shape[-1]
            )
        )

        # Get (B x T x probs) tensor from return distributions.
        value_symlog_logits_HxB = dream_data["values_symlog_dreamed_logits_t0_to_HxBxT"]
        # Unfold time rank and cut last time index to match value targets.
        value_symlog_logits_t0_to_Hm1_B = torch.reshape(
            value_symlog_logits_HxB,
            shape=[H, B, value_symlog_logits_HxB.shape[-1]],
        )[:-1]

        values_log_pred_Hm1_B = value_symlog_logits_t0_to_Hm1_B - torch.logsumexp(
            value_symlog_logits_t0_to_Hm1_B, dim=-1, keepdims=True
        )
        # Multiply with two-hot targets and neg.
        value_loss_two_hot_H_B = -torch.sum(
            values_log_pred_Hm1_B * value_symlog_targets_two_hot_t0_to_Hm1_B, dim=-1
        )

        # Compute EMA regularization loss.
        # Expected values (dreamed) from the EMA (slow critic) net.
        # Note: Slow critic (EMA) outputs are already stop_gradient'd.
        value_symlog_ema_t0_to_Hm1_B = dream_data[
            "v_symlog_dreamed_ema_t0_to_H_BxT"
        ].detach()[:-1]
        # Fold time rank (for two_hot'ing).
        value_symlog_ema_HxB = value_symlog_ema_t0_to_Hm1_B.view(-1)
        value_symlog_ema_two_hot_HxB = two_hot(
            value_symlog_ema_HxB,
            num_buckets=self.args.num_buckets,
            lower_bound=self.args.lower_bound,
            upper_bound=self.args.upper_bound,
        )
        # Unfold time rank.
        value_symlog_ema_two_hot_t0_to_Hm1_B = torch.reshape(
            value_symlog_ema_two_hot_HxB,
            shape=[Hm1, B, value_symlog_ema_two_hot_HxB.shape[-1]],
        )

        # Compute ema regularizer loss.
        # In the paper, it is not described how exactly to form this regularizer term
        # and how to weigh it.
        # So we follow Danijar's repo here:
        # `reg = -dist.log_prob(sg(self.slow(traj).mean()))`
        # with a weight of 1.0, where dist is the bucket'ized distribution output by the
        # fast critic. sg=stop gradient; mean() -> use the expected EMA values.
        # Multiply with two-hot targets and neg.
        ema_regularization_loss_H_B = -torch.sum(
            values_log_pred_Hm1_B * value_symlog_ema_two_hot_t0_to_Hm1_B, dim=-1
        )

        L_critic_H_B = value_loss_two_hot_H_B + ema_regularization_loss_H_B

        # Mask out everything that goes beyond a predicted continue=False boundary.
        L_critic_H_B *= dream_data["dream_loss_weights_t0_to_H_BxT"].detach()[
            :-1]

        # Reduce over both H- (time) axis and B-axis (mean).
        L_critic = L_critic_H_B.mean()

        return L_critic, {
            "CRITIC/CRITIC_L_total": L_critic.item(),
            "CRITIC/CRITIC_L_neg_logp_of_value_targets": value_loss_two_hot_H_B.mean().item(),
            "CRITIC/CRITIC_L_slow_critic_regularization": ema_regularization_loss_H_B.mean().item(),
        }

    def _compute_value_targets(
        self,
        *,
        rewards_t0_to_H_BxT,
        continues_t0_to_H_BxT,
        value_predictions_t0_to_H_BxT,
    ):
        """Helper method computing the value targets.

        All args are (H, BxT, ...) and in non-symlog'd (real) reward space.
        Non-symlog is important b/c log(a+b) != log(a) + log(b).
        See [1] eq. 8 and 10.
        Thus, targets are always returned in real (non-symlog'd space).
        They need to be re-symlog'd before computing the critic loss from them (b/c the
        critic produces predictions in symlog space).
        Note that the original B and T ranks together form the new batch dimension
        (folded into BxT) and the new time rank is the dream horizon (hence: [H, BxT]).

        Variable names nomenclature:
        `H`=1+horizon_H (start state + H steps dreamed),
        `BxT`=batch_size * batch_length (meaning the original trajectory time rank has
        been folded).

        Rewards, continues, and value predictions are all of shape [t0-H, BxT]
        (time-major), whereas returned targets are [t0 to H-1, B] (last timestep missing
        b/c the target value equals vf prediction in that location anyways.

        Args:
            config: The DreamerV3Config to use.
            rewards_t0_to_H_BxT: The reward predictor's predictions over the
                dreamed trajectory t0 to H (and for the batch BxT).
            intrinsic_rewards_t1_to_H_BxT: The predicted intrinsic rewards over the
                dreamed trajectory t0 to H (and for the batch BxT).
            continues_t0_to_H_BxT: The continue predictor's predictions over the
                dreamed trajectory t0 to H (and for the batch BxT).
            value_predictions_t0_to_H_BxT: The critic's value predictions over the
                dreamed trajectory t0 to H (and for the batch BxT).

        Returns:
            The value targets in the shape: [t0toH-1, BxT]. Note that the last step (H)
            does not require a value target as it matches the critic's value prediction
            anyways.
        """
        # The first reward is irrelevant (not used for any VF target).
        rewards_t1_to_H_BxT = rewards_t0_to_H_BxT[1:]

        # In all the following, when building value targets for t=1 to T=H,
        # exclude rewards & continues for t=1 b/c we don't need r1 or c1.
        # The target (R1) for V1 is built from r2, c2, and V2/R2.
        discount = continues_t0_to_H_BxT[1:] * self.gamma  # shape=[2-16, BxT]
        Rs = [value_predictions_t0_to_H_BxT[-1]]  # Rs indices=[16]
        intermediates = (
            rewards_t1_to_H_BxT
            + discount * (1 - self.gae_lambda) *
            value_predictions_t0_to_H_BxT[1:]
        )
        # intermediates.shape=[2-16, BxT]

        # Loop through reversed timesteps (axis=1) from T+1 to t=2.
        for t in reversed(range(discount.shape[0])):
            Rs.append(intermediates[t] + discount[t]
                      * self.gae_lambda * Rs[-1])

        # Reverse along time axis and cut the last entry (value estimate at very end
        # cannot be learnt from as it's the same as the ... well ... value estimate).
        targets_t0toHm1_BxT = torch.stack(list(reversed(Rs))[:-1], dim=0)
        # targets.shape=[t0 to H-1,BxT]

        return targets_t0toHm1_BxT

    def _compute_scaled_value_targets(
        self,
        *,
        value_targets_t0_to_Hm1_BxT,
        value_predictions_t0_to_Hm1_BxT,
    ):
        """Helper method computing the scaled value targets.

        Args:
            module_id: The module_id to compute value targets for.
            config: The DreamerV3Config to use.
            value_targets_t0_to_Hm1_BxT: The value targets computed by
                `self._compute_value_targets` in the shape of (t0 to H-1, BxT)
                and of type float32.
            value_predictions_t0_to_Hm1_BxT: The critic's value predictions over the
                dreamed trajectories (w/o the last timestep). The shape of this
                tensor is (t0 to H-1, BxT) and the type is float32.

        Returns:
            The scaled value targets used by the actor for REINFORCE policy updates
            (using scaled advantages). See [1] eq. 12 for more details.
        """
        actor = self.policy.actor

        value_targets_H_B = value_targets_t0_to_Hm1_BxT
        value_predictions_H_B = value_predictions_t0_to_Hm1_BxT

        # Compute S: [1] eq. 12.
        Per_R_5 = torch.quantile(value_targets_H_B, 0.05)
        Per_R_95 = torch.quantile(value_targets_H_B, 0.95)

        # Update EMA values for 5 and 95 percentile, stored as tf variables under actor
        # network.
        # 5 percentile
        new_val_pct5 = torch.where(
            torch.isnan(actor.ema_value_target_pct5),
            # is NaN: Initial values: Just set.
            Per_R_5,
            # Later update (something already stored in EMA variable): Update EMA.
            (
                self.return_normalization_decay * actor.ema_value_target_pct5
                + (1.0 - self.return_normalization_decay) * Per_R_5
            ),
        )
        actor.ema_value_target_pct5.data.copy_(new_val_pct5)
        # 95 percentile
        new_val_pct95 = torch.where(
            torch.isnan(actor.ema_value_target_pct95),
            # is NaN: Initial values: Just set.
            Per_R_95,
            # Later update (something already stored in EMA variable): Update EMA.
            (
                self.return_normalization_decay * actor.ema_value_target_pct95
                + (1.0 - self.return_normalization_decay) * Per_R_95
            ),
        )
        actor.ema_value_target_pct95.data.copy_(new_val_pct95)

        # [1] eq. 11 (first term).
        offset = actor.ema_value_target_pct5
        invscale = torch.clamp(
            actor.ema_value_target_pct95 - actor.ema_value_target_pct5, min=1e-8
        )
        scaled_value_targets_H_B = (value_targets_H_B - offset) / invscale
        scaled_value_predictions_H_B = (
            value_predictions_H_B - offset) / invscale

        # Return advantages.
        return scaled_value_targets_H_B - scaled_value_predictions_H_B

    def apply_optimizer(self, opt, model, loss, grad_norm):
        opt.zero_grad()
        loss.backward()
        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        else:
            grad_norm = get_gard_norm(model.parameters())
        opt.step()
        return grad_norm

    def to(self, device):
        self.policy.to(device)

    def prep_training(self):
        self.policy.prep_training()

    def prep_rollout(self):
        self.policy.prep_rollout()

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.utils import inverse_symlog


class DreamerModel(nn.Module):
    def __init__(
        self,
        args,
        actor,
        critic,
        world_model,
        device="cpu",
    ):
        super().__init__()

        self.world_model = world_model
        self.actor = actor
        self.critic = critic

        self.horizon = args.horizon_H
        self.gamma = args.gamma

        self._comp_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else torch.float32
        )
        self.device = device

    def forward_inference(
        self, observations, previous_states, is_first, available_actions=None
    ):
        states = self.world_model.forward_inference(
            observations=observations,
            previous_states=previous_states,
            is_first=is_first,
        )
        # Compute action using our actor network and the current states.
        _, distr_params = self.actor(
            h=states["h"],
            z=states["z"],
            available_actions=available_actions,
            ids=states["ids"].reshape(available_actions.shape[0], -1),
        )
        # Use the mode of the distribution (Discrete=argmax, Normal=mean).
        distr = self.actor.get_action_dist_object(distr_params)
        actions = distr.mode
        return actions, {"h": states["h"], "z": states["z"], "a": actions}

    def forward_exploration(
        self, observations, previous_states, is_first, available_actions=None
    ):
        states = self.world_model.forward_inference(
            observations=observations,
            previous_states=previous_states,
            is_first=is_first,
        )
        # Compute action using our actor network and the current states.
        actions, _ = self.actor(
            h=states["h"],
            z=states["z"],
            available_actions=available_actions,
            ids=states["ids"].reshape(available_actions.shape[0], -1),
        )
        return actions, {"h": states["h"], "z": states["z"], "a": actions}

    def forward_train(self, observations, actions, is_first):
        return self.world_model.forward_train(
            observations=observations,
            actions=actions,
            is_first=is_first,
        )

    def forward_imagine(
        self, observations, previous_states, is_first, available_actions=None
    ):
        states = self.world_model.forward_inference(
            observations=observations,
            previous_states=previous_states,
            is_first=is_first,
        )
        
        return self.imagine(
            start_states={
                "obs": observations,
                "h": states["h"],
                "z": states["z"],
                "ava": available_actions,
                "ids": states["ids"],
            }
        )
    
    def get_initial_state(self):
        """Returns the (current) initial state of the dreamer model (a, h-, z-states).

        An initial state is generated using the previous action, the tanh of the
        (learned) h-state variable and the dynamics predictor (or "prior net") to
        compute z^0 from h0. In this last step, it is important that we do NOT sample
        the z^-state (as we would usually do during dreaming), but rather take the mode
        (argmax, then one-hot again).
        """
        states = self.world_model.get_initial_state()

        action_space = self.actor.action_space
        action_dim = (
            action_space.n
            if isinstance(action_space, gym.spaces.Discrete)
            else np.prod(action_space.shape)
        )
        states["a"] = torch.zeros(
            (
                1,
                action_dim,
            ),
            dtype=self._comp_dtype,
        ).to(self.device)
        return states

    def dream_trajectory(self, start_states, start_is_terminated):
        """Dreams trajectories of length H from batch of h- and z-states.

        Note that incoming data will have the shapes (BxT, ...), where the original
        batch- and time-dimensions are already folded together. Beginning from this
        new batch dim (BxT), we will unroll `timesteps_H` timesteps in a time-major
        fashion, such that the dreamed data will have shape (H, BxT, ...).

        Args:
            start_states: Dict of `h` and `z` states in the shape of (B, ...) and
                (B, num_categoricals, num_classes), respectively, as
                computed by a train forward pass. From each individual h-/z-state pair
                in the given batch, we will branch off a dreamed trajectory of len
                `timesteps_H`.
            start_is_terminated: Float flags of shape (B,) indicating whether the
                first timesteps of each batch row is already a terminated timestep
                (given by the actual environment).
        """
        # Dreamed actions (one-hot encoded for discrete actions).
        a_dreamed_t0_to_H = []
        a_dreamed_dist_params_t0_to_H = []

        h = start_states["h"]
        z = start_states["z"]
        init_ids = start_states["ids"]
        ava = start_states["ava"]

        # GRU outputs.
        h_states_t0_to_H = [h]
        # Dynamics model outputs.
        z_states_prior_t0_to_H = [z]
        init_ids = init_ids.detach().reshape(init_ids.shape[0], -1)

        # Compute `a` using actor network (already the first step uses a dreamed action,
        # not a sampled one).
        a, a_dist_params = self.actor(
            # We have to stop the gradients through the states. B/c we are using a
            # differentiable Discrete action distribution (straight through gradients
            # with `a = stop_gradient(sample(probs)) + probs - stop_gradient(probs)`,
            # we otherwise would add dependencies of the `-log(pi(a|s))` REINFORCE loss
            # term on actions further back in the trajectory.
            h=h.detach(),
            z=z.detach(),
            available_actions=ava.detach(),
            ids=init_ids,
        )
        a_dreamed_t0_to_H.append(a)
        a_dreamed_dist_params_t0_to_H.append(a_dist_params)

        for i in range(self.horizon):
            # Move one step in the dream using the RSSM.
            h = self.world_model.sequence_model(a=a, h=h, z=z)
            h_states_t0_to_H.append(h)

            # Compute prior z using dynamics model.
            z, _ = self.world_model.dynamics_predictor(h=h)
            z_states_prior_t0_to_H.append(z)

            ava, _ = self.world_model.av_predictor(h, z)
            # ids, _ = self.world_model.policy_id_predictor(h)
            # Compute `a` using actor network.
            a, a_dist_params = self.actor(
                h=h.detach(),
                z=z.detach(),
                available_actions=ava.detach(),
                ids=init_ids,
            )
            a_dreamed_t0_to_H.append(a)
            a_dreamed_dist_params_t0_to_H.append(a_dist_params)

        h_states_H_B = torch.stack(h_states_t0_to_H, dim=0)  # (T, B, ...)
        h_states_HxB = h_states_H_B.reshape([-1, *h_states_H_B.shape[2:]])

        z_states_prior_H_B = torch.stack(z_states_prior_t0_to_H, dim=0)  # (T, B, ...)
        z_states_prior_HxB = z_states_prior_H_B.reshape(
            [-1, *z_states_prior_H_B.shape[2:]]
        )

        a_dreamed_H_B = torch.stack(a_dreamed_t0_to_H, dim=0)  # (T, B, ...)
        a_dreamed_dist_params_H_B = torch.stack(a_dreamed_dist_params_t0_to_H, dim=0)

        # Compute r using reward predictor.
        r_dreamed_HxB, _ = self.world_model.reward_predictor(
            h=h_states_HxB, z=z_states_prior_HxB
        )

        r_dreamed_H_B = torch.reshape(
            inverse_symlog(r_dreamed_HxB), shape=[self.horizon + 1, -1]
        )

        # Compute continues using continue predictor.
        c_dreamed_HxB, _ = self.world_model.continue_predictor(
            h=h_states_HxB,
            z=z_states_prior_HxB,
        )
        c_dreamed_H_B = torch.reshape(c_dreamed_HxB, [self.horizon + 1, -1])
        # Force-set first `continue` flags to False iff `start_is_terminated`.
        # Note: This will cause the loss-weights for this row in the batch to be
        # completely zero'd out. In general, we don't use dreamed data past any
        # predicted (or actual first) continue=False flags.
        c_dreamed_H_B = torch.cat(
            [
                1.0 - start_is_terminated.to(self._comp_dtype).unsqueeze(0),
                c_dreamed_H_B[1:],
            ],
            dim=0,
        )

        # Loss weights for each individual dreamed timestep. Zero-out all timesteps
        # that lie past continue=False flags. B/c our world model does NOT learn how
        # to skip terminal/reset episode boundaries, dreamed data crossing such a
        # boundary should not be used for critic/actor learning either.
        dream_loss_weights_H_B = (
            torch.cumprod(self.gamma * c_dreamed_H_B, dim=0) / self.gamma
        )

        # Compute the value estimates.
        v, v_symlog_dreamed_logits_HxB = self.critic(
            h=h_states_HxB.detach(),
            z=z_states_prior_HxB.detach(),
            use_ema=False,
            ids=init_ids.repeat(self.horizon + 1, 1),  # (HxB, N)
        )
        v_dreamed_HxB = inverse_symlog(v)
        v_dreamed_H_B = torch.reshape(v_dreamed_HxB, shape=[self.horizon + 1, -1])

        v_symlog_dreamed_ema_HxB, _ = self.critic(
            h=h_states_HxB.detach(),
            z=z_states_prior_HxB.detach(),
            use_ema=True,
            ids=init_ids.repeat(self.horizon + 1, 1),  # (HxB, N)
        )
        v_symlog_dreamed_ema_H_B = torch.reshape(
            v_symlog_dreamed_ema_HxB, shape=[self.horizon + 1, -1]
        )

        ret = {
            "h_states_t0_to_H_BxT": h_states_H_B,
            "z_states_prior_t0_to_H_BxT": z_states_prior_H_B,
            "rewards_dreamed_t0_to_H_BxT": r_dreamed_H_B,
            "continues_dreamed_t0_to_H_BxT": c_dreamed_H_B,
            "actions_dreamed_t0_to_H_BxT": a_dreamed_H_B,
            "actions_dreamed_dist_params_t0_to_H_BxT": a_dreamed_dist_params_H_B,
            "values_dreamed_t0_to_H_BxT": v_dreamed_H_B,
            "values_symlog_dreamed_logits_t0_to_HxBxT": v_symlog_dreamed_logits_HxB,
            "v_symlog_dreamed_ema_t0_to_H_BxT": v_symlog_dreamed_ema_H_B,
            # Loss weights for critic- and actor losses.
            "dream_loss_weights_t0_to_H_BxT": dream_loss_weights_H_B,
        }

        if isinstance(self.actor.action_space, gym.spaces.Discrete):
            ret["actions_ints_dreamed_t0_to_H_B"] = torch.argmax(a_dreamed_H_B, dim=-1)

        return ret

    def imagine(self, start_states):
        a_dreamed_t0_to_H = []
        a_dreamed_dist_params_t0_to_H = []

        h = start_states["h"]
        z = start_states["z"]
        init_ids = start_states["ids"]
        ava = start_states["ava"]
        

        h_states_t0_to_H = [h]
        z_states_prior_t0_to_H = [z]
        if self.world_model.img_obs:
            imagine_obs = [start_states['obs'].reshape(init_ids.shape[0], -1) / 255 ]
        else:
            imagine_obs = [start_states['obs'].reshape(init_ids.shape[0], -1)]
        init_ids = init_ids.detach().reshape(init_ids.shape[0], -1)

        a, a_dist_params = self.actor(
            h=h.detach(),
            z=z.detach(),
            available_actions=ava.detach(),
            ids=init_ids,
        )
        a_dreamed_t0_to_H.append(a)
        a_dreamed_dist_params_t0_to_H.append(a_dist_params)

        for i in range(self.horizon):
            # Move one step in the dream using the RSSM.
            h = self.world_model.sequence_model(a=a, h=h, z=z)
            h_states_t0_to_H.append(h)

            # Compute prior z using dynamics model.
            z, _ = self.world_model.dynamics_predictor(h=h)
            z_states_prior_t0_to_H.append(z)
            imagine_obs.append(self.world_model.decoder(h=h, z=z))
            ava, _ = self.world_model.av_predictor(h, z)
            # ids, _ = self.world_model.policy_id_predictor(h)
            # Compute `a` using actor network.
            a, a_dist_params = self.actor(
                h=h.detach(),
                z=z.detach(),
                available_actions=ava.detach(),
                ids=init_ids,
            )
            a_dreamed_t0_to_H.append(a)
            a_dreamed_dist_params_t0_to_H.append(a_dist_params)

        imagine_obs_H_B = torch.stack(imagine_obs, dim=0) # (T, B, ...)
        return imagine_obs_H_B.permute(1, 0, 2)
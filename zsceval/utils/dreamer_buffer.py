from collections import defaultdict, deque

import numpy as np
import torch

from zsceval.utils.util import get_shape_from_act_space, get_shape_from_obs_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2, 3).reshape(-1, *x.shape[2:])


class Epiosde:
    def __init__(
        self,
        args,
        num_other_agents,
        observation_space,
        share_obs_space,
        action_space,
    ):
        self.args = args
        self.action_space = action_space
        self.observation_space = observation_space
        self.share_obs_space = share_obs_space
        self.episode_length = args.episode_length
        self.num_other_agents = num_other_agents

        self._mixed_obs = False  # for mixed observation

        obs_shape = get_shape_from_obs_space(observation_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # for mixed observation
        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros(
                    (
                        self.episode_length + 1,
                        *obs_shape[key].shape,
                    ),
                    dtype=np.float32,
                )
        else:
            # deal with special attn format
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            self.obs = np.zeros(
                (
                    self.episode_length + 1,
                    *obs_shape,
                ),
                dtype=np.float32,
            )

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]
        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )

        if action_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    action_space.n,
                ),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(action_space)

        self.actions = np.zeros(
            (self.episode_length, act_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length),
            dtype=np.float32,
        )
        self.is_terminated = False
        self.others_actions = np.zeros(
            (
                self.episode_length,
                act_shape * num_other_agents,
            ),
            dtype=np.float32,
        )
        self.step = 0

    def __len__(self):
        return self.episode_length

    def insert_reset(
        self,
        share_obs,
        obs,
        available_action=None,
    ):
        self.share_obs[self.step] = share_obs.copy()
        if self._mixed_obs:
            for key in self.obs.keys():
                self.obs[key][self.step] = obs[key].copy()
        else:
            self.obs[self.step] = obs.copy()
        if available_action is not None:
            self.available_actions[self.step] = available_action.copy()

    def insert(
        self,
        share_obs,
        obs,
        actions,
        others_actions,
        rewards,
        is_terminated,
        available_actions=None,
    ):
        if self._mixed_obs:
            for key in self.obs.keys():
                self.obs[key][self.step + 1] = obs[key].copy()
        else:
            self.obs[self.step + 1] = obs.copy()

        self.share_obs[self.step + 1] = share_obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.item()
        self.is_terminated = is_terminated

        if others_actions is not None:
            self.others_actions[self.step] = others_actions.copy()

        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    @property
    def avg_reward(self):
        return np.mean(self.rewards)

    def after_update(self):
        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][0] = self.share_obs[key][-1].copy()
            for key in self.obs.keys():
                self.obs[key][0] = self.obs[key][-1].copy()
        else:
            self.share_obs[0] = self.share_obs[-1].copy()
            self.obs[0] = self.obs[-1].copy()

        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()


class DreamerReplayBuffer:
    def __init__(
        self,
        args,
        num_other_agents,
        observation_space,
        share_obs_space,
        action_space,
        n_rollout_threads,
    ):
        self.capacity = args.capacity
        self.batch_size_B = args.batch_size_B
        self.batch_length_T = args.batch_length_T
        self.n_rollout_threads = n_rollout_threads

        self.episodes = deque()
        self.sampling_episodes = [
            Epiosde(
                args,
                num_other_agents,
                observation_space,
                share_obs_space,
                action_space,
            )
            for _ in range(n_rollout_threads)
        ]
        self.recent_avg_rewards = []
        self._num_timesteps = 0
        self._last_sampled_timesteps = 0
        self._num_timesteps_added = 0

        if "Dict" in observation_space.__class__.__name__:
            self._mixed_obs = True
            self.obs_keys = observation_space.keys()
        else:
            self._mixed_obs = False

        self.sampled_timesteps = 0
        self.rng = np.random.default_rng(args.seed)
        self.last_obs = None
        self._last_ava = None

    def __len__(self):
        return self._num_timesteps

    def reset_sampling_episodes(
        self,
        args,
        num_other_agents,
        observation_space,
        share_obs_space,
        action_space,
        n_rollout_threads,
    ):
        self.sampling_episodes = [
            Epiosde(
                args,
                num_other_agents,
                observation_space,
                share_obs_space,
                action_space,
            )
            for _ in range(n_rollout_threads)
        ]

    @property
    def last_sampled_timesteps(self):
        return self._last_sampled_timesteps

    def insert_policy_id(self, policy_ids, n=1):
        for i in range(len(policy_ids)):
            if not hasattr(self.sampling_episodes[i], "other_policy_id"):
                self.sampling_episodes[i].other_policy_id = np.zeros(
                    (self.sampling_episodes[i].episode_length + 1, n),
                    dtype=np.int32,
                )
                self.sampling_episodes[i].other_policy_id[:] = policy_ids[i]

    def insert_reset(
        self,
        share_obs,
        obs,
        available_actions=None,
    ):
        self._last_obs = obs
        self._last_ava = available_actions
        for i in range(len(obs)):
            self.sampling_episodes[i].insert_reset(
                share_obs[i],
                obs[i],
                available_actions[i] if available_actions is not None else None,
            )

    def obs(self):
        if self._mixed_obs:
            merged = {}
            for k in self._last_obs[0].keys():
                merged[k] = np.stack([obs[k]
                                     for obs in self._last_obs], axis=0)
            return merged
        else:
            return self._last_obs

    def ava(self):
        return self._last_ava

    @property
    def rewards(self):
        return self.recent_avg_rewards

    def insert(
        self,
        share_obs,
        obs,
        actions,
        others_actions,
        rewards,
        is_terminated,
        available_actions=None,
    ):
        self._last_obs = obs
        self._last_ava = available_actions

        for i in range(len(obs)):
            self.sampling_episodes[i].insert(
                share_obs=share_obs[i],
                obs=obs[i],
                actions=actions[i],
                others_actions=(
                    others_actions[i] if others_actions is not None else None
                ),
                rewards=rewards[i],
                is_terminated=is_terminated[i],
                available_actions=(
                    available_actions[i] if available_actions is not None else None
                ),
            )

    def torch_episode_len_limit(self):
        self._num_timesteps += self.sampling_episodes[0].episode_length * len(
            self.sampling_episodes
        )
        self._num_timesteps_added += self.sampling_episodes[0].episode_length * len(
            self.sampling_episodes
        )
        self._last_sampled_timesteps = self.sampling_episodes[0].episode_length * len(
            self.sampling_episodes
        )
        for episode in self.sampling_episodes:
            self.episodes.append(episode)
            self.recent_avg_rewards.append(episode.avg_reward)

        while self._num_timesteps > self.capacity and len(self.episodes) > 1:
            evicted_eps = self.episodes.popleft()
            evicted_eps_len = len(evicted_eps)
            self._num_timesteps -= evicted_eps_len

    def after_update(self):
        self._last_sampled_timesteps = 0
        self.recent_avg_rewards = []
        # for i in range(len(self.sampling_episodes)):
        #     self.sampling_episodes[i].after_update()

    def sample(self, batch_size_B=None, batch_length_T=None):
        batch_size_B = batch_size_B or self.batch_size_B
        batch_length_T = batch_length_T or self.batch_length_T

        if self._mixed_obs:
            observations = {k: [[] for _ in range(
                batch_size_B)] for k in self.obs_keys}
        else:
            observations = [[] for _ in range(batch_size_B)]
        actions = [[] for _ in range(batch_size_B)]
        others_policy_ids = [[] for _ in range(batch_size_B)]
        rewards = [[] for _ in range(batch_size_B)]
        share_observations = [[] for _ in range(batch_size_B)]
        available_actions = [[] for _ in range(batch_size_B)]
        is_first = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_last = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_terminated = [[False] * batch_length_T for _ in range(batch_size_B)]

        B = 0
        T = 0
        ep_count = len(self.episodes)
        assert ep_count >= 1, "No episodes to sample from!"
        # ep_idx = self.rng.integers(0, ep_count, size=B)

        while B < batch_size_B:
            episode = self.episodes[self.rng.integers(len(self.episodes))]
            episode_ts = self.rng.integers(len(episode) - self.batch_length_T)

            # Starting a new chunk, set is_first to True.
            is_first[B][T] = True

            # Begin of new batch item (row).
            if len(rewards[B]) == 0:
                # And we are at the start of an episode: Set reward to 0.0.
                if episode_ts == 0:
                    rewards[B].append(0.0)
                # We are in the middle of an episode: Set reward to the previous
                # timestep's values.
                else:
                    rewards[B].append(episode.rewards[episode_ts - 1])
            # We are in the middle of a batch item (row). Concat next episode to this
            # row from the next episode's beginning. In other words, we never concat
            # a middle of an episode to another truncated one.
            else:
                episode_ts = 0
                rewards[B].append(0.0)

            if self._mixed_obs:
                for key in self.obs_keys:
                    observations[key][B].extend(episode.obs[key][episode_ts:])
            else:
                observations[B].extend(episode.obs[episode_ts:])
            share_observations[B].extend(episode.share_obs[episode_ts:])
            available_actions[B].extend(episode.available_actions[episode_ts:])
            # Repeat last action to have the same number of actions than observations.
            actions[B].extend(episode.actions[episode_ts:])
            actions[B].append(episode.actions[-1])
            others_policy_ids[B].extend(episode.other_policy_id[episode_ts:])
            # Number of rewards are also the same as observations b/c we have the
            # initial 0.0 one.
            rewards[B].extend(episode.rewards[episode_ts:])
            assert len(actions[B]) == len(rewards[B])

            T = min(len(rewards[B]), batch_length_T)

            # Set is_last=True.
            is_last[B][T - 1] = True
            # If episode is terminated and we have reached the end of it, set
            # is_terminated=True.
            if episode.is_terminated and T == len(rewards[B]):
                is_terminated[B][T - 1] = True

            # We are done with this batch row.
            if T == batch_length_T:
                # We may have overfilled this row: Clip trajectory at the end.
                if self._mixed_obs:
                    for key in self.obs_keys:
                        observations[key][B] = observations[key][B][:batch_length_T]
                else:
                    observations[B] = observations[B][:batch_length_T]
                share_observations[B] = share_observations[B][:batch_length_T]
                available_actions[B] = available_actions[B][:batch_length_T]
                actions[B] = actions[B][:batch_length_T]
                rewards[B] = rewards[B][:batch_length_T]
                others_policy_ids[B] = np.array(
                    others_policy_ids[B][:batch_length_T])
                # Start filling the next row.
                B += 1
                T = 0

        self.sampled_timesteps += batch_size_B * batch_length_T

        if self._mixed_obs:
            observations = {k: np.array(v) for k, v in observations.items()}
        else:
            observations = np.array(observations)

        ret = {
            "obs": observations,
            "share_obs": np.array(share_observations),
            "actions": np.array(actions),
            "available_actions": np.array(available_actions),
            "rewards": np.array(rewards),
            "is_first": np.array(is_first),
            "is_last": np.array(is_last),
            "is_terminated": np.array(is_terminated),
            "others_policy_ids": np.array(others_policy_ids),
        }
        return ret

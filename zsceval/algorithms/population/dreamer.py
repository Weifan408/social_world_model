import copy
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from loguru import logger

from zsceval.algorithms.population.policy_pool import PolicyPool
from zsceval.algorithms.population.mep import MEP_Trainer
from zsceval.algorithms.population.utils import _t2n
from zsceval.utils.dreamer_buffer import DreamerReplayBuffer
from zsceval.utils.shared_buffer import SharedReplayBuffer
from zsceval.utils.util import convert_to_tensor


class Dreamer_Trainer(MEP_Trainer):
    def __init__(self, args, policy_pool: PolicyPool, device=torch.device("cpu")):
        super().__init__(args, policy_pool, device)
        self.stage = args.stage

        self.replay_buffer = None

    def reset(
        self,
        map_ea2t,
        n_rollout_threads,
        num_agents,
        load_unused_to_cpu=False,
        **kwargs,
    ):
        self.map_ea2t = map_ea2t
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents
        self._states = None

        self.control_agent_count = defaultdict(int)
        self.control_agents = defaultdict(list)
        for (e, a), trainer_name in self.map_ea2t.items():
            self.control_agent_count[trainer_name] += 1
            self.control_agents[trainer_name].append((e, a))

        self.active_trainers = []
        self.buffer_pool: Dict[str, DreamerReplayBuffer] = {}
        for trainer_name, trainer in self.trainer_pool.items():
            # set n_rollout_threads as control_agent_count[trainer_name] and num_agents as 1
            if self.control_agent_count[trainer_name] > 0:
                policy_args, obs_space, share_obs_space, act_space = self.policy_config(
                    trainer_name
                )
                if trainer_name in self.on_training:
                    if self.replay_buffer is None:
                        self.replay_buffer = DreamerReplayBuffer(
                            policy_args,
                            self.num_agents - 1,
                            obs_space,
                            share_obs_space,
                            act_space,
                            n_rollout_threads=self.control_agent_count[trainer_name],
                        )
                    else:
                        self.replay_buffer.reset_sampling_episodes(
                            policy_args,
                            self.num_agents - 1,
                            obs_space,
                            share_obs_space,
                            act_space,
                            n_rollout_threads=self.control_agent_count[trainer_name],
                        )
                else:
                    self.buffer_pool[trainer_name] = SharedReplayBuffer(
                        policy_args,
                        1,
                        obs_space,
                        share_obs_space,
                        act_space,
                        n_rollout_threads=self.control_agent_count[trainer_name],
                    )

                self.trainer_pool[trainer_name].to(self.device)
                self.active_trainers.append(trainer_name)
            else:
                if load_unused_to_cpu:
                    self.trainer_pool[trainer_name].to(torch.device("cpu"))
                else:
                    self.trainer_pool[trainer_name].to(self.device)
                self.buffer_pool[trainer_name] = None
        self.__initialized = True

    def init_first_step(
        self,
        share_obs: np.ndarray,
        obs: np.ndarray,
        available_actions: np.ndarray = None,
    ):
        assert self.__initialized
        for trainer_name in self.active_trainers:
            # extract corresponding (e, a) and add num_agent=1 dimension
            obs_lst = self.extract_elements(trainer_name, obs)
            share_obs_lst = np.expand_dims(
                self.extract_elements(trainer_name, share_obs), axis=1
            )
            if trainer_name in self.on_training:
                obs_lst = self.extract_elements(trainer_name, obs)
                share_obs_lst = self.extract_elements(trainer_name, share_obs)

                if available_actions is not None:
                    available_actions_lst = self.extract_elements(
                        trainer_name, available_actions
                    )
                else:
                    available_actions_lst = None

                self.replay_buffer.insert_reset(
                    share_obs_lst, obs_lst, available_actions_lst
                )
            else:
                obs_lst = np.expand_dims(
                    self.extract_elements(trainer_name, obs), axis=1
                )
                share_obs_lst = np.expand_dims(
                    self.extract_elements(trainer_name, share_obs), axis=1
                )
                self.buffer_pool[trainer_name].share_obs[0] = share_obs_lst.copy()
                self.buffer_pool[trainer_name].obs[0] = obs_lst.copy()

                if available_actions is not None:
                    available_actions_lst = np.expand_dims(
                        self.extract_elements(trainer_name, available_actions), axis=1
                    )
                    self.buffer_pool[trainer_name].available_actions[
                        0
                    ] = available_actions_lst.copy()
        self._step = 0

    @torch.no_grad()
    def step(self, step, random_actions=False):
        assert self.__initialized

        actions = np.full(
            (self.n_rollout_threads, self.num_agents), fill_value=None
        ).tolist()
        self.step_data = dict()
        for trainer_name in self.active_trainers:
            self.trainer_total_num_steps[trainer_name] += self.control_agent_count[
                trainer_name
            ]
            self.train_infos[f"{trainer_name}-total_num_steps"] = (
                self.trainer_total_num_steps[trainer_name]
            )

            if self.skip(trainer_name):
                continue

            trainer = self.trainer_pool[trainer_name]
            buffer = (
                self.buffer_pool[trainer_name]
                if trainer_name not in self.on_training
                else self.replay_buffer
            )

            trainer.prep_rollout()

            if self._states is None:
                is_first = np.ones(
                    self.control_agent_count[trainer_name], dtype=np.uint8
                )
                self._states = trainer.policy.get_initial_state(
                    self.control_agent_count[trainer_name],
                )
            else:
                is_first = np.zeros(
                    (self.control_agent_count[trainer_name],), dtype=np.uint8
                )

            to_env = trainer.policy.get_actions(
                {
                    "s_in": self._states,
                    "obs": buffer.obs(),
                    "available_actions": buffer.ava(),
                    # "share_obs": buffer.share_obs[step],
                    "is_first": is_first,
                },
                random_action=random_actions,
            )

            if random_actions:
                action = to_env.reshape(-1, 1)
            else:
                action = to_env["a"].cpu().numpy()
                action = np.argmax(action, -1).reshape(-1, 1)
                self._states = to_env["s_out"]
            self.step_data[trainer_name] = action

            for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                actions[e][a] = action[i]
        return actions

    def insert_data(
        self,
        share_obs,
        obs,
        rewards,
        dones,
        active_masks=None,
        bad_masks=None,
        infos=None,
        available_actions=None,
    ):
        """
        ndarrays of shape (n_rollout_threads, num_agents, *)
        """
        assert self.__initialized
        self._step += 1
        for trainer_name in self.active_trainers:
            if self.skip(trainer_name):
                continue

            # self.trainer_pool[trainer_name]
            buffer = self.replay_buffer

            action = self.step_data[trainer_name]

            # (control_agent_count[trainer_name], 1, *)
            obs_lst = self.extract_elements(trainer_name, obs)
            share_obs_lst = self.extract_elements(trainer_name, share_obs)
            rewards_lst = self.extract_elements(trainer_name, rewards)
            dones_lst = self.extract_elements(trainer_name, dones)

            if infos is not None:
                if "others_actions" in infos[0]:
                    others_actions = np.array(
                        [info["others_actions"] for info in infos]
                    )
                    others_actions_lst = self.extract_elements(
                        trainer_name, others_actions
                    )
                else:
                    others_actions_lst = None
            else:
                others_actions_lst = None

            if available_actions is not None:
                available_actions_lst = self.extract_elements(
                    trainer_name, available_actions
                )

            buffer.insert(
                share_obs_lst,
                obs_lst,
                action,
                others_actions_lst,
                rewards_lst,
                dones_lst,
                available_actions=available_actions_lst,
            )

            if infos is not None:
                # process infos
                pass

            # partner policy info
            policy_ids = []
            for i, (e, a) in enumerate(self.control_agents[trainer_name]):
                policy_ids.append(
                    [
                        self.policy_id(
                            self.map_ea2t[(e, (a + j + 1) % self.num_agents)]
                        )
                        for j in range(self.num_agents - 1)
                    ]
                )
            buffer.insert_policy_id(policy_ids, self.num_agents - 1)

        self.step_data = None

    def prepare_buffer(self):
        self.replay_buffer.torch_episode_len_limit()

    def train(self, **kwargs):
        assert self.__initialized
        self.prepare_buffer()

        for trainer_name in self.on_training:
            trainer = self.trainer_pool[trainer_name]
            buffer = (
                self.buffer_pool[trainer_name]
                if trainer_name not in self.on_training
                else self.replay_buffer
            )

            trainer.prep_training()
            train_info = trainer.train(buffer)
            self.train_infos.update(
                {f"{trainer_name}-{k}": v for k, v in train_info.items()}
            )
            # print(f"Average_episode_rewards:  {np.mean(buffer.recent_avg_rewards) * 200}")
            self.train_infos.update(
                {
                    f"average_episode_rewards": np.mean(buffer.recent_avg_rewards)
                    * self.args.episode_length
                }
            )

            # place first step observation of next episode
            buffer.after_update()

        return copy.deepcopy(self.train_infos)

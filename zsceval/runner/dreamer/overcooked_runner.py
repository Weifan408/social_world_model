import copy
import itertools
import json
import pprint
import time
from collections import defaultdict
import os
from os import path as osp
import pickle
from typing import Dict
from collections.abc import Iterable

import numpy as np
import torch
import wandb
from loguru import logger
from scipy.stats import rankdata
from tqdm import tqdm
from tensorboardX import SummaryWriter

from zsceval.runner.dreamer.base_runner import Runner, make_trainer_policy_cls
from zsceval.utils.log_util import eta, get_table_str


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    """
    A wrapper to start the RL agent training algorithm.
    """

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_wandb = self.all_args.use_wandb
        self.use_single_network = self.all_args.use_single_network
        self.use_render = self.all_args.use_render

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        TrainAlgo, Policy = make_trainer_policy_cls(
            self.algorithm_name, use_single_network=self.use_single_network
        )

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        # policy network
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            device=self.device,
        )

        logger.info(
            f"Action space {self.envs.action_space[0]}, Obs space {self.envs.observation_space[0].shape}, Share obs space {share_observation_space.shape}"
        )

        # dump policy config to allow loading population in yaml form
        self.policy_config = (
            self.all_args,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
        )
        policy_config_path = os.path.join(self.run_dir, "wm_policy_config.pkl")
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")
        if "store" in self.experiment_name:
            exit()

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(
            self.all_args, self.policy, device=self.device)

        self.warmup_episode = self.all_args.warmup_episode

        # for training br
        self.br_best_sparse_r = 0
        self.br_eval_json = {}

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(
                str(self.model_dir) + "/model.pt", map_location=self.device
            )
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(
                str(self.model_dir), map_location=self.device
            )
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not (self.all_args.use_render or self.all_args.use_eval):
                policy_critic_state_dict = torch.load(
                    str(self.model_dir) + "/critic.pt", map_location=self.device
                )
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def save(self, step, save_critic: bool = False):
        # logger.info(f"save sp periodic_{step}.pt")
        if self.use_single_network:
            policy_model = self.trainer.policy.model
            torch.save(
                policy_model.state_dict(),
                str(self.save_dir) + f"/model_periodic_{step}.pt",
            )
        else:
            dreamer_model = self.trainer.policy.dreamer_model
            torch.save(
                dreamer_model.state_dict(),
                str(self.save_dir) + f"/model_{step}.pt",
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        if self.env_name == "Overcooked":
            if self.all_args.overcooked_version == "old":
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info_keys = SHAPED_INFOS
            else:
                from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info_keys = SHAPED_INFOS
        eval_average_episode_rewards = []
        eval_obs, _, eval_available_actions, eval_info = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_info["role"]),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=not self.all_args.eval_stochastic,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            # Obser reward and next obs
            (
                eval_obs,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_average_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        ep_r = []
        _infos = eval_infos["episode"]
        for ps_info in _infos:
            for dummy_info in ps_info:
                for a in range(self.num_agents):
                    eval_env_infos[f"ep_sparse_r_by_agent{a}"].append(
                        dummy_info["ep_sparse_r_by_agent"][a]
                    )
                    eval_env_infos[f"ep_shaped_r_by_agent{a}"].append(
                        dummy_info["ep_shaped_r_by_agent"][a]
                    )
                    for i, k in enumerate(shaped_info_keys):
                        eval_env_infos[f"ep_{k}_by_agent{a}"].append(
                            dummy_info["ep_category_r_by_agent"][a][i]
                        )
                eval_env_infos["eval_ep_sparse_r"].append(
                    dummy_info["ep_sparse_r"])
                eval_env_infos["eval_ep_shaped_r"].append(
                    dummy_info["ep_shaped_r"])

        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            eval_average_episode_rewards, axis=0
        )
        logger.success(
            f'eval average sparse rewards {np.mean(eval_env_infos["eval_ep_sparse_r"]):.3f} {len(eval_env_infos["eval_ep_sparse_r"])} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}'
        )
        self.log_eval_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, _, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in tqdm(range(self.all_args.render_episodes)):
            rnn_states = np.zeros(
                (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []
            for step in range(self.episode_length):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    np.concatenate(available_actions),
                    deterministic=True,
                )
                actions = np.array(
                    np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )
                # Obser reward and next obs
                obs, _, rewards, dones, infos, available_actions = envs.step(
                    actions)
                obs = np.stack(obs)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

            logger.info(
                "average episode rewards is: "
                + str(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            )

    def evaluate_one_episode_with_multi_policy(self, policy_pool: Dict, map_ea2p: Dict):
        """Evaluate one episode with different policy for each agent.
        Params:
            policy_pool (Dict): a pool of policies. Each policy should support methods 'step' that returns actions given observation while maintaining hidden states on its own, and 'reset' that resets the hidden state.
            map_ea2p (Dict): a mapping from (env_id, agent_id) to policy name
        """
        # warnings.warn("Evaluation with multi policy is not compatible with async done.")
        [
            policy.reset(self.n_eval_rollout_threads, self.num_agents)
            for _, policy in policy_pool.items()
        ]
        for e in range(self.n_eval_rollout_threads):
            for agent_id in range(self.num_agents):
                if not map_ea2p[(e, agent_id)].startswith("script:"):
                    policy_pool[map_ea2p[(e, agent_id)]].register_control_agent(
                        e, agent_id
                    )
        if self.all_args.algorithm_name == "cole":
            c_a_str = {
                p_name: len(policy_pool[p_name].control_agents)
                for p_name in self.generated_population_names
                + [self.trainer.agent_name]
            }
            logger.debug(f"control agents num:\n{c_a_str}")

        featurize_type = [
            [
                self.policy.featurize_type[map_ea2p[(e, a)]]
                for a in range(self.num_agents)
            ]
            for e in range(self.n_eval_rollout_threads)
        ]
        self.eval_envs.reset_featurize_type(featurize_type)

        eval_env_infos = defaultdict(list)
        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        extract_info_keys = []  # ['stuck', 'can_begin_cook']
        infos = None
        prev_states = None
        for _ in range(self.all_args.episode_length):
            eval_actions = np.full(
                (self.n_eval_rollout_threads, self.num_agents, 1), fill_value=0
            ).tolist()
            for _, policy in policy_pool.items():
                if len(policy.control_agents) > 0:
                    policy.prep_rollout()
                    policy.to(self.device)
                    obs_lst = [eval_obs[e][a]
                               for (e, a) in policy.control_agents]
                    avail_action_lst = [
                        eval_available_actions[e][a] for (e, a) in policy.control_agents
                    ]
                    info_lst = None
                    if infos is not None:
                        info_lst = {
                            k: [infos[e][k][a]
                                for e, a in policy.control_agents]
                            for k in extract_info_keys
                        }
                    agents = policy.control_agents
                    obs = np.stack(obs_lst, axis=0)
                    avail_actions = np.stack(avail_action_lst)

                    if policy.args.algorithm_name in ["swm", "dreamerV3", "mabl", "mamba"]:
                        if isinstance(obs[0], dict):
                            merged = {}
                            for k in obs[0].keys():
                                merged[k] = np.stack([o[k]
                                                     for o in obs], axis=0)
                            obs = merged

                        if prev_states is None:
                            is_first = np.ones(
                                (avail_actions.shape[0],), dtype=np.uint8
                            )
                            prev_states = policy.policy.get_initial_state(
                                self.n_eval_rollout_threads)
                        else:
                            is_first = np.zeros(
                                (avail_actions.shape[0],), dtype=np.uint8
                            )

                        actions, prev_states = policy.step(
                            obs,
                            agents,
                            info=info_lst,
                            deterministic=not self.all_args.eval_stochastic,
                            return_hidden=True,
                            available_actions=avail_actions,
                            states=prev_states,
                            is_first=is_first,
                        )
                        actions = np.argmax(actions, -1)
                    else:
                        actions = policy.step(
                            obs,
                            agents,
                            info=info_lst,
                            deterministic=not self.all_args.eval_stochastic,
                            available_actions=avail_actions,
                        )

                    for action, (e, a) in zip(actions, agents):
                        eval_actions[e][a] = [action.item()]
            # Observe reward and next obs
            eval_actions = np.array(eval_actions)
            (
                eval_obs,
                _,
                _,
                _,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)

            infos = eval_infos

        if self.all_args.overcooked_version == "old":
            from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                SHAPED_INFOS,
            )

            shaped_info_keys = SHAPED_INFOS
        else:
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                SHAPED_INFOS,
            )

            shaped_info_keys = SHAPED_INFOS

        for eval_info in eval_infos:
            for a in range(self.num_agents):
                for i, k in enumerate(shaped_info_keys):
                    eval_env_infos[f"eval_ep_{k}_by_agent{a}"].append(
                        eval_info["episode"]["ep_category_r_by_agent"][a][i]
                    )
                eval_env_infos[f"eval_ep_sparse_r_by_agent{a}"].append(
                    eval_info["episode"]["ep_sparse_r_by_agent"][a]
                )
                eval_env_infos[f"eval_ep_shaped_r_by_agent{a}"].append(
                    eval_info["episode"]["ep_shaped_r_by_agent"][a]
                )
            eval_env_infos["eval_ep_sparse_r"].append(
                eval_info["episode"]["ep_sparse_r"]
            )
            eval_env_infos["eval_ep_shaped_r"].append(
                eval_info["episode"]["ep_shaped_r"]
            )

        return eval_env_infos

    def evaluate_with_multi_policy(
        self, policy_pool=None, map_ea2p=None, num_eval_episodes=None
    ):
        """Evaluate with different policy for each agent."""
        policy_pool = policy_pool or self.policy.policy_pool
        map_ea2p = map_ea2p or self.policy.map_ea2p
        num_eval_episodes = num_eval_episodes or self.all_args.eval_episodes
        logger.debug(
            f"evaluate {self.population_size} policies with {num_eval_episodes} episodes"
        )
        eval_infos = defaultdict(list)
        for _ in tqdm(
            range(max(1, num_eval_episodes // self.n_eval_rollout_threads)),
            desc="Evaluate with Population",
        ):
            eval_env_info = self.evaluate_one_episode_with_multi_policy(
                policy_pool, map_ea2p
            )
            for k, v in eval_env_info.items():
                for e in range(self.n_eval_rollout_threads):
                    agent0, agent1 = map_ea2p[(e, 0)], map_ea2p[(e, 1)]
                    for log_name in [
                        f"{agent0}-{agent1}-{k}",
                    ]:
                        if k in ["eval_ep_sparse_r", "eval_ep_shaped_r"]:
                            eval_infos[log_name].append(v[e])
                        elif (
                            getattr(self.all_args, "stage", 1) == 1
                            or not self.all_args.use_wandb
                            or ("br" in self.trainer.agent_name)
                        ):
                            eval_infos[log_name].append(v[e])

                    if k in ["eval_ep_sparse_r", "eval_ep_shaped_r"]:
                        for log_name in [
                            f"either-{agent0}-{k}",
                            f"either-{agent0}-{k}-as_agent_0",
                            f"either-{agent1}-{k}",
                            f"either-{agent1}-{k}-as_agent_1",
                        ]:
                            eval_infos[log_name].append(v[e])

        logger.success(
            "eval average sparse rewards:\n{}".format(
                pprint.pformat(
                    {
                        k: f"{np.mean(v):.2f}"
                        for k, v in eval_infos.items()
                        if "ep_sparse_r" in k and "by_agent" not in k
                    },
                    compact=True,
                    width=200,
                )
            )
        )

        eval_infos2dump = {k: np.mean(v) for k, v in eval_infos.items()}

        if hasattr(self.trainer, "agent_name"):
            br_sparse_r = f"either-{self.trainer.agent_name}-eval_ep_sparse_r"
            br_sparse_r = np.mean(eval_infos[br_sparse_r])

            if br_sparse_r >= self.br_best_sparse_r:
                self.br_best_sparse_r = br_sparse_r
                logger.success(
                    f"best eval br sparse reward {self.br_best_sparse_r:.2f} at {self.total_num_steps} steps"
                )
                self.br_eval_json = copy.deepcopy(eval_infos2dump)

                if getattr(self.all_args, "eval_result_path", None):
                    logger.debug(
                        f"dump eval_infos to {self.all_args.eval_result_path}")
                    with open(
                        self.all_args.eval_result_path, "w", encoding="utf-8"
                    ) as f:
                        json.dump(self.br_eval_json, f)
        elif getattr(self.all_args, "eval_result_path", None):
            logger.debug(
                f"dump eval_infos to {self.all_args.eval_result_path}")
            with open(self.all_args.eval_result_path, "w", encoding="utf-8") as f:
                json.dump(eval_infos2dump, f)

        return eval_infos

    def naive_train_with_multi_policy(
        self, reset_map_ea2t_fn=None, reset_map_ea2p_fn=None
    ):
        """This is a naive training loop using TrainerPool and PolicyPool.

        To use PolicyPool and TrainerPool, you should first initialize population in policy_pool, with either:
        >>> self.policy.load_population(population_yaml_path)
        >>> self.trainer.init_population()
        or:
        >>> # mannually register policies
        >>> self.policy.register_policy(policy_name="ppo1", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.policy.register_policy(policy_name="ppo2", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.trainer.init_population()

        To bind (env_id, agent_id) to different trainers and policies:
        >>> map_ea2t = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
        # Qs: 2p? n_eval_rollout_threads?
        >>> map_ea2p = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
        >>> self.trainer.set_map_ea2t(map_ea2t)
        >>> self.policy.set_map_ea2p(map_ea2p)

        # MARK
        Note that map_ea2t is for training while map_ea2p is for policy evaluations

        WARNING: Currently do not support changing map_ea2t and map_ea2p when training. To implement this, we should take the first obs of next episode in the previous buffers and feed into the next buffers.
        """

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.eval_info = dict()
        self.env_info = dict()

        for episode in range(0, episodes):
            if episode < self.warmup_episode:
                random_actions = True
            else:
                random_actions = False
            self.total_num_steps = total_num_steps
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(episode, episodes)

            # reset env agents
            if reset_map_ea2t_fn is not None:
                map_ea2t = reset_map_ea2t_fn(episode)
                self.trainer.reset(
                    map_ea2t,
                    self.n_rollout_threads,
                    self.num_agents,
                    load_unused_to_cpu=True,
                )
                if self.all_args.use_policy_in_env:
                    load_policy_cfg = np.full(
                        (self.n_rollout_threads, self.num_agents), fill_value=None
                    ).tolist()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            trainer_name = map_ea2t[(e, a)]
                            if trainer_name not in self.trainer.on_training:
                                load_policy_cfg[e][a] = (
                                    self.trainer.policy_pool.policy_info[trainer_name]
                                )
                    self.envs.load_policy(load_policy_cfg)

            # init env
            obs, share_obs, available_actions = self.envs.reset()

            # replay buffer
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            s_time = time.time()
            self.trainer.init_first_step(share_obs, obs, available_actions)

            for step in range(self.episode_length):
                # Sample actions
                actions = self.trainer.step(
                    step, random_actions=random_actions)

                # Observe reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor(
                    self.trainer.reward_shaping_steps()
                )

                bad_masks = np.array(
                    [
                        [[0.0] if info["bad_transition"]
                            else [1.0]] * self.num_agents
                        for info in infos
                    ]
                )

                self.trainer.insert_data(
                    share_obs,
                    obs,
                    rewards,
                    dones,
                    bad_masks=bad_masks,
                    infos=infos,
                    available_actions=available_actions,
                )

            # update env infos
            episode_env_infos = defaultdict(list)
            ep_returns_per_trainer = defaultdict(
                lambda: [[] for _ in range(self.num_agents)]
            )
            e2ta = dict()
            if self.env_name == "Overcooked":
                if self.all_args.overcooked_version == "old":
                    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                        SHAPED_INFOS,
                    )

                    shaped_info_keys = SHAPED_INFOS
                else:
                    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                        SHAPED_INFOS,
                    )

                    shaped_info_keys = SHAPED_INFOS
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t[(e, 0)]
                    agent1_trainer = self.trainer.map_ea2t[(e, 1)]
                    for log_name in [
                        f"{agent0_trainer}-{agent1_trainer}",
                    ]:
                        episode_env_infos[f"{log_name}-ep_sparse_r"].append(
                            info["episode"]["ep_sparse_r"]
                        )
                        episode_env_infos[f"{log_name}-ep_shaped_r"].append(
                            info["episode"]["ep_shaped_r"]
                        )
                        for a in range(self.num_agents):
                            if (
                                getattr(self.all_args, "stage", 1) == 1
                                or not self.all_args.use_wandb
                            ):
                                for i, k in enumerate(shaped_info_keys):
                                    episode_env_infos[
                                        f"{log_name}-ep_{k}_by_agent{a}"
                                    ].append(
                                        info["episode"]["ep_category_r_by_agent"][a][i]
                                    )
                            episode_env_infos[
                                f"{log_name}-ep_sparse_r_by_agent{a}"
                            ].append(info["episode"]["ep_sparse_r_by_agent"][a])
                            episode_env_infos[
                                f"{log_name}-ep_shaped_r_by_agent{a}"
                            ].append(info["episode"]["ep_shaped_r_by_agent"][a])
                    for k in ["ep_sparse_r", "ep_shaped_r"]:
                        for log_name in [
                            f"either-{agent0_trainer}-{k}",
                            f"either-{agent0_trainer}-{k}-as_agent_0",
                            f"either-{agent1_trainer}-{k}",
                            f"either-{agent1_trainer}-{k}-as_agent_1",
                        ]:
                            episode_env_infos[log_name].append(
                                info["episode"][k])
                    if agent0_trainer != self.trainer.agent_name:
                        # suitable for both stage 1 and stage 2
                        ep_returns_per_trainer[agent1_trainer][1].append(
                            info["episode"]["ep_sparse_r"]
                        )
                        e2ta[e] = (agent1_trainer, 1)
                    elif agent1_trainer != self.trainer.agent_name:
                        ep_returns_per_trainer[agent0_trainer][0].append(
                            info["episode"]["ep_sparse_r"]
                        )
                        e2ta[e] = (agent0_trainer, 0)
                env_infos.update(episode_env_infos)
            max_ep_sparse_r_dict = defaultdict(lambda: [0, 0])

            self.env_info.update(env_infos)
            e_time = time.time()
            logger.info(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            s_time = time.time()
            # if self.all_args.stage == 1:
            #     self.trainer.adapt_entropy_coef(total_num_steps // self.population_size)
            # else:
            #     self.trainer.adapt_entropy_coef(total_num_steps)

            train_infos = self.trainer.train(
                sp_size=getattr(self, "n_repeats", 0) * self.num_agents
            )
            e_time = time.time()
            logger.info(f"Update models time: {e_time - s_time:.3f}s")

            s_time = time.time()
            if self.all_args.stage == 2:
                # update advantage moving average, used in stage2
                if self.all_args.use_advantage_prioritized_sampling:
                    if not hasattr(self, "avg_adv"):
                        self.avg_adv = defaultdict(float)
                    adv = self.trainer.compute_advantages()
                    for (agent0, agent1, a), vs in adv.items():
                        agent_pair = (agent0, agent1)
                        for v in vs:
                            if agent_pair not in self.avg_adv.keys():
                                self.avg_adv[agent_pair] = v
                            else:
                                self.avg_adv[agent_pair] = (
                                    self.avg_adv[agent_pair] * 0.99 + v * 0.01
                                )

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode < 50:
                if episode % 2 == 0:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)
            elif episode < 100:
                if episode % 5 == 0:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)

            self.trainer.update_best_r(
                {
                    trainer_name: np.mean(
                        self.env_info.get(
                            f"either-{trainer_name}-ep_sparse_r", -1e9)
                    )
                    for trainer_name in self.trainer.active_trainers
                },
                save_dir=self.save_dir,
            )

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )
                average_ep_rew_dict = {
                    k[: k.rfind("-")]: f"{np.mean(train_infos[k]):.3f}"
                    for k in train_infos.keys()
                    if "average_episode_rewards" in k and "either" not in k
                }
                logger.info(
                    f"average episode rewards is\n{pprint.pformat(average_ep_rew_dict, width=600)}"
                )
                average_ep_sparse_rew_dict = {
                    k[: k.rfind("-")]: f"{np.mean(env_infos[k]):.3f}"
                    for k in env_infos.keys()
                    if k.endswith("ep_sparse_r") and "either" not in k
                }
                logger.info(
                    f"average sparse episode rewards is\n{pprint.pformat(average_ep_sparse_rew_dict, width=600, compact=True)}"
                )
                if self.all_args.algorithm_name == "traj":
                    if self.all_args.stage == 1:
                        logger.debug(f'jsd is {train_infos["average_jsd"]}')
                        logger.debug(
                            f'jsd loss is {train_infos["average_jsd_loss"]}')

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)

            # eval
            if (
                episode > 0
                and episode % self.eval_interval == 0
                and self.use_eval
                or episode == episodes - 1
            ):
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                eval_info = self.evaluate_with_multi_policy()
                # logger.debug("eval_info: {}".format(pprint.pformat(eval_info)))
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)

            e_time = time.time()
            logger.info(f"Post update models time: {e_time - s_time:.3f}s")

    def train_wm(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents

        logger.info(
            f"population_size: {self.all_args.population_size}, {self.population}"
        )

        agent_name = self.trainer.agent_name
        # assert self.use_eval
        assert (
            self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0
            and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0
        )
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args.eval_episodes = (
            self.all_args.eval_episodes
            * self.population_size
            // self.all_args.eval_env_batch
        )
        self.eval_idx = 0
        all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(
            itertools.product([agent_name], self.population)
        )
        logger.info(f"all agent pairs: {all_agent_pairs}")

        running_avg_r = - \
            np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

        def mep_reset_map_ea2t_fn(episode):
            # Randomly select agents from population to be trained
            # 1) consistent with MEP to train against one agent each episode 2) sample different agents to train against
            sampling_prob_np = (
                np.ones((self.population_size * 2,)) / self.population_size / 2
            )
            if self.all_args.use_advantage_prioritized_sampling:
                # logger.debug("use advantage prioritized sampling")
                if episode > 0:
                    metric_np = np.array(
                        [self.avg_adv[agent_pair]
                            for agent_pair in all_agent_pairs]
                    )
                    # TODO: retry this
                    sampling_rank_np = rankdata(metric_np, method="dense")
                    sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                    sampling_prob_np /= sampling_prob_np.sum()
                    maxv = 1.0 / (self.population_size * 2) * 10
                    while sampling_prob_np.max() > maxv + 1e-6:
                        sampling_prob_np = sampling_prob_np.clip(max=maxv)
                        sampling_prob_np /= sampling_prob_np.sum()
            elif self.all_args.mep_use_prioritized_sampling:
                metric_np = np.zeros((self.population_size * 2,))
                for i, agent_pair in enumerate(all_agent_pairs):
                    train_r = np.mean(
                        self.env_info.get(
                            f"{agent_pair[0]}-{agent_pair[1]}-ep_sparse_r", -1e9
                        )
                    )
                    eval_r = np.mean(
                        self.eval_info.get(
                            f"{agent_pair[0]}-{agent_pair[1]}-eval_ep_sparse_r",
                            -1e9,
                        )
                    )

                    avg_r = 0.0
                    cnt_r = 0
                    if train_r > -1e9:
                        avg_r += train_r * (
                            self.n_rollout_threads // self.all_args.train_env_batch
                        )
                        cnt_r += self.n_rollout_threads // self.all_args.train_env_batch
                    if eval_r > -1e9:
                        avg_r += eval_r * (
                            self.all_args.eval_episodes
                            // (
                                self.n_eval_rollout_threads
                                // self.all_args.eval_env_batch
                            )
                        )
                        cnt_r += self.all_args.eval_episodes // (
                            self.n_eval_rollout_threads // self.all_args.eval_env_batch
                        )
                    if cnt_r > 0:
                        avg_r /= cnt_r
                    else:
                        avg_r = -1e9
                    if running_avg_r[i] == -1e9:
                        running_avg_r[i] = avg_r
                    else:
                        # running average
                        running_avg_r[i] = running_avg_r[i] * \
                            0.95 + avg_r * 0.05
                    metric_np[i] = running_avg_r[i]
                running_avg_r_dict = {}
                for i, agent_pair in enumerate(all_agent_pairs):
                    running_avg_r_dict[
                        f"running_average_return/{agent_pair[0]}-{agent_pair[1]}"
                    ] = np.mean(running_avg_r[i])
                if self.use_wandb:
                    for k, v in running_avg_r_dict.items():
                        if v > -1e9:
                            wandb.log({k: v}, step=self.total_num_steps)
                running_avg_r_dict = {
                    f"running_average_return/{agent_pair[0]}-{agent_pair[1]}": f"{running_avg_r[i]:.3f}"
                    for i, agent_pair in enumerate(all_agent_pairs)
                }
                logger.trace(
                    f"running avg_r\n{pprint.pformat(running_avg_r_dict, width=600, compact=True)}"
                )
                if (metric_np > -1e9).astype(np.int32).sum() > 0:
                    avg_metric = metric_np[metric_np > -1e9].mean()
                else:
                    # uniform
                    avg_metric = 1.0
                metric_np[metric_np == -1e9] = avg_metric

                # reversed return
                sampling_rank_np = rankdata(
                    1.0 / (metric_np + 1e-6), method="dense")
                sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                sampling_prob_np = sampling_prob_np**self.all_args.mep_prioritized_alpha
                sampling_prob_np /= sampling_prob_np.sum()
            assert abs(sampling_prob_np.sum() - 1) < 1e-3

            # log sampling prob
            sampling_prob_dict = {}
            for i, agent_pair in enumerate(all_agent_pairs):
                sampling_prob_dict[f"sampling_prob/{agent_pair[0]}-{agent_pair[1]}"] = (
                    sampling_prob_np[i]
                )
            if self.use_wandb:
                wandb.log(sampling_prob_dict, step=self.total_num_steps)

            n_selected = self.n_rollout_threads // self.all_args.train_env_batch
            pair_idx = np.random.choice(
                2 * self.population_size, size=(n_selected,), p=sampling_prob_np
            )
            # assert n_selected % (2 * self.all_args.population_size) == 0
            # pair_idx = [
            #     i % (2*self.population_size) for i in range(n_selected)
            # ]

            if self.all_args.uniform_sampling_repeat > 0:
                assert (
                    n_selected
                    >= 2 * self.population_size * self.all_args.uniform_sampling_repeat
                )
                i = 0
                for r in range(self.all_args.uniform_sampling_repeat):
                    for x in range(2 * self.population_size):
                        pair_idx[i] = x
                        i += 1
            map_ea2t = {
                (e, a): all_agent_pairs[pair_idx[e % n_selected]][a]
                for e, a in itertools.product(
                    range(self.n_rollout_threads), range(self.num_agents)
                )
            }

            featurize_type = [
                [
                    self.policy.featurize_type[map_ea2t[(e, a)]]
                    for a in range(self.num_agents)
                ]
                for e in range(self.n_rollout_threads)
            ]
            self.envs.reset_featurize_type(featurize_type)

            return map_ea2t

        def mep_reset_map_ea2p_fn(episode):
            if self.all_args.eval_policy != "":
                map_ea2p = {
                    (e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2]
                    for e, a in itertools.product(
                        range(self.n_eval_rollout_threads), range(
                            self.num_agents)
                    )
                }
            else:
                map_ea2p = {
                    (e, a): all_agent_pairs[
                        (self.eval_idx + e // self.all_args.eval_env_batch)
                        % (self.population_size * 2)
                    ][a]
                    for e, a in itertools.product(
                        range(self.n_eval_rollout_threads), range(
                            self.num_agents)
                    )
                }
                self.eval_idx += (
                    self.n_eval_rollout_threads // self.all_args.eval_env_batch
                )
                self.eval_idx %= self.population_size * 2
            featurize_type = [
                [
                    self.policy.featurize_type[map_ea2p[(e, a)]]
                    for a in range(self.num_agents)
                ]
                for e in range(self.n_eval_rollout_threads)
            ]
            self.eval_envs.reset_featurize_type(featurize_type)
            return map_ea2p

        self.naive_train_with_multi_policy(
            reset_map_ea2t_fn=mep_reset_map_ea2t_fn,
            reset_map_ea2p_fn=mep_reset_map_ea2p_fn,
        )

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.roles[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(
            np.split(_t2n(next_values), self.n_rollout_threads))

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def log_eval_env(self, eval_info, total_num_steps):
        for k, v in eval_info.items():
            if isinstance(v, Iterable):
                # v = np.mean(v)
                for i in range(len(v)):
                    if self.use_wandb:
                        wandb.log(
                            {f"eval{i+1}/{k}": v[i]}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(
                            f"eval{i+1}/{k}", {
                                f"eval{i}/{k}": v[i]}, total_num_steps
                        )

    def log(self, info, total_num_steps):
        for k, v in info.items():
            if isinstance(v, Iterable):
                # v = np.mean(v)
                for i in range(len(v)):
                    if self.use_wandb:
                        wandb.log({k: v[i]}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(k, {k: v[i]}, total_num_steps)

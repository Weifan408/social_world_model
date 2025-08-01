from loguru import logger

import numpy as np
import torch
from torch import nn

from zsceval.utils.util import get_shape_from_obs_space

from .components.encoder import Encoder
from .components.decoder import Decoder
from .world_model import WorldModel
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from .dreamer_model import DreamerModel
from .components.utils import to_torch


class ExDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DreamerPolicy(nn.Module):
    def __init__(
        self,
        args,
        obs_space,
        share_observation_space,
        act_space,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.world_model_lr = args.world_model_lr
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.data_parallel = getattr(args, "data_parallel", False)

        self.obs_space = obs_space
        self.act_space = act_space
        symlog_obs = False

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_observation_space)

        if args.image_obs:
            self.encoder = Encoder(args, obs_shape, args.cnn_layers_params)
            self.decoder = Decoder(
                args,
                args.num_gru_units + args.feat_size,
                obs_shape,
                self.encoder.encoder_output_dims,
                args.cnn_layers_params,
            )
        else:
            self.encoder = Encoder(args, obs_shape, None)
            self.decoder = Decoder(
                args,
                args.num_gru_units + args.feat_size,
                obs_shape
            )

        self.world_model = WorldModel(
            args,
            action_space=act_space,
            encoder=self.encoder,
            decoder=self.decoder,
            symlog_obs=symlog_obs,
        )

        if args.use_vqvae:
            self.actor = ActorNetwork(
                args,
                input_dim=args.num_gru_units + args.feat_size + args.num_gru_units, # vq embedding dim = args.num_gru_units
                action_space=act_space,
            )
            self.critic = CriticNetwork(
                args,
                args.num_gru_units + args.feat_size + args.num_gru_units
            )
        else:
            self.actor = ActorNetwork(
                args,
                input_dim=args.num_gru_units + args.feat_size +
                args.population_size * (args.num_agents-1),  # grf + 1
                action_space=act_space,
            )

            self.critic = CriticNetwork(
                args,
                args.num_gru_units + args.feat_size +
                args.population_size * (args.num_agents-1)   # grf + 1
            )

        self.dreamer_model = DreamerModel(
            args, self.actor, self.critic, self.world_model, self.device
        )

        self.critic.init_ema()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.world_model_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_initial_state(self, batch_size):
        return self.dreamer_model.get_initial_state()

    def forward_inference(self, batch):
        batch = to_torch(batch, device=self.device)

        actions, next_state = self.dreamer_model.forward_inference(
            observations=batch["obs"],
            previous_states=batch["s_in"],
            is_first=batch["is_first"],
            available_actions=batch["available_actions"],
        )
        return actions, next_state

    def forward_exploration(self, batch):
        batch = to_torch(batch, device=self.device)

        actions, next_state = self.dreamer_model.forward_exploration(
            observations=batch["obs"],
            previous_states=batch["s_in"],
            is_first=batch["is_first"],
            available_actions=batch["available_actions"],
        )
        return {"a": actions, "s_out": next_state}

    def forward(self, batch):
        return self.dreamer_model.forward_train(
            observations=batch["obs"],
            actions=batch["actions"],
            is_first=batch["is_first"],
        )

    def to_parallel(self):
        pass
        # if self.data_parallel:
        #     logger.warning(
        #         f"Use Data Parallel for Forwarding in devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
        #     )
        #     for name, children in self.actor.named_children():
        #         setattr(self.actor, name, ExDataParallel(children))
        #     for name, children in self.critic.named_children():
        #         setattr(self.critic, name, ExDataParallel(children))

    def get_actions(self, batch, random_action=False):
        if random_action:
            available_actions = batch["available_actions"]
            random_values = np.random.rand(*available_actions.shape)
            random_values[available_actions == 0] = -np.inf
            sampled_actions = np.argmax(random_values, axis=-1)
            return sampled_actions
        return self.forward_exploration(batch)

    def act(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        states=None,
        is_first=False,
        **kwargs,
    ):
        return self.forward_inference(
            {
                "s_in": states,
                "obs": obs,
                "available_actions": available_actions,
                # "share_obs": buffer.share_obs[step],
                "is_first": is_first,
            }
        )

    def to(self, device):
        # self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.world_model.to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.dreamer_model.to(device)

    def prep_training(self):
        self.dreamer_model.train()

    def prep_rollout(self):
        self.world_model.eval()
        self.actor.eval()
        self.critic.eval()
        self.dreamer_model.eval()

    def load_checkpoint(self, ckpt_path):
        self.dreamer_model.load_state_dict(
            torch.load(ckpt_path["actor"], map_location=self.device)
        )
        # if "dreamer_model" in ckpt_path:
        #     self.dreamer_model.load_state_dict(
        #         torch.load(ckpt_path["dreamer_model"], map_location=self.device)
        #     )

        # if "actor" in ckpt_path:
        #     self.actor.load_state_dict(
        #         torch.load(ckpt_path["actor"], map_location=self.device)
        #     )
        # if "critic" in ckpt_path:
        #     self.critic.load_state_dict(
        #         torch.load(ckpt_path["critic"], map_location=self.device)
        #     )

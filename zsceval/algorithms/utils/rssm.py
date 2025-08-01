from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

from zsceval.algorithms.utils.distributions import DiscreteLatentDist


RSSM_STATE_MODE = "discrete"


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(
            self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1
        )


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {"discrete": RSSMStateDiscrete,
             "cont": RSSMStateCont}[RSSM_STATE_MODE]


@dataclass
class BiRSSMStateBase:
    global_stoch: torch.Tensor
    global_deter: torch.Tensor
    agent_stoch: torch.Tensor
    agent_deter: torch.Tensor

    def map(self, func):
        return BiRSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self, agent_flag):
        if agent_flag:
            return torch.cat((self.agent_stoch, self.agent_deter), dim=-1)
        else:
            return torch.cat(
                (
                    self.global_stoch,
                    self.agent_stoch,
                    self.global_deter,
                    self.agent_deter,
                ),
                dim=-1,
            )

    def get_dist(self, *input):
        pass


@dataclass
class BiRSSMStateDiscrete(BiRSSMStateBase):
    global_logits: torch.Tensor
    agent_logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes, agent_flag):
        if agent_flag:
            return F.softmax(
                self.agent_logits.reshape(
                    *batch_shape, n_categoricals, n_classes), -1
            )
        else:
            return F.softmax(
                self.global_logits.reshape(
                    *batch_shape, n_categoricals, n_classes), -1
            )


BiRSSMState = {
    "discrete": BiRSSMStateDiscrete,
}[RSSM_STATE_MODE]


class RSSMTransition(nn.Module):
    def __init__(
        self, args, action_size, num_agent, hidden_size=200, activation=nn.ReLU, device=torch.device("cpu")
    ):
        super().__init__()
        self._stoch_size = args.wm_stochastic
        self._deter_size = args.wm_determinstic
        self._hidden_size = hidden_size
        self._activation = activation
        self._action_size = action_size

        self._global_cell = nn.GRU(hidden_size, self._deter_size)
        self._global_rnn_input_model = self._build_rnn_input_model(
            action_size * num_agent + self._stoch_size, 2
        )
        self._global_stochastic_prior_model = DiscreteLatentDist(
            self._deter_size,
            args.wm_n_categoricals,
            args.wm_n_classes,
            self._hidden_size,
        )

        self._agent_cell = nn.GRU(hidden_size, self._deter_size)
        self._agent_rnn_input_model = self._build_rnn_input_model(
            action_size + self._stoch_size, 2
        )
        self._agent_stochastic_prior_model = DiscreteLatentDist(
            self._deter_size + self._stoch_size,
            args.wm_n_categoricals,
            args.wm_n_classes,
            self._hidden_size,
        )
        self.device = device

    def _build_rnn_input_model(self, in_dim, n):
        rnn_input_model = []
        input_size = in_dim
        for i in range(n):
            rnn_input_model += [nn.Linear(input_size, self._hidden_size)]
            rnn_input_model += [self._activation()]
            input_size = self._hidden_size
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_others_actions, prev_states):
        batch_size = prev_actions.shape[0]
        n_agents = prev_actions.shape[1]

        prev_actions = F.one_hot(
            prev_actions.long(), num_classes=self._action_size
        ).float()
        prev_flatten_actions = prev_actions.flatten(start_dim=-2)

        prev_others_actions = F.one_hot(
            prev_others_actions.long(), num_classes=self._action_size
        ).float()
        prev_flatten_others_actions = prev_others_actions.flatten(start_dim=-2)

        # (B*T, N, ALL_NUM_AGENTS * ACTION_SIZE)
        prev_all_agent_actions = torch.cat(
            [prev_flatten_actions, prev_flatten_others_actions], dim=-1)

        global_stoch_input = self._global_rnn_input_model(
            torch.cat([prev_all_agent_actions,
                      prev_states.global_stoch], dim=-1)
        )
        global_deter_state = self._global_cell(
            global_stoch_input.reshape(1, batch_size * n_agents, -1),
            prev_states.global_deter.reshape(1, batch_size * n_agents, -1),
        )[0].reshape(batch_size, n_agents, -1)
        global_logits, global_stoch_state = self._global_stochastic_prior_model(
            global_deter_state
        )

        agent_stoch_input = self._agent_rnn_input_model(
            torch.cat([prev_flatten_actions, prev_states.agent_stoch], dim=-1)
        )
        agent_deter_state = self._agent_cell(
            agent_stoch_input.reshape(1, batch_size * n_agents, -1),
            prev_states.agent_deter.reshape(1, batch_size * n_agents, -1),
        )[0].reshape(batch_size, n_agents, -1)
        agent_logits, agent_stoch_state = self._agent_stochastic_prior_model(
            torch.cat([agent_deter_state, global_stoch_state], dim=-1)
        )

        return BiRSSMState(
            agent_logits=agent_logits,
            agent_stoch=agent_stoch_state,
            agent_deter=agent_deter_state,
            global_logits=global_logits,
            global_stoch=global_stoch_state,
            global_deter=global_deter_state,
        )

    def forward_inference(self, prev_actions, prev_states):
        batch_size = prev_actions.shape[0]
        n_agents = prev_actions.shape[1]

        # use one-hot encoding for actions
        prev_actions = F.one_hot(
            prev_actions.long(), num_classes=self._action_size
        ).float().to(self.device).squeeze(-2)  # squeeze to remove the extra dim

        agent_stoch_input = self._agent_rnn_input_model(
            torch.cat([prev_actions, prev_states.agent_stoch], dim=-1)
        )
        agent_deter_state = self._agent_cell(
            agent_stoch_input.reshape(1, batch_size * n_agents, -1),
            prev_states.agent_deter.reshape(1, batch_size * n_agents, -1),
        )[0].reshape(batch_size, n_agents, -1)

        return agent_deter_state  # h_t^a


class RSSMRepresentation(nn.Module):
    def __init__(self, args, state_encoder, transition_model: RSSMTransition, device=torch.device("cpu")):
        super().__init__()
        self._transition_model = transition_model
        self._stoch_size = args.wm_stochastic
        self._deter_size = args.wm_determinstic
        self.device = device

        self._agent_stochastic_posterior_model = DiscreteLatentDist(
            self._deter_size + args.wm_model_hidden,
            args.wm_n_categoricals,
            args.wm_n_classes,
            args.wm_model_hidden,
        )

        self.state_encoder = state_encoder

        self._global_stochastic_posterior_model = DiscreteLatentDist(
            self._deter_size + self.state_encoder.output_size + self._stoch_size,
            args.wm_n_categoricals,
            args.wm_n_classes,
            args.wm_model_hidden,
        )

    def initial_state(self, batch_size, n_agents, **kwargs):
        return BiRSSMState(
            agent_stoch=torch.zeros(
                batch_size, n_agents, self._stoch_size, **kwargs),
            agent_logits=torch.zeros(
                batch_size, n_agents, self._stoch_size, **kwargs),
            agent_deter=torch.zeros(
                batch_size, n_agents, self._deter_size, **kwargs),
            global_stoch=torch.zeros(
                batch_size, n_agents, self._stoch_size, **kwargs),
            global_logits=torch.zeros(
                batch_size, n_agents, self._stoch_size, **kwargs),
            global_deter=torch.zeros(
                batch_size, n_agents, self._deter_size, **kwargs),
        )

    def forward(self, obs_embed, prev_actions, prev_others_actions, prev_states, state_obs):
        """
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_states = self._transition_model(
            prev_actions, prev_others_actions, prev_states)
        agent_logits, agent_stoch_state = self._agent_stochastic_posterior_model(
            torch.cat([prior_states.agent_deter, obs_embed], dim=-1)
        )

        state_embed = self.state_encoder(state_obs)
        global_logits, global_stoch_state = self._global_stochastic_posterior_model(
            torch.cat(
                [prior_states.global_deter, agent_stoch_state, state_embed], dim=-1
            )
        )

        posterior_states = BiRSSMState(
            global_logits=global_logits,
            global_stoch=global_stoch_state,
            global_deter=prior_states.global_deter,
            agent_logits=agent_logits,
            agent_stoch=agent_stoch_state,
            agent_deter=prior_states.agent_deter,
        )
        return prior_states, posterior_states

    def forward_inference(self, obs_embed, prev_actions, prev_states):
        agent_deter_state = self._transition_model.forward_inference(
            prev_actions, prev_states
        )
        agent_logits, agent_stoch_state = self._agent_stochastic_posterior_model(
            torch.cat([agent_deter_state, obs_embed], dim=-1)
        )
        return BiRSSMState(
            global_logits=None,
            global_deter=None,
            global_stoch=None,
            agent_logits=agent_logits,
            agent_deter=agent_deter_state,
            agent_stoch=agent_stoch_state,
        )


def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return BiRSSMState(
        *[
            func([getattr(state, key) for state in rssm_states], dim=dim)
            for key in rssm_states[0].__dict__.keys()
        ]
    )


def rollout_representation(
    representation_model, steps, obs_embed, action, others_action, prev_states, done, state_obs
):
    """
    Roll out the model with actions and observations from data.
    :param steps: number of steps to roll out
    :param obs_embed: size(time_steps, batch_size, n_agents, embedding_size)
    :param action: size(time_steps, batch_size, n_agents, action_size)
    :param prev_states: RSSM state, size(batch_size, n_agents, state_size)
    :return: prior, posterior states. size(time_steps, batch_size, n_agents, state_size)
    """

    priors = []
    posteriors = []
    for t in range(steps):
        prior_states, posterior_states = representation_model(
            obs_embed[t], action[t], others_action[t], prev_states, state_obs[t]
        )
        prev_states = posterior_states.map(lambda x: x * (1.0 - done[t]))
        priors.append(prior_states)
        posteriors.append(posterior_states)

    prior = stack_states(priors, dim=0)
    post = stack_states(posteriors, dim=0)
    return (
        prior.map(lambda x: x[:-1]),
        post.map(lambda x: x[:-1]),
        post.global_deter[1:],
        post.agent_deter[1:],
    )


def rollout_policy(
    transition_model,
    av_action_model,
    others_action_pred_model,
    steps,
    policy,
    prev_state,
):
    """
    Roll out the model with a policy function.
    :param steps: number of steps to roll out
    :param policy: RSSMState -> action
    :param prev_state: RSSM state, size(batch_size, state_size)
    :return: next states size(time_steps, batch_size, state_size),
             actions size(time_steps, batch_size, action_size)
    """
    state = prev_state
    next_states = []
    actions = []
    others_actions = []
    av_actions = []
    action_log_probs = []
    for t in range(steps):
        agent_feat = state.get_features(agent_flag=True).detach()
        all_feat = state.get_features(agent_flag=False).detach()

        if av_action_model is not None:
            avail_actions = av_action_model(all_feat).sample()
            av_actions.append(avail_actions.squeeze(0))
        else:
            avail_actions = None

        action, action_log_prob = policy(agent_feat, avail_actions)

        next_states.append(state)
        action_log_probs.append(action_log_prob)
        actions.append(action)

        others_action = others_action_pred_model(all_feat)
        others_actions.append(others_action)
        state = transition_model(action, others_action, state)

    return {
        "imag_states": stack_states(next_states, dim=0),
        "actions": torch.stack(actions, dim=0),
        "others_actions": torch.stack(others_actions, dim=0),
        "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
        "old_action_log_probs": torch.stack(action_log_probs, dim=0),
    }

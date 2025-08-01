from gym.spaces import Box, Dict
import numpy as np


class ObsWrapper:
    def __init__(self, env):
        self._env = env
        self.mlp_keys = []
        self.cnn_keys = []

        if isinstance(self._env.observation_space, dict):
            mlp_size = 0
            for k, v in self._env.observation_space.items():
                if v.shape == 3:
                    self.cnn_keys.append(k)
                else:
                    self.mlp_keys.append(k)
                    mlp_size += np.prod(v.shape)
            self.observation_space = Dict({
                "mlp": Box(0, 1, shape=(mlp_size,)),
                **{key: self._env.observation_space[key] for key in self.cnn_keys}
            })
        else:
            self.observation_space = self._env.observation_space

        self.share_observation_space = self._env.share_observation_space
        self.action_space = self._env.action_space

    def process_observations(self, obs):
        if len(self.mlp_keys) > 0:
            mlp_obs = []
            for key in self.mlp_keys:
                mlp_obs.append(obs[key].flatten())
            return {
                'mlp': np.concatenate(mlp_obs, axis=-1),
                ** {
                    key: obs[key] for key in self.cnn_keys
                }
            }
        else:
            return obs

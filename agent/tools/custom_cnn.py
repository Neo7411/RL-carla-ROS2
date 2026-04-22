"""Extractor for Dict({bev, vector}) observations.

CNN branch over the BEV + small MLP branch over the goal-relative vector,
concatenated into a single feature vector for the PPO policy.
"""
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BEVGoalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 cnn_dim: int = 512, mlp_dim: int = 64):
        super().__init__(observation_space, features_dim=cnn_dim + mlp_dim)

        bev_space = observation_space.spaces['bev']
        vec_space = observation_space.spaces['vector']
        h, w, c = bev_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        with th.no_grad():
            dummy = th.zeros(1, c, h, w)
            flat = self.cnn(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, cnn_dim), nn.ReLU(),
            nn.Linear(cnn_dim, cnn_dim), nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(vec_space.shape[0], 64), nn.ReLU(),
            nn.Linear(64, mlp_dim), nn.ReLU(),
        )

    def forward(self, obs: dict) -> th.Tensor:
        bev = obs['bev'].permute(0, 3, 1, 2)    # NHWC → NCHW
        feat = self.head(self.cnn(bev))
        return th.cat([feat, self.mlp(obs['vector'])], dim=1)

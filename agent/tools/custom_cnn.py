"""
custom_cnn.py — PyTorch CNN for SB3
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class BEVFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # BEV shape is (H, W, Channels). We need the channel count.
        n_input_channels = observation_space.shape[-1]
        
        # PyTorch equivalent of your TF2 Conv2D layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Equivalent to GlobalAveragePooling2D
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # SB3 passes obs as [Batch, H, W, C]. PyTorch expects [Batch, C, H, W]
        x = observations.permute(0, 3, 1, 2)
        x = self.cnn(x)
        return self.linear(x)
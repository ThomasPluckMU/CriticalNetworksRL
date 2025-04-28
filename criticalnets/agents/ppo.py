import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict, List, Tuple
from . import BaseAtariAgent
from ..utils.numerical_helpers import criticality_regularization


def _init_conv_layers(frame_stack: int, device: torch.device):
    conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4).to(device)
    conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)
    conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device)
    return conv1, conv2, conv3


class PPOAgent(BaseAtariAgent):
    """
    PPO Agent with shared convolutional backbone and
    separate policy and value heads.
    """

    def __init__(self, config: Dict, action_space: int):
        super().__init__(config, action_space)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)
        self.epsilon = config.get("epsilon", 0.0)
        # build conv backbone
        self.conv1, self.conv2, self.conv3 = _init_conv_layers(
            self.frame_stack, self.device
        )
        self.activation = torch.tanh
        # fully connected and heads initialized later
        self._fc = None
        self.policy_head = None
        self.value_head = None

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0  # Simple normalization only
        # Reshape parallel input [1,32,4,84,84] -> [32,4,84,84]
        return x.squeeze(0)  # Remove batch dimension

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        x = self._extract_features(x)
            
        z1 = self.conv1(x).to(self.device)
        a1 = self.activation(z1)
        z2 = self.conv2(a1).to(self.device)
        a2 = self.activation(z2)
        z3 = self.conv3(a2).to(self.device)
        a3 = self.activation(z3)
        flat = a3.view(a3.size(0), -1)
        if self._fc is None:
            self._fc = nn.Linear(flat.size(1), 512).to(self.device)
            self.policy_head = nn.Linear(512, 6).to(self.device)
            self.value_head = nn.Linear(512, 1).to(self.device)
        z4 = self._fc(flat)
        a4 = self.activation(z4)
        logits = self.policy_head(a4)
        value = self.value_head(a4).squeeze(-1)
        return logits, value

    def act(self, states: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        if self.epsilon > 0:
            mask = torch.rand(actions.size(0)) < self.epsilon
            actions[mask] = torch.randint(0, 6, (mask.sum(),))
        return actions

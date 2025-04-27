from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseAtariAgent


class PongA2CAgent(BaseAtariAgent):
    """
    Advantage Actor-Critic (A2C) agent specialized for Pong (6 actions).
    Uses a shared conv backbone with separate actor and critic heads.
    """

    def __init__(self, config: Dict, action_space: int):
        # Always force 6 actions for Pong regardless of passed action_space
        super().__init__(config, action_space=6)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # FC representation
        self.fc = None
        # Actor and critic heads
        self.actor = nn.Linear(512, 6)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device).float() / 255.0  # Simple normalization only
        x = x.squeeze(1)  # Remove extra dimension from envpool [batch,1,channels,height,width]

        # Conv backbone
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 512).to(self.device)
        x = F.relu(self.fc(x))

        # Outputs
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, states: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values.squeeze(-1), log_probs, entropy

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
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(self.device)

        # FC representation
        self.fc = None
        # Actor and critic heads
        self.actor = nn.Linear(512, 6).to(self.device)
        self.critic = nn.Linear(512, 1).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device).float() / 255.0  # Simple normalization only
        
        # Conv backbone
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=2).transpose(1, 2).flatten(start_dim=1, end_dim=2)
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

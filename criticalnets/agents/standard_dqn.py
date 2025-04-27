from typing import Dict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseAtariAgent


class StandardAtariDQN(BaseAtariAgent):
    def __init__(self, config: Dict, action_space: int):
        """
        Initialize the standard DQN network

        Args:
            config (Dict): Configuration dictionary
            action_space (int): Number of possible actions in the environment
        """
        super().__init__(config, action_space)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)
        self.epsilon = config.get("epsilon", 0.1)

        # Define standard convolutional layers
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Feature size will be determined after first forward pass
        self.fc = None
        self.head = nn.Linear(512, action_space)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Q-values for each action
        """
        x = x.to(self.device).float() / 255.0  # Simple normalization only
        x = x.squeeze(1)  # Remove extra dimension from envpool [batch,1,channels,height,width]

        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Initialize fc layer with correct input size if not already done
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 512).to(self.device)

        # Apply fully connected layer with ReLU activation
        x = F.relu(self.fc(x))

        # Output layer (no activation as these are Q-values)
        return self.head(x)

    def act(self, states):
        """Select actions using epsilon-greedy policy"""
        with torch.no_grad():
            q_values = self.forward(states)
            actions = q_values.argmax(dim=1)
            if self.epsilon > 0:
                mask = torch.rand(actions.size(0)) < self.epsilon
                actions[mask] = torch.randint(0, 6, (mask.sum(),))
            return actions

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
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(self.device)

        # Feature size will be determined after first forward pass
        self.fc = None
        self.head = nn.Linear(512, action_space).to(self.device)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Q-values for each action
        """
        x = x.to(self.device).float() / 255.0  # Simple normalization only
        
        # Handle all possible input shapes:
        # (4,84,84) -> single frame
        # (32,4,84,84) -> batch of frames
        # (1,32,4,84,84) -> batched batch of frames
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dim
        elif x.dim() == 5:
            x = x.squeeze(0)  # Remove outer batch dim
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D, 4D or 5D input, got {x.dim()}D")
            
        # Ensure we have proper batch dimension (32,4,84,84)
        if x.size(0) != 32:
            x = x.repeat(32, 1, 1, 1)  # Repeat single frame to match batch size

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
                device = states.device
                mask = torch.rand(actions.size(0), device=device) < self.epsilon
                actions[mask] = torch.randint(0, 6, (mask.sum(),), device=device)
            return actions

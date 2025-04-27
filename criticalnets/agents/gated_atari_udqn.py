from typing import Dict
import random
import torch
import torch.nn as nn
from . import BaseAtariAgent
from criticalnets.layers import GatedDynamicBiasCNN


class GatedAtariUDQN(BaseAtariAgent):
    def __init__(self, config: Dict, action_space: int):
        """
        Initialize the Gated DQN network with dynamic bias layers

        Args:
            config (Dict): Configuration dictionary
            action_space (int): Number of possible actions in the environment
        """
        super().__init__(config, action_space)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)
        self.epsilon = config.get("epsilon", 0.1)
        self.batch_size = config.get("batch_size", 32)

        # Define the convolutional layers with gated dynamic bias
        self.conv1 = GatedDynamicBiasCNN(self.frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = GatedDynamicBiasCNN(32, 64, kernel_size=4, stride=2)
        self.conv3 = GatedDynamicBiasCNN(64, 64, kernel_size=3, stride=1)

        # Fully connected layers - will be initialized after first forward pass
        self.fc = None
        self.head = nn.Linear(512, action_space)

        # Initialize the output shapes for the conv layers
        self._initialize_network()

    def _initialize_network(self):
        """Initialize the network by setting up the output shapes for convolutional layers"""
        # Use actual expected input dimensions (84x84 after resizing)
        dummy_input = torch.zeros(self.batch_size, self.frame_stack, 84, 84)

        # Initialize conv1 with proper input dimensions
        self.conv1._initialize_parameters(dummy_input)

        # Get actual output dimensions from conv1
        with torch.no_grad():
            conv1_out = self.conv1(dummy_input)

        # Initialize conv2 with actual conv1 output dimensions
        self.conv2._initialize_parameters(conv1_out)

        # Get actual output dimensions from conv2
        with torch.no_grad():
            conv2_out = self.conv2(conv1_out)

        # Initialize conv3 with actual conv2 output dimensions
        self.conv3._initialize_parameters(conv2_out)

    def forward(self, x):
        x = x.to(self.device)
        # Convert from (B,H,W,C) to (B,C,H,W) and normalize
        if x.dim() == 4:  # Batch input
            x = x.permute(0, 3, 1, 2).float() / 255.0
        else:  # Single input
            x = x.permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Handle frame stacking if needed
        if x.size(1) == 3 and self.frame_stack > 1:
            # Repeat current frame to match frame_stack
            x = x.repeat(1, self.frame_stack // 3 + 1, 1, 1)[:, : self.frame_stack]

        # Resize input to expected 84x84 dimensions
        if x.size(-2) != 84 or x.size(-1) != 84:
            x = torch.nn.functional.interpolate(
                x, size=(84, 84), mode="bilinear", align_corners=False
            )
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, frame_stack, height, width)
            
        Returns:
            Q-values for each action
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flatten with actual dimensions
        x = x.view(x.size(0), -1)
        if self.fc is None:
            # Initialize fc layer with correct input size
            self.fc = nn.Linear(x.size(1), 512).to(self.device)
        x = torch.sigmoid(self.fc(x))
        return self.head(x)

    def reset_bias(self):
        """Reset the bias maps for all dynamic bias layers in the network"""
        for module in self.children():
            if hasattr(module, "reset_bias"):
                module.reset_bias()

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Return random valid action as tensor (0-3 for Breakout)
            return torch.tensor(random.randint(0, 3), device=self.device)
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            # Get top action and clamp to Breakout's action space (0-3)
            return torch.clamp(q_values.argmax(), 0, 3)

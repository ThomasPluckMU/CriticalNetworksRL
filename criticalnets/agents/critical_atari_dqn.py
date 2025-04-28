import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict
from . import BaseAtariAgent

from ..utils.numerical_helpers import criticality_regularization


class CriticalAgent(BaseAtariAgent):
    """
    Atari agent implementing the Edge of Chaos regularization from the paper.
    This agent aims to maintain network criticality by constraining the Jacobian norm.
    """

    def __init__(self, config: Dict, action_space: int):
        """
        Initialize the Critical Agent network

        Args:
            config (Dict): Configuration dictionary
            action_space (int): Number of possible actions in the environment
        """
        super().__init__(config, action_space)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)
        self.reg_strength = config.get("reg_strength", 1e0)
        self.epsilon = config.get("epsilon", 0.1)

        # Define standard convolutional layers (without dynamic bias)
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(self.device)

        self.activation_function = F.tanh

        # Fully connected layers - will be initialized after first forward pass
        self.fc = None
        self.head = nn.Linear(512, action_space).to(self.device)

        # Track activation values for regularization
        self.saved_activations = {}
        self.saved_inputs = {}

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Q-values for each action
        """
        x = x.to(self.device).float() / 255.0  # Simple normalization only
        
        # Handle parallel input shapes [1,32,4,84,84] or [32,4,84,84]
        if x.dim() == 5:
            x = x.squeeze(0)  # Remove batch dimension
        elif x.dim() != 4:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
        
        # Save input for regularization
        self.saved_inputs["input"] = x

        # First convolutional layer with self.activation_function
        z1 = self.conv1(x)
        self.saved_activations["conv1"] = {
            "x": x,
            "z": z1,
            "a": self.activation_function(z1),
            "model": self.conv1,
        }
        a1 = self.activation_function(z1)

        # Second convolutional layer with self.activation_function
        z2 = self.conv2(a1)
        self.saved_activations["conv2"] = {
            "x": a1,
            "z": z2,
            "a": self.activation_function(z2),
            "model": self.conv2,
        }
        a2 = self.activation_function(z2)

        # Third convolutional layer with self.activation_function
        z3 = self.conv3(a2)
        self.saved_activations["conv3"] = {
            "x": a2,
            "z": z3,
            "a": self.activation_function(z3),
            "model": self.conv3,
        }
        a3 = self.activation_function(z3)

        # Flatten with actual dimensions
        x_flat = a3.view(a3.size(0), -1)

        if self.fc is None:
            # Initialize fc layer with correct input size
            self.fc = nn.Linear(x_flat.size(1), 512).to(self.device)

        # Fully connected layer
        z4 = self.fc(x_flat)
        self.saved_activations["fc"] = {
            "x": x_flat,
            "z": z4,
            "a": F.sigmoid(z4),
            "model": self.fc,
        }
        a4 = F.sigmoid(z4)

        # Output layer
        output = self.head(a4)

        return output

    def act(self, states):
        """Select actions using epsilon-greedy policy"""
        with torch.no_grad():
            q_values = self.forward(states)
            actions = q_values.argmax(dim=1)
            if self.epsilon > 0:
                mask = torch.rand(actions.size(0)) < self.epsilon
                actions[mask] = torch.randint(0, 6, (mask.sum(),), device=self.device)
            return actions

    def get_metrics(self):
        """
        Get metrics for this agent

        Returns:
            Dictionary of metrics
        """
        metrics = super().get_metrics() if hasattr(super(), "get_metrics") else {}

        # Add criticality metrics
        criticality_total = 0
        for name, saved in self.saved_activations.items():
            if name in ["conv1", "conv2", "conv3", "fc"]:
                # Ensure tensors have gradients preserved
                z = saved["z"].detach().requires_grad_(True)
                x = saved["x"].detach().requires_grad_(True)
                model = saved["model"]

                # For all layers
                # Number of neurons in this layer
                if name in ["conv1", "conv2", "conv3"]:
                    N = z.size(1) * z.size(2) * z.size(3)
                    ltype = name[:-1]
                else:  # 'fc'
                    N = z.size(1)
                    ltype = "fc"

                # Compute approximation of Jacobian norm
                activation_func = (
                    self.activation_function if name != "fc" else F.sigmoid
                )
                # Calculate how close we are to N
                criticality_total += criticality_regularization(
                    model, x, activation_func, ltype
                )

        # Convert tensor to scalar for logging
        metrics[f"criticality_loss"] = self.reg_strength * criticality_total.item()

        return metrics

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from . import BaseAtariAgent

from ..utils.numerical_helpers import criticality_regularization


class CriticalA2C(BaseAtariAgent):
    """
    Advantage Actor-Critic (A2C) agent implementing the Edge of Chaos regularization.
    This agent aims to maintain network criticality by constraining the Jacobian norm,
    while using the A2C algorithm architecture with actor and critic heads.
    """

    def __init__(self, config: Dict, action_space: int):
        """
        Initialize the Critical A2C Agent network

        Args:
            config (Dict): Configuration dictionary
            action_space (int): Number of possible actions in the environment
        """
        # Always force 6 actions for Pong regardless of passed action_space
        super().__init__(config, action_space=6)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)
        self.reg_strength = config.get("reg_strength", 1e0)

        # Define standard convolutional layers (without dynamic bias)
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(self.device)

        self.activation_function = F.tanh

        # Fully connected layers - will be initialized after first forward pass
        self.fc = None
        # Actor and critic heads
        self.actor = nn.Linear(512, 6).to(self.device)  # 6 actions for Pong
        self.critic = nn.Linear(512, 1).to(self.device)

        # Track activation values for regularization
        self.saved_activations = {}
        self.saved_inputs = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple of (policy logits, value estimate)
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
            # Initialize fc layer with correct input size on same device
            self.fc = nn.Linear(x_flat.size(1), 512).to(self.device)
            # Ensure actor/critic are on same device
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)

        # Fully connected layer
        z4 = self.fc(x_flat)
        self.saved_activations["fc"] = {
            "x": x_flat,
            "z": z4,
            "a": self.activation_function(z4),
            "model": self.fc,
        }
        a4 = self.activation_function(z4)

        # Actor and critic outputs
        logits = self.actor(a4)
        value = self.critic(a4)

        return logits, value

    def act(self, states: torch.Tensor) -> torch.Tensor:
        """
        Select actions based on current policy

        Args:
            states: Batch of state observations

        Returns:
            Tensor of selected actions
        """
        with torch.no_grad():
            logits, _ = self.forward(states)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the value, log probability, and entropy of actions

        Args:
            states: Batch of states
            actions: Batch of actions to evaluate

        Returns:
            Tuple of (values, log_probs, entropy)
        """
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values.squeeze(-1), log_probs, entropy

    def get_metrics(self):
        """
        Get metrics for this agent including criticality regularization

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
                activation_func = self.activation_function
                # Calculate how close we are to N
                criticality_total += criticality_regularization(
                    model, x, activation_func, ltype
                )

        # Convert tensor to scalar for logging
        metrics[f"criticality_loss"] = self.reg_strength * criticality_total.item()

        return metrics

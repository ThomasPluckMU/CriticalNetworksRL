from typing import Dict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseAtariAgent
from criticalnets.layers import GatedDynamicBiasCNN

from ..utils.numerical_helpers import criticality_regularization


class GatedCriticalAtariUDQN(BaseAtariAgent):

    def __init__(self, config: Dict, action_space: int):
        """
        Initialize the Gated DQN network with dynamic bias layers and criticality regularization

        Args:
            config (Dict): Configuration dictionary
            action_space (int): Number of possible actions in the environment
        """
        super().__init__(config, action_space)
        self.device = torch.device(config.get("device", "cpu"))
        self.frame_stack = config.get("frame_stack", 4)
        self.reg_strength = config.get("reg_strength", 1e0)
        self.epsilon = config.get("epsilon", 0.1)
        self.batch_size = config.get("batch_size", 32)

        # Define the activation function
        self.activation_function = F.tanh

        # Define the convolutional layers with gated dynamic bias
        self.conv1 = GatedDynamicBiasCNN(self.frame_stack, 32, kernel_size=8, stride=4).to(self.device)
        self.conv2 = GatedDynamicBiasCNN(32, 64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = GatedDynamicBiasCNN(64, 64, kernel_size=3, stride=1).to(self.device)

        # Fully connected layers - will be initialized after first forward pass
        self.fc = None
        self.head = nn.Linear(512, action_space).to(self.device)

        # Track activation values for regularization
        self.saved_activations = {}
        self.saved_inputs = {}

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
        """
        Forward pass through the network with activation tracking for criticality

        Args:
            x: Input tensor of shape (batch_size, height, width, channels)

        Returns:
            Q-values for each action
        """
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

        # Save input for regularization
        self.saved_inputs["input"] = x

        # First convolutional layer with activation tracking
        z1 = self.conv1(x)
        self.saved_activations["conv1"] = {
            "x": x,
            "z": z1,
            "a": self.activation_function(z1),
            "model": self.conv1,
        }
        a1 = self.activation_function(z1)

        # Second convolutional layer with activation tracking
        z2 = self.conv2(a1)
        self.saved_activations["conv2"] = {
            "x": a1,
            "z": z2,
            "a": self.activation_function(z2),
            "model": self.conv2,
        }
        a2 = self.activation_function(z2)

        # Third convolutional layer with activation tracking
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

        # Fully connected layer with activation tracking
        z4 = self.fc(x_flat)
        self.saved_activations["fc"] = {
            "x": x_flat,
            "z": z4,
            "a": torch.sigmoid(z4),
            "model": self.fc,
        }
        a4 = torch.sigmoid(z4)

        # Output layer
        return self.head(a4)

    def reset_bias(self):
        """Reset the bias maps for all dynamic bias layers in the network"""
        for module in self.children():
            if hasattr(module, "reset_bias"):
                module.reset_bias()

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Return random valid action
            return random.randint(0, 5)
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            # Get top action within the valid action space
            return min(q_values.argmax().item(), 5)

    def get_metrics(self):
        """
        Get metrics for this agent, including criticality metrics

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

                # Compute criticality regularization
                activation_func = (
                    self.activation_function if name != "fc" else torch.sigmoid
                )
                criticality_total += criticality_regularization(
                    model, x, activation_func, ltype
                )

        # Convert tensor to scalar for logging
        metrics["criticality_loss"] = self.reg_strength * criticality_total.item()

        return metrics

    def compute_loss(self, *args, **kwargs):
        """
        Compute loss with criticality regularization

        Returns:
            Total loss with criticality regularization
        """
        # Compute the standard loss
        base_loss = (
            super().compute_loss(*args, **kwargs)
            if hasattr(super(), "compute_loss")
            else 0
        )

        # Add criticality metrics
        criticality_total = 0
        for name, saved in self.saved_activations.items():
            if name in ["conv1", "conv2", "conv3", "fc"]:
                # Ensure tensors have gradients preserved
                z = saved["z"]
                x = saved["x"]
                model = saved["model"]

                # For all layers
                # Set layer type for criticality function
                if name in ["conv1", "conv2", "conv3"]:
                    ltype = name[:-1]
                else:  # 'fc'
                    ltype = "fc"

                # Compute criticality regularization
                activation_func = (
                    self.activation_function if name != "fc" else torch.sigmoid
                )
                criticality_total += criticality_regularization(
                    model, x, activation_func, ltype
                )

        # Add regularization term to the loss
        total_loss = base_loss + self.reg_strength * criticality_total

        return total_loss

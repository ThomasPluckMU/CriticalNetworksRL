import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict
from . import BaseAtariAgent

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
        self.device = torch.device(config.get('device', 'cpu'))
        self.frame_stack = config.get('frame_stack', 4)
        self.reg_strength = config.get('reg_strength', 0.01)
        
        # Define standard convolutional layers (without dynamic bias)
        self.conv1 = nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers - will be initialized after first forward pass
        self.fc = None
        self.head = nn.Linear(512, action_space)
        
        # Track activation values for regularization
        self.saved_activations = {}
        self.saved_inputs = {}
        
    def forward(self, x):
        """
        Forward pass through the network
        
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
            x = x.repeat(1, self.frame_stack // 3 + 1, 1, 1)[:, :self.frame_stack]
        
        # Resize input to expected 84x84 dimensions
        if x.size(-2) != 84 or x.size(-1) != 84:
            x = torch.nn.functional.interpolate(x, size=(84, 84), mode='bilinear', align_corners=False)
        
        # Save input for regularization
        self.saved_inputs['input'] = x
        
        # First convolutional layer with ReLU
        z1 = self.conv1(x)
        self.saved_activations['conv1'] = {'z': z1, 'a': F.relu(z1)}
        a1 = F.relu(z1)
        
        # Second convolutional layer with ReLU
        z2 = self.conv2(a1)
        self.saved_activations['conv2'] = {'z': z2, 'a': F.relu(z2)}
        a2 = F.relu(z2)
        
        # Third convolutional layer with ReLU
        z3 = self.conv3(a2)
        self.saved_activations['conv3'] = {'z': z3, 'a': F.relu(z3)}
        a3 = F.relu(z3)
        
        # Flatten with actual dimensions
        x_flat = a3.view(a3.size(0), -1)
        
        if self.fc is None:
            # Initialize fc layer with correct input size
            self.fc = nn.Linear(x_flat.size(1), 512).to(self.device)
            
        # Fully connected layer
        z4 = self.fc(x_flat)
        self.saved_activations['fc'] = {'z': z4, 'a': torch.sigmoid(z4)}
        a4 = torch.sigmoid(z4)
        
        # Output layer
        output = self.head(a4)
        
        return output

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.config.get('epsilon', 0.1):
            # Return random valid action (0-3 for Breakout)
            return random.randint(0, 5)
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            # Get top action and clamp to Breakout's action space (0-3)
            return min(q_values.argmax().item(), 5)



    def calculate_criticality_regularization(self):
        """
        Calculate the Edge of Chaos regularization term:
        R(layer) = (2σ′′(z)∇²ₓσ(z)/√N) * (1/N - 1/||∇ₓσ(z)||)
        
        Returns:
            Regularization loss
        """
        reg_loss = 0.0
        
        for name, saved in self.saved_activations.items():
            if name in ['conv1', 'conv2', 'conv3', 'fc']:
                z = saved['z']
                
                # For ReLU layers (convolutional layers)
                if name in ['conv1', 'conv2', 'conv3']:
                    # Compute σ'(z) (derivative of ReLU)
                    sigma_prime = (z > 0).float()
                    
                    # Compute σ''(z) (second derivative of ReLU)
                    # Note: For ReLU, σ''(z) is zero almost everywhere except at z=0
                    # We use a small epsilon approximation to avoid zero gradients
                    epsilon = 1e-5
                    sigma_double_prime = torch.zeros_like(z)
                    sigma_double_prime[(z.abs() < epsilon)] = 1.0 / epsilon
                    
                    # Number of neurons in this layer
                    N = z.size(1) * z.size(2) * z.size(3)
                    
                    # Compute Jacobian norm approximation
                    jacobian_norm = torch.sqrt(torch.sum(sigma_prime**2))
                    
                    # Compute Laplacian approximation 
                    laplacian = torch.sum(sigma_double_prime)
                    
                    # Compute the regularization term
                    layer_reg = (2 * sigma_double_prime * laplacian / torch.sqrt(torch.tensor(N, device=z.device))) * \
                               (1.0 / N - 1.0 / jacobian_norm)
                               
                    reg_loss += torch.abs(layer_reg.mean())
                
                # For Sigmoid layer (fc layer)
                elif name == 'fc':
                    # Compute σ'(z) (derivative of sigmoid)
                    a = torch.sigmoid(z)
                    sigma_prime = a * (1 - a)
                    
                    # Compute σ''(z) (second derivative of sigmoid)
                    sigma_double_prime = a * (1 - a) * (1 - 2 * a)
                    
                    # Number of neurons in this layer
                    N = z.size(1)
                    
                    # Compute Jacobian norm approximation
                    jacobian_norm = torch.sqrt(torch.sum(sigma_prime**2))
                    
                    # Compute Laplacian approximation
                    laplacian = torch.sum(sigma_double_prime)
                    
                    # Compute the regularization term
                    layer_reg = (2 * sigma_double_prime * laplacian / torch.sqrt(torch.tensor(N, device=z.device))) * \
                               (1.0 / N - 1.0 / jacobian_norm)
                               
                    reg_loss += torch.abs(layer_reg.mean())
        
        return reg_loss * self.reg_strength
    
    def get_metrics(self):
        """
        Get metrics for this agent
        
        Returns:
            Dictionary of metrics
        """
        metrics = super().get_metrics() if hasattr(super(), 'get_metrics') else {}
        
        # Add criticality metrics
        with torch.no_grad():
            for name, saved in self.saved_activations.items():
                if name in ['conv1', 'conv2', 'conv3', 'fc']:
                    z = saved['z']
                    
                    # For convolutional layers
                    if name in ['conv1', 'conv2', 'conv3']:
                        # Number of neurons in this layer
                        N = z.size(1) * z.size(2) * z.size(3)
                        
                        # Compute σ'(z) (derivative of ReLU)
                        sigma_prime = (z > 0).float()
                        
                        # Compute Jacobian norm
                        jacobian_norm = torch.sqrt(torch.sum(sigma_prime**2))
                        
                        # Calculate how close we are to N
                        criticality_metric = (jacobian_norm**2 / N - 1.0).abs().item()
                        metrics[f'{name}_criticality'] = criticality_metric
                        
                    # For fc layer
                    elif name == 'fc':
                        # Number of neurons in this layer
                        N = z.size(1)
                        
                        # Compute σ'(z) (derivative of sigmoid)
                        a = torch.sigmoid(z)
                        sigma_prime = a * (1 - a)
                        
                        # Compute Jacobian norm
                        jacobian_norm = torch.sqrt(torch.sum(sigma_prime**2))
                        
                        # Calculate how close we are to N
                        criticality_metric = (jacobian_norm**2 / N - 1.0).abs().item()
                        metrics[f'{name}_criticality'] = criticality_metric
        
        return metrics
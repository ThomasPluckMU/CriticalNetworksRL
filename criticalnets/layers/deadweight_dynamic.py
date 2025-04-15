import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg
from .dynamic_bias import DynamicBiasBase

class DeadWeightDynamicBiasNN(DynamicBiasBase):
    def __init__(self, input_size, hidden_size, spectral_radius=0.99, connectivity=1.0, seed=None):
        super().__init__(input_size, hidden_size)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        W = self._initialize_reservoir_weights(hidden_size, input_size, spectral_radius, connectivity)
        self.weight = nn.Parameter(W, requires_grad=False)
        # Initialize velocity parameters with larger scale
        self.velocity_weight = nn.Parameter(torch.randn(input_size, 1) * 0.1)
        self.velocity_bias = nn.Parameter(torch.randn(1) * 0.1)
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        # Initialize weights for better gradient flow
        nn.init.kaiming_normal_(self.velocity_weight, mode='fan_in')
    
    def _initialize_reservoir_weights(self, out_size, in_size, spectral_radius, connectivity):
        W_np = np.random.randn(out_size, in_size) * (np.random.rand(out_size, in_size) < connectivity)
        
        if out_size != in_size:
            max_dim = max(out_size, in_size)
            square_matrix = np.zeros((max_dim, max_dim))
            square_matrix[:out_size, :in_size] = W_np
            eigenvalues = linalg.eigvals(square_matrix)
        else:
            eigenvalues = linalg.eigvals(W_np)
            
        max_abs_eigenvalue = max(abs(eigenvalues))
        if max_abs_eigenvalue > 0:
            W_np = W_np * (spectral_radius / max_abs_eigenvalue)
            
        return torch.tensor(W_np, dtype=torch.float32)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.current_bias is None:
            self.current_bias = self._init_bias(batch_size)
        
        velocity = self.gelu(torch.matmul(x, self.velocity_weight) + self.velocity_bias)
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        activation = self.selu(z)
        
        # Update bias with stronger velocity effect
        self.current_bias = self.current_bias - velocity * activation
        
        return activation

    def reset_bias(self):
        """Reset current bias to initialized state with noise"""
        if hasattr(self, 'current_bias') and self.current_bias is not None:
            batch_size = self.current_bias.shape[0]
            self.current_bias = self._init_bias(batch_size)
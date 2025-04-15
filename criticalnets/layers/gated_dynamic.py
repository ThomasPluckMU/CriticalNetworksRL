import torch
import torch.nn as nn
import math
from .dynamic_bias import DynamicBiasBase

class GatedDynamicBiasNN(DynamicBiasBase):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity_weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)/2)
        nn.init.kaiming_uniform_(self.velocity_weight, a=math.sqrt(5)/2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.current_bias is None:
            self.current_bias = self._init_bias(batch_size)
            # Add small random noise to ensure unique initialization
            self.current_bias += torch.randn_like(self.current_bias) * 0.01
        
        velocity = self.gelu(torch.matmul(x, self.velocity_weight.t()))
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        activation = self.selu(z)
        
        # Update bias with stronger velocity effect
        self.current_bias = self.current_bias - velocity * activation * 0.5
        
        return activation

    def reset_bias(self):
        """Reset current bias to initialized state with noise"""
        if hasattr(self, 'current_bias') and self.current_bias is not None:
            batch_size = self.current_bias.shape[0]
            self.current_bias = self._init_bias(batch_size)
            self.current_bias += torch.randn_like(self.current_bias) * 0.01

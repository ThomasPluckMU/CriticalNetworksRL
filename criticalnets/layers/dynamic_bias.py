import torch
import torch.nn as nn
import math

class DynamicBiasBase(nn.Module):
    """Base class for all dynamic bias layers"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.current_bias = None
        
    def reset_bias(self):
        """Reset the current bias state"""
        self.current_bias = None
        
    def _init_bias(self, batch_size):
        """Initialize bias tensor for a new sequence"""
        return torch.zeros(batch_size, self.hidden_size)
        
    def forward(self, x):
        """Base forward pass with dynamic bias update"""
        raise NotImplementedError("Subclasses must implement forward()")

class DynamicBiasNN(DynamicBiasBase):
    def __init__(self, input_size, hidden_size, velocity_init=0.1):
        super().__init__(input_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity = nn.Parameter(torch.Tensor(hidden_size).fill_(velocity_init))
        self.selu = nn.SELU()
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.current_bias is None:
            self.current_bias = self._init_bias(batch_size)
            
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        activation = self.selu(z)
        self.current_bias -= self.velocity * activation
        
        return activation

import torch
import torch.nn as nn
import math

class DynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, velocity_init=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        
        self.velocity = None
        self.current_bias_maps = None
        self._velocity_init = velocity_init
        self.selu = nn.SELU()
        
    def _initialize_parameters(self, out_shape=None):
        if out_shape is not None:
            self.batch_size, self.out_channels, self.out_height, self.out_width = out_shape
        elif hasattr(self, 'out_channels'):
            out_shape = (self.batch_size, self.out_channels, self.out_height, self.out_width)
        else:
            raise ValueError("Output shape must be provided for first initialization")
        
        self.velocity = nn.Parameter(torch.full(
            (self.out_channels, self.out_height, self.out_width), 
            self._velocity_init
        ))
        self.current_bias_maps = torch.zeros(
            self.batch_size, self.out_channels, self.out_height, self.out_width
        )
        
    def reset_bias(self):
        self._initialize_parameters()
    
    def forward(self, x):
        conv_output = self.conv(x)
        
        if self.current_bias_maps is None:
            self._initialize_parameters(conv_output.shape)
            
        z = conv_output + self.current_bias_maps
        activation = self.selu(z)
        velocity_expanded = self.velocity.unsqueeze(0).expand_as(activation)
        self.current_bias_maps -= velocity_expanded * (activation + 0.1)
        
        return activation

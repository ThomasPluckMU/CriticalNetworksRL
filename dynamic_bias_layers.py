import torch
import torch.nn as nn
import math

class DynamicBiasNN(nn.Module):
    def __init__(self, input_size, hidden_size, base_init=-1, velocity_init=0.1):
        super(DynamicBiasNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.base = nn.Parameter(torch.Tensor(hidden_size).fill_(base_init))
        self.velocity = nn.Parameter(torch.Tensor(hidden_size).fill_(velocity_init))
        
        # This will not be a parameter but a state that we track
        self.current_bias = None
        
        self.reset_parameters()
        self.selu = nn.SELU()
        
    def reset_parameters(self):
        # Initialize main weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def reset_bias(self):
        pass
    
    def forward(self, x):
        """
        Forward pass with dynamic bias update
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            activation: Output after activation function
        """
        batch_size = x.shape[0]
        
        # Initialize or retrieve bias for this batch
        if self.current_bias is None:
            # Create a new bias tensor for this sequence
            # Each sequence in the batch gets its own bias
            self.current_bias = self.base.expand(batch_size, -1).clone()
        
        # Main path: compute pre-activation 
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        
        # Apply activation
        activation = self.selu(z)
            
        # Update bias based on activation 
        self.current_bias -= self.velocity * activation
                
        # Return activation plus residual connection
        return activation
    
class DynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 base_init=-1, velocity_init=0.1):
        super(DynamicBiasCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Conv layer without bias (we'll handle bias separately)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        
        # Initialize learnable parameters
        # These will be properly initialized during the first forward pass
        # once we know the output spatial dimensions
        self.base = None
        self.velocity = None
        self.current_bias_maps = None
        
        # Store initialization values
        self._base_init = base_init
        self._velocity_init = velocity_init
        
        # Activation function
        self.selu = nn.SELU()
        
    def _initialize_parameters(self, out_shape):
        """Initialize parameters based on output shape"""
        _, out_channels, out_height, out_width = out_shape
        
        # Create learnable parameters with proper shapes
        self.base = nn.Parameter(torch.full((out_channels, out_height, out_width),
                                           float(self._base_init), dtype=torch.float32))
        self.velocity = nn.Parameter(torch.full((out_channels, out_height, out_width),
                                               float(self._velocity_init), dtype=torch.float32))
        
    def reset_bias(self):
        """Maintained for compatibility, but no longer needed"""
        pass
    
    def forward(self, x):
        """
        Forward pass with spatially-aware dynamic bias update
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            activation: Output after activation with dynamic bias
        """
        batch_size = x.shape[0]
        
        # Apply convolution (without bias)
        conv_output = self.conv(x)
        
        # Initialize parameters if this is the first forward pass
        if self.base is None:
            self._initialize_parameters(conv_output.shape)
        
        # Create fresh bias maps for this forward pass
        bias_maps = self.base.expand(batch_size, -1, -1, -1)
        
        # Add bias maps to convolution output
        z = conv_output + bias_maps
        
        # Apply activation
        activation = self.selu(z)
        
        # Compute updated bias maps (but don't store them)
        velocity_expanded = self.velocity.expand(batch_size, -1, -1, -1)
        updated_bias_maps = bias_maps - velocity_expanded * activation
        
        return activation

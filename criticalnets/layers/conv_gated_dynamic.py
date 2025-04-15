import torch
import torch.nn as nn
import math

class GatedDynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ema_factor=0.9):
        """
        Initialize the Gated Dynamic Bias Convolutional Neural Network
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolutional kernel
            stride (int or tuple): Stride of the convolution (default: 1)
            padding (int or tuple): Padding added to all sides of the input (default: 0)
            ema_factor (float): The decay factor for the exponential moving average (default: 0.9)
        """
        super(GatedDynamicBiasCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.ema_factor = ema_factor
        
        # Main convolution layer without bias (we'll handle bias separately)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        
        # Velocity computation convolution layer
        self.velocity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5)/2)
        nn.init.kaiming_uniform_(self.velocity_conv.weight, a=math.sqrt(5)/2)
        
        # Output shape tracking
        self.out_height = None
        self.out_width = None
        self.batch_size = None
        
        # Initialize learnable parameters
        # These will be properly initialized during the first forward pass
        self.base = None
        self.current_bias_maps = None
        
        # Activation functions
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        
    def _initialize_parameters(self, x):
        """Initialize parameters based on input and output shape"""
        # Calculate output dimensions
        batch_size, _, h_in, w_in = x.shape
        h_out = ((h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        w_out = ((w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        
        # Store output dimensions
        self.batch_size = batch_size
        self.out_height = h_out
        self.out_width = w_out
        
        # Create learnable base parameter
        self.base = nn.Parameter(
            torch.zeros(1, self.out_channels, self.out_height, self.out_width)
        )
        
        # Initialize current bias maps
        self.current_bias_maps = self.base.expand(batch_size, -1, -1, -1).data.clone()
                
    def reset_bias(self):
        """Reset current bias maps to base values"""
        if self.base is not None and self.out_height is not None:
            self.current_bias_maps = self.base.expand(self.batch_size, -1, -1, -1).data.clone()
        else:
            self.current_bias_maps = None
    
    def forward(self, x):
        """
        Forward pass with spatially-aware gated dynamic bias update
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            activation: Output after activation with dynamic bias
        """
        batch_size = x.shape[0]
        
        # Initialize parameters if not done already or if batch size changed
        if self.base is None or self.current_bias_maps is None or batch_size != self.batch_size:
            self._initialize_parameters(x)
        elif batch_size != self.current_bias_maps.size(0):
            # Handle batch size changes by expanding the current bias maps
            self.batch_size = batch_size
            self.current_bias_maps = self.base.expand(batch_size, -1, -1, -1).clone()
        
        # Apply main convolution (without bias)
        conv_output = self.conv(x)
        
        # Compute dynamic velocity using GELU-gated computation
        velocity = self.gelu(self.velocity_conv(x))
        
        # Add bias maps to convolution output
        z = conv_output + self.current_bias_maps
        
        # Apply activation
        activation = self.selu(z)
        
        # Update current_bias_maps with EMA toward base, pushed by velocity*activation
        # Formula: current_bias_maps = ema_factor * current_bias_maps + (1 - ema_factor) * base - velocity * activation
        self.current_bias_maps = (
            self.ema_factor * self.current_bias_maps + 
            (1 - self.ema_factor) * self.base - 
            velocity * activation
        )
        
        return activation

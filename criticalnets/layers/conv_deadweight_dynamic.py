import torch
import torch.nn as nn
import numpy as np
from scipy import linalg

class DeadWeightDynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 spectral_radius=0.99, connectivity=1.0, seed=None):
        """
        Initialize the DeadWeight Gated Dynamic Bias CNN with frozen weights
        initialized according to reservoir computing principles.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolutional kernel
            stride (int or tuple): Stride of the convolution (default: 1)
            padding (int or tuple): Padding added to all sides of the input (default: 0)
            spectral_radius (float): Spectral radius for weight initialization (default: 0.99)
            connectivity (float): Proportion of non-zero weights (default: 1.0)
            seed (int): Random seed for reproducibility (default: None)
        """
        super(DeadWeightDynamicBiasCNN, self).__init__()
        
        # Set seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Network parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Create Conv2d layer first to get the shape right
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        
        # Initialize the conv weights using reservoir computing principles and freeze them
        W = self._initialize_reservoir_weights(out_channels, in_channels, 
                                             kernel_size, spectral_radius, connectivity)
        self.conv.weight = nn.Parameter(W, requires_grad=False)
        
        # Simple velocity computation layer (will remain trainable)
        self.velocity_conv = nn.Conv2d(in_channels, 1, kernel_size,
                                     stride=stride, padding=padding)
        
        # Initialize learnable parameters
        # These will be properly initialized during the first forward pass
        self.current_bias_maps = None
        
        # Activation functions
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        
    def _initialize_reservoir_weights(self, out_channels, in_channels, kernel_size, spectral_radius, connectivity):
        """
        Initialize convolutional weights according to reservoir computing principles.
        
        Args:
            out_channels (int): Number of output channels
            in_channels (int): Number of input channels
            kernel_size (tuple): Size of the kernel
            spectral_radius (float): Target spectral radius
            connectivity (float): Proportion of non-zero weights
        
        Returns:
            torch.Tensor: Properly initialized weight tensor
        """
        # Create sparse random tensor with specified connectivity
        if isinstance(kernel_size, tuple):
            k_height, k_width = kernel_size
        else:
            k_height, k_width = kernel_size, kernel_size
            
        # Generate random weights with proper connectivity
        W_np = np.random.randn(out_channels, in_channels, k_height, k_width) * \
               (np.random.rand(out_channels, in_channels, k_height, k_width) < connectivity)
        
        # For convolutional layers, we need to reshape to compute spectral radius
        W_flat = W_np.reshape(out_channels, -1)
        
        # Create a square matrix by padding with zeros if necessary
        max_dim = max(W_flat.shape)
        square_matrix = np.zeros((max_dim, max_dim))
        square_matrix[:W_flat.shape[0], :W_flat.shape[1]] = W_flat
        
        # Compute eigenvalues of the square matrix
        eigenvalues = linalg.eigvals(square_matrix)
        max_abs_eigenvalue = max(abs(eigenvalues))
        
        # Scale to desired spectral radius
        if max_abs_eigenvalue > 0:  # Avoid division by zero
            W_np = W_np * (spectral_radius / max_abs_eigenvalue)
        
        # Convert to PyTorch tensor
        W = torch.tensor(W_np, dtype=torch.float32)
        return W
    
    def _initialize_parameters(self, out_shape=None):
        """Initialize parameters based on output shape"""
        if out_shape is not None:
            self.batch_size, self.out_channels, self.out_height, self.out_width = out_shape
        elif hasattr(self, 'out_channels'):
            # Default to small shape that will be expanded in forward pass
            self.batch_size, self.out_height, self.out_width = 1, 1, 1
        else:
            raise ValueError("Output shape must be provided for first initialization")
            
        self.current_bias_maps = torch.zeros(
            self.batch_size, self.out_channels, self.out_height, self.out_width
        )
        
    def reset_bias(self):
        """Reset current bias maps to None, allowing reinitialization in next forward pass"""
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
        
        # Apply main convolution (without bias, weights are frozen)
        conv_output = self.conv(x)
        
        # Compute dynamic velocity using GELU-gated computation
        velocity = self.gelu(self.velocity_conv(x))
        # Expand velocity to match activation dimensions
        velocity = velocity.expand(-1, self.out_channels, -1, -1)
        
        # Initialize bias maps if needed
        if self.current_bias_maps is None:
            self._initialize_parameters(conv_output.shape)
            # Add small random noise to ensure unique initialization
            self.current_bias_maps += torch.randn_like(self.current_bias_maps) * 0.01
            
        # Add bias maps to convolution output
        z = conv_output + self.current_bias_maps
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias maps based on activation and dynamically computed velocity
        self.current_bias_maps -= velocity * activation
        
        return activation

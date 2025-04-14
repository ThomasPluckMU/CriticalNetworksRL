import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg
import math

class DynamicBiasNN(nn.Module):
    def __init__(self, input_size, hidden_size, velocity_init=0.1):
        super(DynamicBiasNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity = nn.Parameter(torch.Tensor(hidden_size).fill_(velocity_init))
        
        # This will not be a parameter but a state that we track
        self.current_bias = None
        
        self.reset_parameters()
        self.selu = nn.SELU()
        
    def reset_parameters(self):
        # Initialize main weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def reset_bias(self):
        # Reset function now clears the current_bias so it will be reinitialized
        # for the next sequence
        self.current_bias = None
    
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
            self.current_bias = torch.zeros(batch_size, self.hidden_size)
        
        # Main path: compute pre-activation 
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        
        # Apply activation
        activation = self.selu(z)
            
        # Update bias based on activation 
        self.current_bias -= self.velocity * activation
                
        # Return activation plus residual connection
        return activation

class GatedDynamicBiasNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialize the Gated Dynamic Bias Neural Network
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Number of hidden units
        """
        super(GatedDynamicBiasNN, self).__init__()
        
        # Core network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # Velocity computation parameters
        self.velocity_weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # Activation functions
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        
        # Tracking state
        self.current_bias = None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize network parameters using random Gaussian with SVD normalization
        """
        # Initialize main weights and velocity weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5)/2)
    
    def reset_bias(self):
        """
        Reset the current bias to None, allowing reinitialization in next forward pass
        """
        self.current_bias = None
    
    def forward(self, x):
        """
        Forward pass with dynamically gated bias
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output after activation
        """
        batch_size = x.shape[0]
        
        # Initialize or retrieve bias for this batch
        if self.current_bias is None:
            # Create a zero tensor for current bias
            self.current_bias = torch.zeros(batch_size, self.hidden_size)
        
        # Compute dynamic velocity using ReLU-gated computation
        velocity = self.gelu(torch.matmul(x, self.velocity_weight.t()))
        
        # Main path: compute pre-activation 
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias based on dynamically computed velocity
        self.current_bias -= velocity * activation

        
        return activation
    
class DynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 velocity_init=0.1):
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
        self.velocity = None
        self.current_bias_maps = None
        
        # Store initialization values
        self._velocity_init = velocity_init
        
        # Activation function
        self.selu = nn.SELU()
        
    def _initialize_parameters(self, out_shape = None):
        """Initialize parameters based on output shape"""
        if out_shape is not None:
            self.batch_size, self.out_channels, self.out_height, self.out_width = out_shape
        
        self.velocity = nn.Parameter(torch.full((self.out_channels, self.out_height, self.out_width), 
                                               self._velocity_init))
         # Initialize or retrieve bias for this batch
        self.current_bias = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)
        
    def reset_bias(self):
        """Reset current bias maps to base values"""
        self._initialize_parameters()
    
    def forward(self, x):
        """
        Forward pass with spatially-aware dynamic bias update
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            activation: Output after activation with dynamic bias
        """        
        # Apply convolution (without bias)
        conv_output = self.conv(x)
        
        # Add bias maps to convolution output
        z = conv_output + self.current_bias_maps
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias maps based on activation and velocity
        velocity_expanded = self.velocity.unsqueeze(0).expand_as(activation)
        self.current_bias_maps -= velocity_expanded * activation
        
        return activation

class DeadWeightDynamicBiasNN(nn.Module):
    def __init__(self, input_size, hidden_size, spectral_radius=0.99, connectivity=0.3, seed=None):
        """
        Initialize the DeadWeight Gated Dynamic Bias Neural Network with frozen weights
        initialized according to reservoir computing principles.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Number of hidden units
            spectral_radius (float): Spectral radius for weight initialization (default: 0.99)
            connectivity (float): Proportion of non-zero weights (default: 1.0)
            seed (int): Random seed for reproducibility (default: None)
        """
        super(DeadWeightDynamicBiasNN, self).__init__()
        
        # Set seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Core network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize and freeze the main weights using reservoir computing principles
        W = self._initialize_reservoir_weights(hidden_size, input_size, spectral_radius, connectivity)
        self.weight = nn.Parameter(W, requires_grad=False)
        
        # Initialize and freeze the velocity weights using reservoir computing principles
        self.velocity_weight = nn.Parameter(torch.Tensor(input_size,1))
        self.velocity_bias = nn.Parameter(torch.Tensor(1))
        
        # Activation functions
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        
        # Tracking state
        self.current_bias = None
    
    def _initialize_reservoir_weights(self, out_size, in_size, spectral_radius, connectivity):
        """
        Initialize weights according to reservoir computing principles from RMT.
        
        Args:
            out_size (int): Output dimension
            in_size (int): Input dimension
            spectral_radius (float): Target spectral radius
            connectivity (float): Proportion of non-zero weights
        
        Returns:
            torch.Tensor: Properly initialized weight tensor
        """
        # Create sparse random matrix with specified connectivity
        W_np = np.random.randn(out_size, in_size) * (np.random.rand(out_size, in_size) < connectivity)
        
        # For rectangular matrices, we need to compute SVD to get an estimate of the spectral radius
        if out_size != in_size:
            # Create a square matrix by padding with zeros or truncating
            max_dim = max(out_size, in_size)
            square_matrix = np.zeros((max_dim, max_dim))
            square_matrix[:out_size, :in_size] = W_np
            
            # Compute eigenvalues of the square matrix
            eigenvalues = linalg.eigvals(square_matrix)
            max_abs_eigenvalue = max(abs(eigenvalues))
        else:
            # Direct eigenvalue computation for square matrices
            eigenvalues = linalg.eigvals(W_np)
            max_abs_eigenvalue = max(abs(eigenvalues))
        
        # Scale to desired spectral radius
        if max_abs_eigenvalue > 0:  # Avoid division by zero
            W_np = W_np * (spectral_radius / max_abs_eigenvalue)
        
        # Convert to PyTorch tensor
        W = torch.tensor(W_np, dtype=torch.float32)
        return W
    
    def reset_bias(self):
        """
        Reset the current bias to None, allowing reinitialization in next forward pass
        """
        self.current_bias = None
    
    def forward(self, x):
        """
        Forward pass with dynamically gated bias
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output after activation
        """
        batch_size = x.shape[0]
        
         # Initialize or retrieve bias for this batch
        if self.current_bias is None:
            # Create a tensor for current bias from the learnable base
            self.current_bias = torch.zeros(batch_size, self.hidden_size)
        
        # Compute dynamic velocity using GELU-gated computation
        velocity = self.gelu(torch.matmul(x, self.velocity_weight)+self.velocity_bias)
        
        # Main path: compute pre-activation 
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias based on dynamically computed velocity
        self.current_bias -= velocity * activation
        
        return activation
class DeadWeightDynamicBiasNN(nn.Module):
    def __init__(self, input_size, hidden_size, spectral_radius=0.99, connectivity=1.0, seed=None):
        """
        Initialize the DeadWeight Gated Dynamic Bias Neural Network with frozen weights
        initialized according to reservoir computing principles.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Number of hidden units
            spectral_radius (float): Spectral radius for weight initialization (default: 0.99)
            connectivity (float): Proportion of non-zero weights (default: 1.0)
            seed (int): Random seed for reproducibility (default: None)
        """
        super(DeadWeightDynamicBiasNN, self).__init__()
        
        # Set seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Core network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize and freeze the main weights using reservoir computing principles
        W = self._initialize_reservoir_weights(hidden_size, input_size, spectral_radius, connectivity)
        self.weight = nn.Parameter(W, requires_grad=False)
        
        # Initialize and freeze the velocity weights using reservoir computing principles
        self.velocity_weight = nn.Parameter(torch.Tensor(input_size,1))
        self.velocity_bias = nn.Parameter(torch.Tensor(1))
        
        # Activation functions
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        
        # Tracking state
        self.current_bias = None
    
    def _initialize_reservoir_weights(self, out_size, in_size, spectral_radius, connectivity):
        """
        Initialize weights according to reservoir computing principles from RMT.
        
        Args:
            out_size (int): Output dimension
            in_size (int): Input dimension
            spectral_radius (float): Target spectral radius
            connectivity (float): Proportion of non-zero weights
        
        Returns:
            torch.Tensor: Properly initialized weight tensor
        """
        # Create sparse random matrix with specified connectivity
        W_np = np.random.randn(out_size, in_size) * (np.random.rand(out_size, in_size) < connectivity)
        
        # For rectangular matrices, we need to compute SVD to get an estimate of the spectral radius
        if out_size != in_size:
            # Create a square matrix by padding with zeros or truncating
            max_dim = max(out_size, in_size)
            square_matrix = np.zeros((max_dim, max_dim))
            square_matrix[:out_size, :in_size] = W_np
            
            # Compute eigenvalues of the square matrix
            eigenvalues = linalg.eigvals(square_matrix)
            max_abs_eigenvalue = max(abs(eigenvalues))
        else:
            # Direct eigenvalue computation for square matrices
            eigenvalues = linalg.eigvals(W_np)
            max_abs_eigenvalue = max(abs(eigenvalues))
        
        # Scale to desired spectral radius
        if max_abs_eigenvalue > 0:  # Avoid division by zero
            W_np = W_np * (spectral_radius / max_abs_eigenvalue)
        
        # Convert to PyTorch tensor
        W = torch.tensor(W_np, dtype=torch.float32)
        return W
    
    def reset_bias(self):
        """
        Reset the current bias to None, allowing reinitialization in next forward pass
        """
        self.current_bias = None
    
    def forward(self, x):
        """
        Forward pass with dynamically gated bias
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output after activation
        """
        batch_size = x.shape[0]
        
        # Initialize or retrieve bias for this batch
        if self.current_bias is None:
            # Create a tensor for current bias from the learnable base
            self.current_bias = torch.zeros(batch_size, self.hidden_size)
        
        # Compute dynamic velocity using GELU-gated computation
        velocity = self.gelu(torch.matmul(x, self.velocity_weight)+self.velocity_bias)
        
        # Main path: compute pre-activation 
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias based on dynamically computed velocity
        self.current_bias -= velocity * activation
        
        return activation

class GatedDynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize the Gated Dynamic Bias Convolutional Neural Network
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolutional kernel
            stride (int or tuple): Stride of the convolution (default: 1)
            padding (int or tuple): Padding added to all sides of the input (default: 0)
        """
        super(GatedDynamicBiasCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Main convolution layer without bias (we'll handle bias separately)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        
        # Velocity computation convolution layer
        self.velocity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5)/2)
        nn.init.kaiming_uniform_(self.velocity_conv.weight, a=math.sqrt(5)/2)
        
        # Initialize learnable parameters
        # These will be properly initialized during the first forward pass
        self.current_bias_maps = None
        
        # Activation functions
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        
    def _initialize_parameters(self, out_shape = None):
        """Initialize parameters based on output shape"""
        if out_shape is not None:
            self.batch_size, self.out_channels, self.out_height, self.out_width = out_shape
        self.current_bias_maps = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)
                
    def reset_bias(self):
        """Reset current bias maps to None, allowing reinitialization in next forward pass"""
        self._initialize_parameters()
    
    def forward(self, x):
        """
        Forward pass with spatially-aware gated dynamic bias update
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            activation: Output after activation with dynamic bias
        """
        batch_size = x.shape[0]
        
        # Apply main convolution (without bias)
        conv_output = self.conv(x)
        
        # Compute dynamic velocity using GELU-gated computation
        velocity = self.gelu(self.velocity_conv(x))
        
        # Add bias maps to convolution output
        z = conv_output + self.current_bias_maps
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias maps based on activation and dynamically computed velocity
        self.current_bias_maps -= velocity * activation
        
        return activation


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
    
    def _initialize_parameters(self, out_shape = None):
        """Initialize parameters based on output shape"""
        if out_shape is not None:
            self.batch_size, self.out_channels, self.out_height, self.out_width = out_shape
        self.current_bias_maps = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)
        
    def reset_bias(self):
        """Reset current bias maps to None, allowing reinitialization in next forward pass"""
        self._initialize_parameters()
    
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
        
        # Add bias maps to convolution output
        z = conv_output + self.current_bias_maps
        
        # Apply activation
        activation = self.selu(z)
        
        # Update bias maps based on activation and dynamically computed velocity
        self.current_bias_maps -= velocity * activation
        
        return activation
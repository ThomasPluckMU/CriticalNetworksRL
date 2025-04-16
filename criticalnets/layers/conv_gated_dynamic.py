import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

# Import your GatedDynamicBiasCNN class or define it here
# from your_module import GatedDynamicBiasCNN

# Define the class for debugging
import math

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
        self.gelu = nn.Sigmoid()
        
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
            nn.init.normal_(torch.zeros(1, self.out_channels, self.out_height, self.out_width))
        )
        
        # Initialize current bias maps
        self.current_bias_maps = self.base.expand(batch_size, -1, -1, -1).detach().clone()
                
    def reset_bias(self):
        """Reset current bias maps to base values"""
        if self.base is not None and self.out_height is not None:
            self.current_bias_maps = self.base.expand(self.batch_size, -1, -1, -1).detach().clone()
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
            self.current_bias_maps = self.current_bias_maps.detach().clone().expand(batch_size, -1, -1, -1)
        
        # Compute dynamic velocity using Sigmoid-gated computation
        velocity = self.gelu(self.velocity_conv(x))
        
        # Apply main convolution (without bias)
        conv_output = self.conv(x)
        
        # Calculate dynamic bias - CRITICAL CHANGE: keep the gradient connection
        dynamic_bias = self.current_bias_maps * (1 - velocity) - velocity * self.selu(conv_output + self.base)
        
        # Final output with gradient flow to base
        final_output = self.selu(conv_output + self.base + dynamic_bias)
        
        # Update the current bias maps for the next iteration
        self.current_bias_maps = dynamic_bias.detach().clone()
        
        if self.kernel_size == (8,8):
            print(f"\r{self._get_name()}: Velocity: {torch.norm(velocity).item():.4f}, Activation: {torch.norm(final_output).item():.4f}, Biases: {torch.norm(dynamic_bias).item():.4f}, Bases: {torch.norm(self.base).item()}", end="")
        
        return final_output

# Setup debug environment
def debug_gradients():
    # Create a small test input
    batch_size = 2
    in_channels = 3
    height, width = 32, 32
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    
    # Create the model
    model = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
    
    # Set model to training mode
    model.train()
    
    # Register hooks to track gradients
    base_grads = []
    def hook_fn(grad):
        base_grads.append(grad.clone().detach())
    
    # Forward pass
    output = model(x)
    
    # Print model state before backward
    print(f"Base parameter exists: {model.base is not None}")
    print(f"Base requires grad: {model.base.requires_grad}")
    print(f"Base shape: {model.base.shape}")
    print(f"Current bias maps shape: {model.current_bias_maps.shape}")
    
    # Register gradient hook for base
    if model.base is not None and model.base.requires_grad:
        model.base.register_hook(hook_fn)
    
    # Create a dummy loss and backward
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    print("\nAfter backward pass:")
    
    # Check all parameters for gradients
    print("\nGradients for all parameters:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - Gradient norm: {param.grad.norm().item()}")
        else:
            print(f"{name} - No gradient")
    
    # Specifically check base gradient
    if model.base.grad is not None:
        print(f"\nBase gradient norm: {model.base.grad.norm().item()}")
        print(f"Sample of base gradient: {model.base.grad.flatten()[:5]}")
    else:
        print("\nBase has no gradient!")
    
    # Check captured gradients from hook
    if base_grads:
        print(f"\nCaptured {len(base_grads)} gradient(s) from hook")
        for i, grad in enumerate(base_grads):
            print(f"Hook gradient {i} norm: {grad.norm().item()}")
    else:
        print("\nNo gradients captured by hook!")
    
    # Try direct gradient computation
    print("\nTrying direct gradient computation:")
    try:
        # Create a fresh model and input
        x_fresh = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        model_fresh = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # Forward pass to initialize parameters
        output_fresh = model_fresh(x_fresh)
        
        # We need base to be initialized
        if model_fresh.base is not None:
            # Try to directly compute gradient
            direct_grad = grad(outputs=output_fresh.sum(), 
                               inputs=model_fresh.base, 
                               create_graph=False, 
                               retain_graph=True)
            
            if direct_grad[0] is not None:
                print(f"Direct gradient computation result: {direct_grad[0].norm().item()}")
            else:
                print("Direct gradient computation returned None")
        else:
            print("Fresh model base is None")
    except Exception as e:
        print(f"Error in direct gradient computation: {e}")
    
    # Track computation graph and debug
    print("\nAnalyzing computational graph...")
    try:
        # Create another fresh instance
        x_debug = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        model_debug = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # Forward pass to initialize
        output_debug = model_debug(x_debug)
        
        # Check if output requires gradient
        print(f"Output requires grad: {output_debug.requires_grad}")
        
        # Check where the base is used
        # Modify forward to track this separately
        
        # Add a debugging hook to see data flow
        def debug_hook(name):
            def hook(grad):
                print(f"Gradient flow through {name}: {grad.norm().item()}")
            return hook
        
        x_debug.register_hook(debug_hook("input"))
        model_debug.conv.weight.register_hook(debug_hook("conv_weight"))
        
        if model_debug.base is not None:
            model_debug.base.register_hook(debug_hook("base"))
            
            # Try a simpler test
            print("\nSimple test with direct base usage:")
            # Create a simple computation directly using base
            simple_output = model_debug.conv(x_debug) + model_debug.base.expand_as(model_debug.conv(x_debug))
            simple_loss = simple_output.mean()
            simple_loss.backward(retain_graph=True)
            
            print(f"Simple test - Base grad exists: {model_debug.base.grad is not None}")
            if model_debug.base.grad is not None:
                print(f"Simple test - Base grad norm: {model_debug.base.grad.norm().item()}")
    except Exception as e:
        print(f"Error in graph analysis: {e}")
        
    # Detailed inspection of updated_bias calculation
    print("\nInspecting the updated_bias calculation:")
    try:
        # One more fresh instance
        x_inspect = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        model_inspect = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # Forward pass to initialize and to make modified computation
        # We'll run separately for visualization
        _ = model_inspect(x_inspect)
        
        # Now run each step with checks
        velocity = model_inspect.sigmoid(model_inspect.velocity_conv(x_inspect))
        conv_output = model_inspect.conv(x_inspect)
        z = conv_output + model_inspect.current_bias_maps
        activation = model_inspect.selu(z)
        
        # The key computation
        print("Computing updated_bias with detailed checks:")
        base_expanded = model_inspect.base.expand_as(activation)
        print(f"base_expanded requires_grad: {base_expanded.requires_grad}")
        
        diff = base_expanded - activation
        print(f"diff requires_grad: {diff.requires_grad}")
        
        weighted_diff = velocity * diff
        print(f"weighted_diff requires_grad: {weighted_diff.requires_grad}")
        
        updated_bias = weighted_diff + activation
        print(f"updated_bias requires_grad: {updated_bias.requires_grad}")
        
        # Now try to compute gradient directly to this point
        if updated_bias.requires_grad:
            direct_grad_ub = grad(outputs=updated_bias.sum(), 
                                inputs=model_inspect.base,
                                create_graph=False, 
                                retain_graph=True)
            
            if direct_grad_ub[0] is not None:
                print(f"Gradient to base from updated_bias: {direct_grad_ub[0].norm().item()}")
            else:
                print("No gradient to base from updated_bias")
    except Exception as e:
        print(f"Error in updated_bias inspection: {e}")

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()
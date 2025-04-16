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
        self.current_bias_maps -= velocity_expanded * activation
        
        return activation

# Setup debug environment
def debug_gradients():
    # Create a small test input
    batch_size = 2
    in_channels = 3
    height, width = 32, 32
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    
    # Create the model
    model = DynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
    
    # Set model to training mode
    model.train()
    
    # Register hooks to track gradients
    velocity_grads = []
    bias_grads = []
    
    def velocity_hook(grad):
        velocity_grads.append(grad.clone().detach())
    
    def bias_hook(grad):
        bias_grads.append(grad.clone().detach())
    
    # Forward pass to initialize parameters
    output = model(x)
    
    # Print model state before backward
    print(f"Velocity parameter exists: {model.velocity is not None}")
    print(f"Velocity requires grad: {model.velocity.requires_grad}")
    print(f"Velocity shape: {model.velocity.shape}")
    print(f"Current bias maps shape: {model.current_bias_maps.shape}")
    
    # Register gradient hooks
    if model.velocity is not None and model.velocity.requires_grad:
        model.velocity.register_hook(velocity_hook)
    
    # Need to retain graph since bias maps are updated in forward pass
    # Create a dummy loss and backward
    loss = output.mean()
    loss.backward(retain_graph=True)
    
    # Check gradients
    print("\nAfter backward pass:")
    
    # Check all parameters for gradients
    print("\nGradients for all parameters:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - Gradient norm: {param.grad.norm().item()}")
        else:
            print(f"{name} - No gradient")
    
    # Specifically check velocity gradient
    if model.velocity.grad is not None:
        print(f"\nVelocity gradient norm: {model.velocity.grad.norm().item()}")
        print(f"Sample of velocity gradient: {model.velocity.grad.flatten()[:5]}")
    else:
        print("\nVelocity has no gradient!")
    
    # Check captured gradients from hooks
    if velocity_grads:
        print(f"\nCaptured {len(velocity_grads)} velocity gradient(s) from hook")
        for i, grad in enumerate(velocity_grads):
            print(f"Velocity hook gradient {i} norm: {grad.norm().item()}")
    else:
        print("\nNo velocity gradients captured by hook!")
    
    # Track computation graph and debug
    print("\nAnalyzing computational graph...")
    try:
        # Create another fresh instance
        x_debug = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        model_debug = DynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # Forward pass to initialize
        output_debug = model_debug(x_debug)
        
        # Check if output requires gradient
        print(f"Output requires grad: {output_debug.requires_grad}")
        
        # Add debugging hooks to see data flow
        def debug_hook(name):
            def hook(grad):
                print(f"Gradient flow through {name}: {grad.norm().item()}")
            return hook
        
        x_debug.register_hook(debug_hook("input"))
        model_debug.conv.weight.register_hook(debug_hook("conv_weight"))
        
        if model_debug.velocity is not None:
            model_debug.velocity.register_hook(debug_hook("velocity"))
            
            # Try a simpler test
            print("\nSimple test with direct velocity usage:")
            # Create a simple computation directly using velocity
            conv_output = model_debug.conv(x_debug)
            z = conv_output + model_debug.current_bias_maps
            activation = model_debug.selu(z)
            velocity_expanded = model_debug.velocity.unsqueeze(0).expand_as(activation)
            simple_output = activation - velocity_expanded * activation
            simple_loss = simple_output.mean()
            simple_loss.backward(retain_graph=True)
            
            print(f"Simple test - Velocity grad exists: {model_debug.velocity.grad is not None}")
            if model_debug.velocity.grad is not None:
                print(f"Simple test - Velocity grad norm: {model_debug.velocity.grad.norm().item()}")
    except Exception as e:
        print(f"Error in graph analysis: {e}")

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()

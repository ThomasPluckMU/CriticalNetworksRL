import torch
import torch.nn as nn
import math

class DynamicBiasCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, velocity_init=0.1, bias_decay=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        self.velocity = nn.Parameter(torch.Tensor(out_channels, 1, 1).fill_(velocity_init))
        self.bias_decay = bias_decay
        self.selu = nn.SELU()
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        
    def _initialize_parameters(self, *args, **kwargs):
        pass
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Convolutional layer output
        a = self.conv(x) + self.bias
        
        # Apply dynamic bias correction
        # Broadcast velocity to match activation dimensions
        velocity_expanded = self.velocity.expand(batch_size, -1, a.size(2), a.size(3))
        
        # Apply the bias adjustment as in the NN implementation
        z = a - velocity_expanded * a
        activation = self.selu(z)
        print(activation.shape, velocity_expanded.shape)
        print(self.bias.shape)
        # Update bias in-place with no_grad to prevent tracking in backward pass
        with torch.no_grad():
            bias_update = velocity_expanded * activation
            self.bias.data = self.bias.data - bias_update
        
        return activation
    
    def reset_bias(self):
        """Reset the current bias state"""
        nn.init.zeros_(self.bias)

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
    
    # Register gradient hooks
    model.velocity.register_hook(velocity_hook)
    model.bias.register_hook(bias_hook)
    
    # Forward pass
    output = model(x)
    
    # Print model state before backward
    print(f"Velocity parameter exists: {model.velocity is not None}")
    print(f"Velocity requires grad: {model.velocity.requires_grad}")
    print(f"Velocity shape: {model.velocity.shape}")
    print(f"Conv bias shape: {model.bias.shape}")
    
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
    
    if bias_grads:
        print(f"\nCaptured {len(bias_grads)} bias gradient(s) from hook")
        for i, grad in enumerate(bias_grads):
            print(f"Bias hook gradient {i} norm: {grad.norm().item()}")
    else:
        print("\nNo bias gradients captured by hook!")
    
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
        model_debug.velocity.register_hook(debug_hook("velocity"))
        
        # Try a simpler test to isolate velocity gradient issues
        print("\nSimple test with direct velocity usage:")
        # Create a simple computation directly using velocity
        conv_output = model_debug.conv(x_debug)
        velocity_expanded = model_debug.velocity.expand(-1, conv_output.size(2), conv_output.size(3))
        z = conv_output - velocity_expanded * conv_output
        activation = model_debug.selu(z)
        simple_loss = activation.mean()
        simple_loss.backward()
        
        print(f"Simple test - Velocity grad exists: {model_debug.velocity.grad is not None}")
        if model_debug.velocity.grad is not None:
            print(f"Simple test - Velocity grad norm: {model_debug.velocity.grad.norm().item()}")
            
        # Test gradient flow through the in-place bias update
        print("\nTesting gradient through bias update:")
        model_test = DynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        x_test = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        
        # Store original bias
        original_bias = model_test.bias.clone()
        
        # Forward pass (which updates bias in-place)
        out_test = model_test(x_test)
        
        # Check if bias was updated
        bias_changed = not torch.allclose(original_bias, model_test.bias)
        print(f"Bias changed during forward: {bias_changed}")
        
        # Loss and backward
        test_loss = out_test.mean()
        test_loss.backward()
        
        # Check if gradients flow to velocity despite in-place bias update
        print(f"Velocity grad after in-place bias update: {model_test.velocity.grad is not None}")
        if model_test.velocity.grad is not None:
            print(f"Velocity grad norm: {model_test.velocity.grad.norm().item()}")
        
    except Exception as e:
        print(f"Error in graph analysis: {e}")

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()
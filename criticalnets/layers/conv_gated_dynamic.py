import torch
import torch.nn as nn
import math
from torch.autograd import grad

# Import the parent class (assuming it's in the same module)
from criticalnets.layers import DynamicBiasCNN

class GatedDynamicBiasCNN(DynamicBiasCNN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, velocity_init=0.1, bias_decay=1.0):
        """
        Initialize the Gated Dynamic Bias Convolutional Neural Network
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolutional kernel
            stride (int or tuple): Stride of the convolution (default: 1)
            padding (int or tuple): Padding added to all sides of the input (default: 0)
            velocity_init (float): Initial velocity value (default: 0.1)
            bias_decay (float): Bias decay factor (default: 1.0)
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, velocity_init, bias_decay)
        
        # Remove the constant velocity parameter from parent class
        self.register_parameter('velocity', None)
        
        # Add velocity computation convolution layer
        self.velocity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, bias=True)
        
        # Initialize velocity_conv weights
        nn.init.kaiming_uniform_(self.velocity_conv.weight, a=math.sqrt(5))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Convolutional layer output
        a = self.conv(x) + self.bias
        velocity = self.sigmoid(self.velocity_conv(x))
        z = a - velocity * a
        activation = self.selu(z)
        with torch.no_grad():
            bias_update = velocity * activation
            self.bias.data = self.bias.data - bias_update
        
        return activation

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
    velocity_conv_grads = []
    bias_grads = []
    
    def velocity_hook(grad):
        velocity_conv_grads.append(grad.clone().detach())
    
    def bias_hook(grad):
        bias_grads.append(grad.clone().detach())
    
    # Register gradient hooks
    model.velocity_conv.weight.register_hook(velocity_hook)
    model.bias.register_hook(bias_hook)
    
    # Forward pass
    output = model(x)
    
    # Print model state before backward
    print(f"Velocity conv exists: {model.velocity_conv is not None}")
    print(f"Velocity conv requires grad: {model.velocity_conv.weight.requires_grad}")
    print(f"Velocity conv shape: {model.velocity_conv.weight.shape}")
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
    
    # Specifically check velocity_conv gradient
    if model.velocity_conv.weight.grad is not None:
        print(f"\nVelocity conv gradient norm: {model.velocity_conv.weight.grad.norm().item()}")
        print(f"Sample of velocity conv gradient: {model.velocity_conv.weight.grad.flatten()[:5]}")
    else:
        print("\nVelocity conv has no gradient!")
    
    # Check captured gradients from hooks
    if velocity_conv_grads:
        print(f"\nCaptured {len(velocity_conv_grads)} velocity conv gradient(s) from hook")
        for i, grad in enumerate(velocity_conv_grads):
            print(f"Velocity conv hook gradient {i} norm: {grad.norm().item()}")
    else:
        print("\nNo velocity conv gradients captured by hook!")
    
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
        model_debug = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
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
        model_debug.velocity_conv.weight.register_hook(debug_hook("velocity_conv_weight"))
        
        # Try a simpler test with direct computation
        print("\nSimple test with direct velocity_conv usage:")
        
        # Create a simple computation directly using velocity_conv
        conv_output = model_debug.conv(x_debug)
        vel_output = model_debug.sigmoid(model_debug.velocity_conv(x_debug))
        z = conv_output - vel_output * conv_output
        activation = model_debug.selu(z)
        simple_loss = activation.mean()
        simple_loss.backward()
        
        print(f"Simple test - Velocity conv grad exists: {model_debug.velocity_conv.weight.grad is not None}")
        if model_debug.velocity_conv.weight.grad is not None:
            print(f"Simple test - Velocity conv grad norm: {model_debug.velocity_conv.weight.grad.norm().item()}")
    except Exception as e:
        print(f"Error in graph analysis: {e}")
    
    # Test the in-place bias update impact on gradients
    print("\nTesting the impact of in-place bias updates:")
    try:
        # Create a new model and input
        x_test = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        model_test = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # Store original bias
        original_bias = model_test.bias.clone()
        
        # Forward pass (which includes in-place bias update)
        output_test = model_test(x_test)
        
        # Verify bias was updated
        bias_changed = not torch.allclose(original_bias, model_test.bias)
        print(f"Bias was updated in-place: {bias_changed}")
        
        # Test gradient flow
        test_loss = output_test.mean()
        test_loss.backward()
        
        print(f"Gradient exists on velocity_conv despite in-place updates: {model_test.velocity_conv.weight.grad is not None}")
        if model_test.velocity_conv.weight.grad is not None:
            print(f"Velocity conv grad norm: {model_test.velocity_conv.weight.grad.norm().item()}")
    except Exception as e:
        print(f"Error in in-place bias update test: {e}")
    
    # Dissect the forward pass to examine velocity computation
    print("\nDissecting forward pass to examine velocity computation:")
    try:
        # Create new model and input
        x_dissect = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
        model_dissect = GatedDynamicBiasCNN(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # Compute each step separately
        conv_output = model_dissect.conv(x_dissect)
        print(f"Conv output shape: {conv_output.shape}, requires_grad: {conv_output.requires_grad}")
        
        velocity_output = model_dissect.velocity_conv(x_dissect)
        print(f"Raw velocity output shape: {velocity_output.shape}, requires_grad: {velocity_output.requires_grad}")
        
        velocity_activated = model_dissect.sigmoid(velocity_output)
        print(f"Activated velocity shape: {velocity_activated.shape}, requires_grad: {velocity_activated.requires_grad}")
        
        z = conv_output - velocity_activated * conv_output
        print(f"Z shape: {z.shape}, requires_grad: {z.requires_grad}")
        
        activation = model_dissect.selu(z)
        print(f"Activation shape: {activation.shape}, requires_grad: {activation.requires_grad}")
        
        # Compute the bias update (but don't apply it)
        bias_update = velocity_activated * activation.mean(dim=0, keepdim=True)
        print(f"Bias update shape: {bias_update.shape}, requires_grad: {bias_update.requires_grad}")
        
        # Verify gradient flow to velocity_conv
        dissect_loss = activation.mean()
        dissect_loss.backward()
        
        print(f"Velocity conv grad exists in dissection: {model_dissect.velocity_conv.weight.grad is not None}")
        if model_dissect.velocity_conv.weight.grad is not None:
            print(f"Velocity conv grad norm in dissection: {model_dissect.velocity_conv.weight.grad.norm().item()}")
    except Exception as e:
        print(f"Error in forward pass dissection: {e}")

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()
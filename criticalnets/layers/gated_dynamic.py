import torch
import torch.nn as nn
import math
from criticalnets.layers.dynamic_bias import DynamicBiasNN

class GatedDynamicBiasNN(DynamicBiasNN):
    def __init__(self, input_size, hidden_size, velocity_init=0.1, bias_decay=1.0):
        super().__init__(input_size, hidden_size, velocity_init, bias_decay)
        self.velocity_weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.gelu = nn.GELU()
        self.reset_parameters()
    
    def reset_parameters(self):

        super().reset_parameters()
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        a = torch.matmul(x, self.weight.t())
        gated_velocity = self.gelu(torch.matmul(x, self.velocity_weight.t())+self.velocity_bias)
        z = a + self.bias - gated_velocity * a
        activation = self.selu(z)
        
        with torch.no_grad():
            # Update bias with gated velocity
            bias_update = gated_velocity * activation
            self.bias.data = self.bias.data - bias_update
        
        return activation
        
# Setup debug environment
def debug_gradients():
    # Create a small test input
    batch_size = 2
    input_size = 10
    hidden_size = 8
    x = torch.randn(batch_size, input_size, requires_grad=True)
    
    # Create the model
    model = GatedDynamicBiasNN(input_size, hidden_size)
    
    # Set model to training mode
    model.train()
    
    # Register hooks to track gradients for all parameters
    param_grads = {}
    
    def create_hook(param_name):
        def hook_fn(grad):
            param_grads[param_name] = grad.clone().detach()
        return hook_fn
    
    # Register gradient hooks for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(create_hook(name))
    
    # Forward pass
    output = model(x)
    
    # Print model information before backward
    print("\n========== Model Structure ==========")
    print(f"Model type: {type(model).__name__}")
    print(f"Parent class: {model.__class__.__bases__[0].__name__}")
    
    # Print inherited attributes from DynamicBiasNN
    print("\n========== Inherited Attributes ==========")
    print(f"Input size: {model.input_size}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Weight shape: {model.weight.shape}")
    print(f"Velocity shape: {model.velocity.shape}")
    print(f"Bias shape: {model.bias.shape}")
    print(f"Bias decay: {model.bias_decay}")
    
    # Print GatedDynamicBiasNN specific attributes
    print("\n========== GatedDynamicBiasNN Specific Attributes ==========")
    print(f"Velocity weight shape: {model.velocity_weight.shape}")
    print(f"Velocity bias shape: {model.velocity_bias.shape}")
    
    # Create a dummy loss and backward
    loss = output.mean()
    loss.backward(retain_graph=True)
    
    # Check gradients for all parameters
    print("\n========== Gradients After Backward Pass ==========")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - Gradient norm: {param.grad.norm().item()}")
            # Print a few sample values
            print(f"  Sample values: {param.grad.flatten()[:3].tolist()}")
        else:
            print(f"{name} - No gradient")
    
    # Check hook-captured gradients to confirm they match
    print("\n========== Hook-Captured Gradients ==========")
    for name, grad in param_grads.items():
        print(f"{name} - Hook gradient norm: {grad.norm().item()}")
    
    # Test multiple forward passes to see bias updates
    print("\n========== Testing Bias Updates Over Multiple Forward Passes ==========")
    initial_bias = model.bias.clone().detach()
    initial_velocity = model.velocity.clone().detach()
    
    # Run several forward passes
    for i in range(5):
        new_x = torch.randn(batch_size, input_size)
        output = model(new_x)
        bias_change = (initial_bias - model.bias).norm().item()
        print(f"Pass {i+1}:")
        print(f"  Bias change norm: {bias_change}")
        print(f"  Current velocity values: {model.velocity[:3].tolist()}")
    
    # Test the reset_bias method
    print("\n========== Testing reset_bias Method ==========")
    current_bias = model.bias.clone().detach()
    model.reset_bias()
    reset_bias = model.bias.clone().detach()
    print(f"Bias before reset norm: {current_bias.norm().item()}")
    print(f"Bias after reset norm: {reset_bias.norm().item()}")
    print(f"Difference norm: {(current_bias - reset_bias).norm().item()}")
    
    # Test interaction between inherited velocity and new gated velocity
    print("\n========== Testing Interaction Between Inherited and New Components ==========")
    # Set specific test values
    test_x = torch.ones(batch_size, input_size)
    
    # Create a simple matrix multiplication trace for debugging
    with torch.no_grad():
        a = torch.matmul(test_x, model.weight.t())
        gated_vel = model.gelu(torch.matmul(test_x, model.velocity_weight.t()))
        inherited_effect = model.velocity * a
        gated_effect = gated_vel * a
        
        print(f"Base activation shape: {a.shape}")
        print(f"Inherited velocity effect norm: {inherited_effect.norm().item()}")
        print(f"Gated velocity effect norm: {gated_effect.norm().item()}")
        print(f"Are both mechanisms active? {'Yes' if inherited_effect.norm().item() > 0 and gated_effect.norm().item() > 0 else 'No'}")

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()
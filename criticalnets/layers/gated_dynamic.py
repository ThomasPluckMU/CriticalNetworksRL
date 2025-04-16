import torch
import torch.nn as nn
import math
from criticalnets.layers.dynamic_bias import DynamicBiasBase

class GatedDynamicBiasNN(DynamicBiasBase):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity_weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)/2)
        nn.init.kaiming_uniform_(self.velocity_weight, a=math.sqrt(5)/2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.current_bias is None:
            self.current_bias = self._init_bias(batch_size)
            self.current_bias += torch.randn_like(self.current_bias)
        
        velocity = self.gelu(torch.matmul(x, self.velocity_weight.t()))
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        activation = self.selu(z)
        
        # Update bias while maintaining gradient flow to velocity_weight
        bias_update = velocity * activation
        self.current_bias = (self.current_bias - bias_update).detach()
        # Add direct velocity_weight dependency to output
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
    
    # Register hooks to track gradients
    velocity_grads = []
    
    def velocity_hook(grad):
        velocity_grads.append(grad.clone().detach())
    
    # Forward pass to initialize parameters
    output = model(x)
    
    # Print model state before backward
    print(f"Velocity weight exists: {model.velocity_weight is not None}")
    print(f"Velocity weight requires grad: {model.velocity_weight.requires_grad}")
    print(f"Current bias shape: {model.current_bias.shape}")
    
    # Register gradient hooks
    if model.velocity_weight is not None and model.velocity_weight.requires_grad:
        model.velocity_weight.register_hook(velocity_hook)
    
    # Need to retain graph since bias is updated in forward pass
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
    if model.velocity_weight.grad is not None:
        print(f"\nVelocity weight gradient norm: {model.velocity_weight.grad.norm().item()}")
        print(f"Sample of velocity gradient: {model.velocity_weight.grad.flatten()[:5]}")
    else:
        print("\nVelocity weight has no gradient!")
    
    # Check captured gradients from hooks
    if velocity_grads:
        print(f"\nCaptured {len(velocity_grads)} velocity gradient(s) from hook")
        for i, grad in enumerate(velocity_grads):
            print(f"Velocity hook gradient {i} norm: {grad.norm().item()}")
    else:
        print("\nNo velocity gradients captured by hook!")

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()

    def reset_bias(self):
        """Reset current bias to initialized state with noise"""
        if hasattr(self, 'current_bias') and self.current_bias is not None:
            batch_size = self.current_bias.shape[0]
            self.current_bias = self._init_bias(batch_size)

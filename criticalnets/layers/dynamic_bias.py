import torch
import torch.nn as nn
import math

class DynamicBiasBase(nn.Module):
    """Base class for all dynamic bias layers"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.current_bias = None
        
    def reset_bias(self):
        """Reset the current bias state"""
        self.current_bias = None
        
    def _init_bias(self, batch_size):
        """Initialize bias tensor for a new sequence"""
        return torch.zeros(batch_size, self.hidden_size)
        
    def forward(self, x):
        """Base forward pass with dynamic bias update"""
        raise NotImplementedError("Subclasses must implement forward()")

class DynamicBiasNN(DynamicBiasBase):
    def __init__(self, input_size, hidden_size, velocity_init=0.1):
        super().__init__(input_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity = nn.Parameter(torch.Tensor(hidden_size).fill_(velocity_init))
        self.selu = nn.SELU()
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.current_bias is None:
            self.current_bias = self._init_bias(batch_size)
            
        a = torch.matmul(x, self.weight.t())
        z = a + self.current_bias
        activation = self.selu(z)
        # Store bias update while maintaining gradient flow to velocity
        bias_update = self.velocity * activation
        self.current_bias = (self.current_bias - bias_update).detach()
        
        return activation

# Setup debug environment
def debug_gradients():
    # Create a small test input
    batch_size = 2
    input_size = 10
    hidden_size = 8
    x = torch.randn(batch_size, input_size, requires_grad=True)
    
    # Create the model
    model = DynamicBiasNN(input_size, hidden_size)
    
    # Set model to training mode
    model.train()
    
    # Register hooks to track gradients
    velocity_grads = []
    
    def velocity_hook(grad):
        velocity_grads.append(grad.clone().detach())
    
    # Forward pass to initialize parameters
    output = model(x)
    
    # Print model state before backward
    print(f"Velocity parameter exists: {model.velocity is not None}")
    print(f"Velocity requires grad: {model.velocity.requires_grad}")
    print(f"Velocity shape: {model.velocity.shape}")
    print(f"Current bias shape: {model.current_bias.shape}")
    
    # Register gradient hooks
    if model.velocity is not None and model.velocity.requires_grad:
        model.velocity.register_hook(velocity_hook)
    
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

# Run the debug function        
if __name__ == "__main__":
    debug_gradients()

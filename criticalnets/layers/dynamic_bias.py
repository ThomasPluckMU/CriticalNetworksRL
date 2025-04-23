import torch
import torch.nn as nn
import math


class DynamicBiasBase(nn.Module):
    """Base class for all dynamic bias layers"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Use register_buffer instead of a regular attribute for the bias
        self.register_buffer("bias", None)

    def reset_bias(self):
        """Reset the current bias state"""
        self.bias = None

    def _init_bias(self, batch_size):
        """Initialize bias tensor for a new sequence"""
        return torch.zeros(
            batch_size, self.hidden_size, device=self.device_param.device
        )

    def forward(self, x):
        """Base forward pass with dynamic bias update"""
        raise NotImplementedError("Subclasses must implement forward()")


class DynamicBiasNN(DynamicBiasBase):
    def __init__(self, input_size, hidden_size, velocity_init=0.1, bias_decay=1.0):
        super().__init__(input_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.velocity = nn.Parameter(torch.Tensor(hidden_size).fill_(velocity_init))
        self.bias = nn.Parameter(
            torch.Tensor(hidden_size)
        )  # This is a nn.Parameter now, not a buffer
        self.selu = nn.SELU()
        self.bias_decay = bias_decay
        # Add a dummy parameter to get the device
        self.device_param = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # Initialize bias properly

    def forward(self, x):

        # Linear transformation
        a = torch.matmul(x, self.weight.t())
        z = a + self.bias - self.velocity * a
        activation = self.selu(z)

        # Update bias directly during forward pass - detach from computation graph
        with torch.no_grad():
            bias_update = self.velocity * activation
            self.bias.data = self.bias.data - bias_update

        return activation

    def reset_bias(self):
        """Reset the bias to its initial state"""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)


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
    weight_grads = []
    bias_grads = []  # Track bias gradients too

    def velocity_hook(grad):
        velocity_grads.append(grad.clone().detach())

    def weight_hook(grad):
        weight_grads.append(grad.clone().detach())

    def bias_hook(grad):
        bias_grads.append(grad.clone().detach())

    # Register hooks before forward pass
    model.velocity.register_hook(velocity_hook)
    model.weight.register_hook(weight_hook)
    model.bias.register_hook(bias_hook)  # Add hook for bias gradients

    # Store initial bias for comparison
    initial_bias = model.bias.clone()

    # Forward pass
    output = model(x)

    # Print model state before backward
    print("=== MODEL STATE BEFORE BACKWARD ===")
    print(f"Velocity parameter exists: {model.velocity is not None}")
    print(f"Velocity requires grad: {model.velocity.requires_grad}")
    print(f"Velocity shape: {model.velocity.shape}")
    print(f"Current bias shape: {model.bias.shape}")
    print(f"Current bias requires grad: {model.bias.requires_grad}")
    print(f"Weight requires grad: {model.weight.requires_grad}")
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")

    # Debug: Check bias changes from forward pass
    bias_change = (model.bias - initial_bias).abs().mean().item()
    print(f"\nBias change after forward pass: {bias_change:.6f}")

    # Create a dummy loss and backward
    loss = output.mean()
    print(f"\nLoss value: {loss.item()}")

    # Perform backward pass
    loss.backward(retain_graph=True)

    # Check gradients
    print("\n=== AFTER BACKWARD PASS ===")

    # Check all parameters for gradients
    print("\nGradients for all parameters:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - Gradient norm: {param.grad.norm().item():.6f}")
            print(f"{name} - Gradient mean: {param.grad.abs().mean().item():.6f}")
        else:
            print(f"{name} - No gradient")

    # Specifically check velocity gradient
    if model.velocity.grad is not None:
        print(f"\nVelocity gradient details:")
        print(f"Norm: {model.velocity.grad.norm().item():.6f}")
        print(f"Mean: {model.velocity.grad.abs().mean().item():.6f}")
        print(f"Min: {model.velocity.grad.min().item():.6f}")
        print(f"Max: {model.velocity.grad.max().item():.6f}")
        print(f"Sample values: {model.velocity.grad.flatten()[:5].tolist()}")
    else:
        print("\nVelocity has no gradient!")

    # Check bias gradient
    if model.bias.grad is not None:
        print(f"\nBias gradient details:")
        print(f"Norm: {model.bias.grad.norm().item():.6f}")
        print(f"Mean: {model.bias.grad.abs().mean().item():.6f}")
        print(f"Min: {model.bias.grad.min().item():.6f}")
        print(f"Max: {model.bias.grad.max().item():.6f}")
        print(f"Sample values: {model.bias.grad.flatten()[:5].tolist()}")
    else:
        print("\nBias has no gradient!")

    # Check captured gradients from hooks
    print(f"\nCaptured {len(velocity_grads)} velocity gradient(s) from hook")
    if velocity_grads:
        for i, grad in enumerate(velocity_grads):
            print(f"Velocity hook gradient {i} norm: {grad.norm().item():.6f}")

    print(f"\nCaptured {len(bias_grads)} bias gradient(s) from hook")
    if bias_grads:
        for i, grad in enumerate(bias_grads):
            print(f"Bias hook gradient {i} norm: {grad.norm().item():.6f}")

    # Test multiple forward passes to see bias updates
    print("\n=== TESTING MULTIPLE FORWARD PASSES ===")

    print("Initial bias values (sample):", model.bias.flatten()[:3].tolist())

    # Store bias values across multiple passes
    bias_values = [model.bias.clone()]

    # Perform multiple forward passes
    for i in range(3):
        output = model(x)
        bias_values.append(model.bias.clone())

        # Calculate change from previous bias
        bias_diff = (bias_values[-1] - bias_values[-2]).abs().mean().item()
        print(f"Pass {i+1}: Bias change: {bias_diff:.6f}")

    # Print sample bias values for comparison
    print("\nBias evolution (first 3 values):")
    for i, bias in enumerate(bias_values):
        print(f"Pass {i}: {bias.flatten()[:3].tolist()}")

    # Run a backward pass after multiple forwards
    loss = model(x).mean()
    try:
        loss.backward()
        print("\nBackward pass after multiple forwards completed successfully")

        # Check if bias gradient exists (it should since it's a parameter)
        if model.bias.grad is not None:
            print(
                f"Bias gradient norm after multiple passes: {model.bias.grad.norm().item():.6f}"
            )
        else:
            print("Bias has no gradient after multiple passes")

    except Exception as e:
        print(f"\nError in backward pass after multiple forwards: {e}")


# Run the debug function
if __name__ == "__main__":
    debug_gradients()

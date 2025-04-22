# Layers Module

## Available Layer Types
1. **Dynamic Bias Layers**:
   - `DynamicBiasNN`: Fully-connected layer with dynamic bias (SELU activation)
   - `DynamicBiasCNN`: Convolutional layer with dynamic bias (Tanh activation)
   - `GatedDynamicBiasNN`: Gated version of dynamic bias NN
   - `ConvDynamicBias`: Convolutional version with spatial bias updates

## Key Features
- Velocity-based bias updates during forward pass
- Automatic bias state management
- Configurable bias decay rates
- Different activation functions per layer type
- Reset capabilities between sequences

## Usage Examples

### DynamicBiasNN (Fully-connected)
```python
from criticalnets.layers import DynamicBiasNN

layer = DynamicBiasNN(
    input_size=64,
    hidden_size=128,
    velocity_init=0.1,  # Initial update rate
    bias_decay=0.9      # Bias retention rate
)

output = layer(input_tensor)  # Automatically updates bias
layer.reset_bias()            # Reset between sequences
```

### DynamicBiasCNN (Convolutional)
```python
from criticalnets.layers import DynamicBiasCNN

layer = DynamicBiasCNN(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    velocity_init=0.05,
    bias_decay=0.95
)

output = layer(image_batch)  # Updates spatial bias maps
```

## Implementation Details
All layers implement:
- `forward()`: Computes output and updates bias
- `reset_bias()`: Resets bias state
- Velocity parameter controls update magnitude
- Bias decay controls how much previous bias is retained

## Adding New Layers
1. Inherit from base class
2. Implement required methods:
```python
from .dynamic_bias import DynamicBiasBase
import torch.nn as nn

class CustomDynamicLayer(DynamicBiasBase):
    def __init__(self, input_size, hidden_size, custom_param=0.1):
        super().__init__(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        self.custom_param = nn.Parameter(torch.tensor(custom_param))
        
    def forward(self, x):
        # Your custom bias update logic here
        output = self.linear(x) + self.current_bias
        self.current_bias = ... # Update bias
        return output
```

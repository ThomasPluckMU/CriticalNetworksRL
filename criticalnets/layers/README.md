# Layers Module

## 1. What is this module?
This module provides dynamic bias neural network layers that maintain evolving bias states. Key features:
- Base class (`DynamicBiasBase`) for all dynamic bias layers
- Standard NN layers with dynamic bias (`DynamicBiasNN`, `GatedDynamicBiasNN`, `DeadWeightDynamicBiasNN`)
- CNN versions (`DynamicBiasCNN`, `GatedDynamicBiasCNN`, `DeadWeightDynamicBiasCNN`)
- Automatic bias state management and reset capabilities

## 2. How to use it
```python
from criticalnets.layers import DynamicBiasNN

# Create a dynamic bias layer
layer = DynamicBiasNN(input_size=64, hidden_size=128)

# Forward pass (automatically updates bias state)
output = layer(input_tensor)

# Reset bias state between sequences
layer.reset_bias()
```

## 3. How to add new layers
To create a new dynamic bias layer:
1. Inherit from `DynamicBiasBase`
2. Implement the `forward()` method
3. Optionally override `_init_bias()` for custom bias initialization
4. Add any additional parameters needed for your bias update logic

Example minimal layer:
```python
from .dynamic_bias import DynamicBiasBase
import torch.nn as nn

class MyDynamicLayer(DynamicBiasBase):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        if self.current_bias is None:
            self.current_bias = self._init_bias(x.size(0))
            
        output = self.linear(x) + self.current_bias
        self.current_bias = ... # Your update logic here
        return output

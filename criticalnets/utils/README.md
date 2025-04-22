# Utils Module

## Core Utilities

### ReplayMemory
```python
from criticalnets.utils import ReplayMemory

memory = ReplayMemory(capacity=10000)
memory.push(state, action, next_state, reward, done, game_id)
batch = memory.sample(32)  # Returns Transition tuples
```

### Frame Processing
```python
from criticalnets.utils import preprocess_frame

# Convert RGB to grayscale, crop, downsample
processed = preprocess_frame(raw_frame)  # (84, 84) float32 [0,1]
```

### Keyboard Controls
```python
from criticalnets.utils import KeyboardController

controller = KeyboardController(initial_delay=0.01)
controller.start()  # Runs in background thread
# Use +/- to adjust render speed
controller.stop()
```

## Numerical Helpers

### Criticality Regularization
Calculates Edge of Chaos regularization term:
```
R(layer) = (2σ′(z)∇²ₓσ(z)/√N) * (1/N - 1/||∇ₓσ(z)||)
```

Usage:
```python
from criticalnets.utils.numerical_helpers import (
    criticality_regularization,
    get_activation_derivatives
)

# For a conv layer:
reg_loss = criticality_regularization(
    model=conv_layer,
    x=input_tensor,
    activation_func=torch.tanh,
    layer_type='conv'
)
```

### Jacobian/Laplacian
```python
jacobian = compute_jacobian_approximation(model, x, activation_func)
laplacian = compute_laplacian_approximation(model, x, activation_func)
```

## Adding New Utilities
1. Add to appropriate file:
   - `atari_helpers.py`: Game-specific utilities
   - `numerical_helpers.py`: Math operations
2. Follow existing patterns:
```python
def new_utility(param: type) -> return_type:
    """Docstring explaining purpose and math"""
    # Implementation
    return result
```

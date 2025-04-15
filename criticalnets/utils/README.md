# Utils Module

## 1. What is this module?
This module provides utility functions and classes for Atari reinforcement learning. Key features:
- `ReplayMemory`: Experience replay buffer implementation
- `preprocess_frame()`: Frame preprocessing for Atari games
- `KeyboardController`: Interactive rendering speed control

## 2. How to use it
```python
from criticalnets.utils import ReplayMemory, preprocess_frame

# Create replay memory
memory = ReplayMemory(capacity=10000)

# Preprocess frames
processed_frame = preprocess_frame(raw_frame)

# Use keyboard controls (optional)
controller = KeyboardController()
controller.start()
```

## 3. How to add new utilities
To add new utility functions:
1. Create a new function or class in the appropriate file
2. Add any necessary imports
3. Document the functionality clearly

Example utility function:
```python
def normalize_rewards(rewards: List[float]) -> List[float]:
    """Normalize rewards to [-1, 1] range"""
    max_reward = max(abs(r) for r in rewards)
    return [r/max_reward for r in rewards]

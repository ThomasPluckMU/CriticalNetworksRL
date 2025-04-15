# Training Logic Module

## 1. What is this module?
This module provides swappable training logic implementations for reinforcement learning. Key features:
- Base abstract class (`TrainingLogic`) defining the training interface
- Concrete implementations for different training scenarios
- Pluggable architecture allowing easy switching of training strategies

Current implementations:
- `MultiGameLogic`: Handles training across multiple games with periodic switching
- (Additional implementations can be added)

## 2. How to use it
```python
from criticalnets.training.logic import MultiGameLogic

# Create training logic
training_logic = MultiGameLogic(switch_interval=10)

# Use with a trainer
trainer = MyTrainer(logic=training_logic)
trainer.run()
```

## 3. How to add new training logic
To create new training logic:
1. Inherit from `TrainingLogic` base class
2. Implement the required `run_episode()` method
3. Optionally override:
   - `configure_optimizer()` for custom optimizer setup
   - `on_checkpoint()` for checkpoint callbacks

Example minimal implementation:
```python
from .base import TrainingLogic

class MyTrainingLogic(TrainingLogic):
    def run_episode(self, env, agent, memory, episode_idx):
        # Implement your training logic
        total_reward = 0.0
        metrics = {}
        return total_reward, metrics

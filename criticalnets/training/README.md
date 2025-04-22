# Training Module

## Trainer Classes
- `BaseTrainer`: Core training infrastructure
  - Environment management
  - Checkpoint saving
  - Memory buffer
  - Keyboard controls
- `SingleGameTrainer`: Specialized for single-game training
  - Target network for stability
  - Progress tracking
  - Reward/loss logging
- `MultiGameTrainer`: Handles training across multiple games

## Training Logic Implementations
Available algorithms:
- `TD`: Temporal Difference learning
- `SARSA`: State-Action-Reward-State-Action
- `A2C_TD`: Advantage Actor-Critic with TD
- `PPO_TD`: Proximal Policy Optimization with TD
- `DirectReward`: Direct reward optimization
- `MultiGameLogic`: Multi-game training with switching

## Usage Example
```python
from criticalnets.training import SingleGameTrainer
from criticalnets.agents import CriticalAtariDQN
from criticalnets.training.logic import TD

# Configure training
config = {
    'memory_size': 100000,
    'render': False,
    'save_dir': 'checkpoints',
    'lr': 0.0001,
    'debug': True
}

# Create trainer with TD learning
trainer = SingleGameTrainer(
    config=config,
    logic=TD(),
    agent_cls=CriticalAtariDQN
)

# Train on Pong
trainer.train("ALE/Pong-v5", episodes=1000)
```

## Key Features
- **Checkpointing**: Automatic model saving
- **Logging**: Episode rewards and debug metrics
- **Memory**: Experience replay buffer
- **Visualization**: Real-time training progress
- **Customization**: Extensible trainer and logic
# Training Module

## 1. What is this module?
This module provides training infrastructure for Atari agents. Key components:
- `MultiGameTrainer`: Handles training across multiple games
- `SingleGameTrainer`: Specialized for single-game training
- Training logic implementations (see training/logic/README.md)

## 2. How to use it
```python
from criticalnets.training import MultiGameTrainer
from criticalnets.agents import GatedAtariUDQN
from criticalnets.training.logic import MultiGameLogic

# Create trainer
trainer = MultiGameTrainer(
    config={...},
    logic=MultiGameLogic(),
    agent_cls=GatedAtariUDQN
)

# Start training
trainer.train()
```

## 3. How to customize training
To customize training behavior:
1. Configure via the config dictionary (memory size, render settings, etc.)
2. Implement custom training logic (see training/logic/README.md)
3. Extend existing trainers or create new ones

Example config:
```python
config = {
    'memory_size': 100000,
    'render': True,
    'save_dir': 'custom_checkpoints'
}

# Configuration System

## Overview
The config handler provides:
- YAML configuration file loading
- Parameter validation against agent/logic schemas
- Grid search hyperparameter tuning
- Configuration saving

## Configuration File Format
The configuration file must contain:
- Global parameters (episodes, render, etc.)
- An 'agents' section defining agent/logic combinations

```yaml
episodes: 100
render: false

agents:
  - AgentClassName:
      agent_params:
        param1: [value1, value2]  # Grid search values
        param2: [valueA, valueB]
      logics:
        - LogicClassName:
            logic_params:
              param1: [value1, value2]
              param2: [valueA, valueB]
```

## Example Config
```yaml
episodes: 1000
render: false

agents:
  - CriticalAtariDQNAgent:
      agent_params:
        epsilon: [0.0, 0.05, 0.1]
      logics:
        - TDLogic:
            logic_params:
              lr: [0.001, 0.01]
              gamma: [0.9, 0.99]
              batch_size: [32, 64]
        - DiracRewardLogic:
            logic_params:
              rescale: [1, 10]
              lr: [0.001, 0.01]
```

## Usage
```python
from criticalnets.configs import ConfigHandler

# Load config
handler = ConfigHandler("path/to/config.yaml")

# Get all config combinations (for grid search)
for config in handler.generate_configs():
    # Run training with this config
    trainer = get_trainer(config['trainer'])
    trainer.train(config)
```

## Validation Rules
1. Must contain 'agents' section with at least one agent
2. Each agent must have at least one logic defined
3. Agent and logic classes must exist in their respective modules
4. Parameters are validated against their schemas

## Grid Search
- Any parameter with a list of values will generate combinations
- Combinations are generated for both agent_params and logic_params
- Use `generate_configs()` to iterate through all combinations

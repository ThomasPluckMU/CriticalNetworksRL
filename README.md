# Critical Networks RL Framework [![codecov](https://codecov.io/gh/CriticalNetworksRL/CriticalNetworksRL/branch/main/graph/badge.svg)](https://codecov.io/gh/CriticalNetworksRL/CriticalNetworksRL)

> **Disclaimer:** The initial format of this framework were generated using DeepSeek-v3 and DeepSeek-R1 in plan/generate/debug/test loop using the Cline Autonomous Coding Agent extension.

A reinforcement learning framework implementing Edge of Chaos theory for training dynamic agents on Atari games.

## Key Features

- **Criticality Regularization**: Maintains networss ks near phase transitions
- **Modular Architecture**: Swappable agents, training logic and environments
- **Comprehensive Monitoring**: Tracks activations, gradients and criticality metrics
- **Multi-Game Training**: Supports training acromultiple Atari games

## Theoretical Background

The framework implements Edge of Chaos theory by:
1. Calculating Jacobian norms for each layer
2. Applying regularization to maintain critical dynamics:
   \[\mathcal{R}_{layer} = \dfrac{2\sigma'(z)\nabla^2_x\sigma(z)}{\sqrt{N}} \left(\dfrac1N - \dfrac{1}{\|\nabla_x\sigma(z)\|}\right)\]
3. Optimizing networks to operate near phase transitions

## Installation

```bash
git clone https://github.com/CriticalNetworksRL/CriticalNetworksRL.git
cd CriticalNetworksRL
pip install -r requirements.txt
```

## Quick Start Example

Train a critical DQN agent on Pong:

```bash
python scripts/train_atari.py \
  --trainer SingleGameTrainer \
  --game "ALE/Pong-v5" \
  --agent CriticalAtariDQN \
  --logic TD \
  --episodes 1000 \
  --save-dir pong_checkpoints \
  --debug
```

## Complete Usage

### Config File Mode (Recommended)
```bash
python scripts/train_atari.py --config examples/config_example.yaml
```

### CLI Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--trainer` | Trainer class | Required |
| `--game` | Atari game name | Required |
| `--agent` | Agent class | Required |
| `--logic` | Training logic | Required |
| `--episodes` | Training episodes | 1000 |
| `--render` | Enable rendering | False |
| `--save-dir` | Checkpoint directory | model_checkpoints |
| `--log-dir` | Debug logs | logs |
| `--debug` | Enable metrics logging | False |

## Project Structure

### Core Modules
| Module | Description |
|--------|-------------|
| [Agents](criticalnets/agents/README.md) | RL agent implementations |
| [Layers](criticalnets/layers/README.md) | Custom neural network layers |
| [Training](criticalnets/training/README.md) | Training infrastructure |
| [Environments](criticalnets/environments/README.md) | Atari game management |
| [Utils](criticalnets/utils/README.md) | Helper functions |

### Key Components
- **Agents**: Implement `BaseAtariAgent` interface
- **Training Logic**: Algorithm implementations (TD, SARSA, etc)
- **Criticality Metrics**: Jacobian/Laplacian calculations

## Advanced Usage

### Monitoring Criticality
```python
from criticalnets.utils.numerical_helpers import criticality_regularization

# During training:
crit_loss = criticality_regularization(
    model=layer,
    x=inputs,
    activation_func=torch.tanh,
    layer_type='conv'
)
```

### Custom Agents
1. Inherit from `BaseAtariAgent`
2. Implement `forward()`
3. Optionally override:
   - `act()` for custom action selection
   - `get_param_schema()` for config validation

See [module documentation](criticalnets/agents/README.md) for details.

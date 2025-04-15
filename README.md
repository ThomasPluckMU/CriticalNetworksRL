# Critical Networks RL Framework [![codecov](https://codecov.io/gh/CriticalNetworksRL/CriticalNetworksRL/branch/main/graph/badge.svg)](https://codecov.io/gh/CriticalNetworksRL/CriticalNetworksRL)

A reinforcement learning framework for training dynamic agents on Atari games.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CriticalNetworksRL/CriticalNetworksRL.git
cd CriticalNetworksRL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main training script is `scripts/train_atari.py`. Basic commands:

### Multi-game training
```bash
python scripts/train_atari.py --multi \
  --agent GatedAtariUDQN \
  --logic criticalnets.training.logic.multi_game.MultiGameLogic \
  --episodes 5000 \
  --render
```

### Single-game training
```bash
python scripts/train_atari.py --single "ALE/Pong-v5" \
  --agent GatedAtariUDQN \
  --logic criticalnets.training.logic.base.TrainingLogic \
  --episodes 1000 \
  --save-dir my_checkpoints
```

Full options:
```
--multi              Train across multiple Atari games
--single <game>      Train on specific Atari game
--episodes <count>   Number of episodes [default: 1000]
--render             Enable rendering
--checkpoint <path>  Load existing checkpoint
--save-dir <path>    Model save directory [default: model_checkpoints]
--lr <rate>          Learning rate [default: 0.001]
--memory-size <size> Replay memory size [default: 100000]
--agent <name>       Agent class name [required]
--logic <path>       Training logic class path [required]
```

## Project Structure

### Core Modules

1. [Agents](criticalnets/agents/README.md) - RL agent implementations
2. [Layers](criticalnets/layers/README.md) - Custom neural network layers
3. [Training](criticalnets/training/README.md) - Training infrastructure
   - [Training Logic](criticalnets/training/logic/README.md) - Training strategies
4. [Environments](criticalnets/environments/README.md) - Atari game management
5. [Utils](criticalnets/utils/README.md) - Helper functions and classes

### Scripts

- `train_atari.py`: Main training script
- Other utility scripts in `scripts/` directory

## Getting Started

1. Choose an agent from the [agents module](criticalnets/agents/README.md)
2. Select a training strategy from [training logic](criticalnets/training/logic/README.md)
3. Run training with your preferred configuration

See individual module READMEs for detailed documentation on extending the framework.

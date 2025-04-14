# Dynamic Bias Atari RL Agent

A reinforcement learning agent that can train on multiple Atari games using a unified action space with dynamic bias layers.

## Features

- Unified action space for multiple Atari games  
- Dynamic bias layers for game-specific adaptation
- Two training modes:
- Optional no-target-model mode for direct policy learning
  - Multi-game training with automatic game switching
  - Single-game focused training
- Checkpoint saving and resuming
- Rendering controls during training

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- ALE-py
- Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Modes

1. **Multi-game training** (cycles through available games):
```bash
python dynamic_bias_atari.py --mode multi
```

2. **Single-game training**:
```bash
python dynamic_bias_atari.py --mode single --game "ALE/DemonAttack-v5"
```

### Checkpoint Options

- Resume training from checkpoint:
```bash
python dynamic_bias_atari.py --mode single --game "ALE/DemonAttack-v5" --checkpoint model_checkpoints/ALE_DemonAttack-v5_single_100.pt
```

- Force loading mismatched checkpoint:
```bash
python dynamic_bias_atari.py --mode single --game "ALE/DemonAttack-v5" --checkpoint old_checkpoint.pt --ignore_shape_mismatch
```

### Runtime Controls

During training:
- `+` Increase render delay (slower)
- `-` Decrease render delay (faster) 
- `0` Reset to default delay
- `f` Toggle super fast mode (no delay)

## File Structure

- `dynamic_bias_atari.py`: Main training script
- `dynamic_bias_layers.py`: Dynamic bias layer implementation
- `model_checkpoints/`: Saved model weights
- `requirements.txt`: Python dependencies

## Training Options

- `--episodes`: Set number of training episodes (default: 1000)
- `--checkpoint`: Path to checkpoint file to resume training  
- `--ignore_shape_mismatch`: Allow loading checkpoints with different network architecture
- `--layer-type`: Choose dynamic bias layer variant (gated/deadweight, default: gated)
- `--reward-mode`: Select reward calculation (instant/discounted, default: discounted)
- `--no-target-model`: Bypass target network and feed rewards directly to policy network

## Layer Variants
- `gated`: Uses gating mechanism for dynamic bias
- `deadweight`: Uses dead weight pruning approach

## Reward Modes  
- `instant`: Uses immediate rewards only (γ=0)
- `discounted`: Standard temporal difference (γ=0.99)

## Monitoring

Training progress is saved as PNG plots every 10 episodes (multi-game) or 100 episodes (single-game).

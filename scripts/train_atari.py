
#!/usr/bin/env python3
"""
Atari Training CLI Entry Point

Usage:
  train_atari.py (--multi | --single <game>) [options]

Options:
  --multi              Train across multiple Atari games
  --single <game>      Train on specific Atari game
  --episodes <count>   Number of episodes [default: 1000]
  --render             Enable rendering
  --checkpoint <path>  Load existing checkpoint
  --save-dir <path>    Model save directory [default: model_checkpoints]
  --lr <rate>          Learning rate [default: 0.001]
  --memory-size <size> Replay memory size [default: 100000]
  --agent <name>       Agent class name from agents directory [required]
  --logic <path>       Training logic class path [required]
"""
import argparse
import os
import sys
from pathlib import Path
import gymnasium as gym

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from criticalnets.environments.atari_manager import AtariManager
from criticalnets.agents import get_agent_class
from criticalnets.training.logic import get_logic_class
from criticalnets.training import get_trainer_class

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent on Atari games')
    parser.add_argument('--trainer', type=str,
                      help='Trainer class from training directory')
    parser.add_argument('--game', type=str,
                      help='Choose a game available in py-ALE')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                      help='Enable rendering')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint file')
    parser.add_argument('--save-dir', type=str, default='model_checkpoints',
                      help='Model save directory')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--memory-size', type=int, default=100000,
                      help='Replay memory size')
    parser.add_argument('--agent', type=str, required=True,
                      help='Agent class name from agents directory')
    parser.add_argument('--logic', type=str, required=True,
                      help='Training logic class name from training/logic/ directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory if needed
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare config
    config = {
        'render': args.render,
        'save_dir': args.save_dir,
        'lr': args.lr,
        'memory_size': args.memory_size
    }
    
    # Get agent class
    trainer_cls = get_trainer_class(args.trainer)
    agent_cls = get_agent_class(args.agent)
    logic_cls = get_logic_class(args.logic)
    
    env = gym.make(args.game)
    config['action_space'] = env.action_space.n
    env.close()
    
    trainer = trainer_cls(config, logic_cls, agent_cls, atari_manager=AtariManager())
    trainer.train(args.game, args.episodes)

if __name__ == "__main__":
    main()

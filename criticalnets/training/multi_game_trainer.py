import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Type, Tuple
from collections import defaultdict
from .logic.base import TrainingLogic
from ..agents import BaseAtariAgent

from ..environments.atari_manager import AtariManager
from ..utils.atari_helpers import (
    Transition,
    ReplayMemory,
    preprocess_frame,
    KeyboardController
)

class MultiGameTrainer:
    """Handles training across multiple Atari games"""
    
    def __init__(self, config: Dict, logic: TrainingLogic, agent_cls: Type[BaseAtariAgent], atari_manager=None):
        self.config = config
        self.logic = logic
        self.logic.trainer = self  # Give logic access to trainer
        self.atari_manager = atari_manager if atari_manager else AtariManager()
        self.max_actions = self.atari_manager.get_max_action_space()
        self.keyboard = KeyboardController()
        self.episode_count = 0
        
        # Initialize agent
        self.agent_cls = agent_cls
        self.agent = self.agent_cls(config, self.max_actions)
        self.policy_net = self.agent
        self.target_net = self.agent_cls(config, self.max_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = self.logic.configure_optimizer(self.policy_net)
        self.memory = ReplayMemory(config.get('memory_size', 100000))

    def train(self):
        """Main training loop"""
        if self.config.get('render', True):
            self.keyboard.start()
            
        print(f"Available working games: {self.atari_manager.working_games}")
        env = self._make_env(random.choice(self.atari_manager.working_games))
        
        try:
            while True:
                total_reward, metrics = self.logic.run_episode(
                    env, self.agent, self.memory, self.episode_count
                )
                self.episode_count += 1
                
                if self.episode_count % 100 == 0:
                    self.logic.on_checkpoint(self.episode_count)
                    self.save_checkpoint(metrics['game'], self.episode_count)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            env.close()
            self.keyboard.stop()

    def _make_env(self, game_name: str):
        """Create and initialize environment"""
        print(f"Original game_name: {game_name}")
        # Ensure we're using the full game name with version
        full_game = game_name if '-' in game_name else f"{game_name}-v4"
        print(f"Full game name: {full_game}")
        env = gym.make(
            full_game,
            render_mode='human' if self.config.get('render', True) else None,
            full_action_space=True
        )
        print(f"Successfully created env for {full_game}")
        self.atari_manager.set_current_game(full_game)
        return env

    def _switch_game(self, current_game: str, env) -> Tuple[str, gym.Env]:
        """Switch to a new random game"""
        env.close()
        new_game = random.choice([
            g for g in self.atari_manager.working_games 
            if g != current_game or len(self.atari_manager.working_games) == 1
        ])
        print(f"\nSwitching from {current_game} to {new_game}\n")
        return new_game, self._make_env(new_game)

    def save_checkpoint(self, game_name: str, episode: int):
        """Save model checkpoint"""
        save_dir = self.config.get('save_dir', 'model_checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(
            save_dir,
            f"{game_name.replace('/', '_')}_{episode}.pt"
        )
        torch.save({
            'state_dict': self.policy_net.state_dict(),
            'max_actions': self.max_actions,
            'games': self.atari_manager.working_games,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        print(f"Model checkpoint saved to {save_path}")

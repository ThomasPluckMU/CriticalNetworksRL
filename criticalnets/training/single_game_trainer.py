import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Type
import os
import gymnasium as gym

from ..agents.gated_atari_udqn import GatedAtariUDQN
from ..training.logic.base import TrainingLogic

from ..environments.atari_manager import AtariManager
from ..utils.atari_helpers import (
    Transition,
    ReplayMemory,
    preprocess_frame,
    KeyboardController
)

class SingleGameTrainer:
    """Handles training for a single Atari game"""
    
    def __init__(self, config: Dict, logic: TrainingLogic, agent_cls: Type[GatedAtariUDQN], atari_manager=None):
        self.config = config
        self.atari_manager = atari_manager if atari_manager else AtariManager()
        self.max_actions = self.atari_manager.get_max_action_space()
        self.keyboard = KeyboardController()
        
        # Initialize agent
        self.agent = agent_cls(config, self.max_actions)
        self.policy_net = self.agent
        self.target_net = agent_cls(config, self.max_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.logic = logic
        self.optimizer = self.logic.configure_optimizer(self.policy_net, lr = config.get('lr'))
        
        self.memory = ReplayMemory(config.get('memory_size', 100000))
        self.all_rewards = []


    def train(self, game_name: str, episodes: int):
        """Train on a single game"""
        if game_name not in self.atari_manager.working_games:
            raise ValueError(f"Game {game_name} not in working games list")
            
        if self.config.get('render', True):
            self.keyboard.start()
            
        env = self._make_env(game_name)
        
        try:
            for episode in range(1, episodes + 1):
                total_reward, metrics = self.logic.run_episode(
                    env, self.agent, self.memory, episode
                )
                self.all_rewards.append(total_reward)
                
                if episode % 100 == 0:
                    self.logic.on_checkpoint(episode)
                    self.save_checkpoint(game_name, episode)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            env.close()
            self.keyboard.stop()

    def _make_env(self, game_name: str):
        """Create and initialize environment"""
        env = gym.make(
            game_name,
            render_mode='human' if self.config.get('render', True) else None
        )
        return env

    def save_checkpoint(self, game_name: str, episode: int):
        """Save model checkpoint"""
        save_dir = self.config.get('save_dir', 'model_checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(
            save_dir,
            f"{game_name.replace('/', '_')}_single_{episode}.pt"
        )
        torch.save({
            'state_dict': self.policy_net.state_dict(),
            'max_actions': self.max_actions,
            'game': game_name,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        print(f"Model checkpoint saved to {save_path}")

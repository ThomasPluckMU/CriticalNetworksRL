import torch
from datetime import datetime
from typing import Dict, Type, Optional
from pathlib import Path
import os
import gymnasium as gym
from importlib import import_module

from ..agents import BaseAtariAgent
from ..training.logic import TrainingLogic

from ..environments.atari_manager import AtariManager
from ..utils.atari_helpers import ReplayMemory, KeyboardController


class BaseTrainer:
    """Handles training for a single Atari game"""

    def __init__(
        self,
        config: Dict,
        logic: TrainingLogic,
        agent_cls: BaseAtariAgent,
        atari_manager=None,
    ):

        self.config = config
        self.atari_manager = atari_manager if atari_manager else AtariManager()
        self.max_actions = self.atari_manager.get_max_action_space()
        self.keyboard = KeyboardController()

        # Initialize agent
        self.agent = agent_cls(config, self.max_actions)
        self.logic = logic(config)

        self.memory = ReplayMemory(config.get("memory_size", 100000))
        self.all_rewards = []

    def train(self, game_name: str, episodes: int):

        raise NotImplementedError("train function not implemented")

    def _make_env(self, game_name: str):
        """Create and initialize environment"""
        env = gym.make(
            game_name, render_mode="human" if self.config.get("render", True) else None
        )
        return env

    def save_checkpoint(self, game_name: str, episode: int):
        """Save model checkpoint"""
        save_dir = self.config.get("save_dir", "model_checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(
            save_dir, f"{game_name.replace('/', '_')}_{episode}.pt"
        )
        torch.save(
            {
                "state_dict": self.policy_net.state_dict(),
                "max_actions": self.max_actions,
                "game": game_name,
                "timestamp": datetime.now().isoformat(),
            },
            save_path,
        )
        print(f"Model checkpoint")


# Auto-discover and register agents
TRAINING_REGISTRY = {}

training_dir = Path(__file__).parent
for file in training_dir.glob("*.py"):
    if file.name.startswith("_"):
        continue
    module = import_module(f"criticalnets.training.{file.stem}")
    for name, cls in vars(module).items():
        if (
            isinstance(cls, type)
            and issubclass(cls, BaseTrainer)
            and cls != BaseTrainer
        ):
            TRAINING_REGISTRY[name] = cls


def get_trainer_class(name: str) -> Type[BaseTrainer]:
    """Get trainer class by name from registry"""
    if name not in TRAINING_REGISTRY:
        raise ValueError(
            f"Unknown trainer: {name}. Available trainer: {list(TRAINING_REGISTRY.keys())}"
        )
    return TRAINING_REGISTRY[name]

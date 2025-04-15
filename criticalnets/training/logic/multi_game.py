from .base import TrainingLogic
import random
from typing import Any, Dict, Tuple
from collections import defaultdict

class MultiGameLogic(TrainingLogic):
    """Training logic implementation for multi-game scenarios"""
    
    def __init__(self, switch_interval: int = 10):
        self.switch_interval = switch_interval
        self.game_metrics = defaultdict(list)
        self.trainer = None  # Will be set by trainer
        
    def run_episode(self, env, agent, memory, episode_idx) -> Tuple[float, Dict]:
        """Execute one training episode with game switching logic"""
        if episode_idx % self.switch_interval == 0:
            env = self._switch_game(env)
            self.current_game = self.trainer.atari_manager.get_current_game()
            
        # Core training logic would go here
        total_reward = 0.0
        metrics = {
            'game': self.current_game,
            'steps': 0,
            'loss': 0.0
        }
        
        # TODO: Implement actual episode logic
        return total_reward, metrics
        
    def _switch_game(self, env) -> Any:
        """Switch to a new random game"""
        new_game = random.choice(self.trainer.atari_manager.working_games)
        print(f"\nSwitching to {new_game}\n")
        env.close()
        return self.trainer._make_env(new_game)
        
    def on_checkpoint(self, episode: int):
        """Save game-specific metrics"""
        print(f"Checkpoint at episode {episode}")
        for game, metrics in self.game_metrics.items():
            print(f"{game}: {sum(metrics)/len(metrics):.2f} avg reward")

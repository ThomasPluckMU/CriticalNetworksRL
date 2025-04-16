from typing import Dict, Type
import numpy as np
from tqdm import tqdm

from ..agents import BaseAtariAgent
from ..training.logic import TrainingLogic

from . import BaseTrainer

class SingleGameTrainer(BaseTrainer):
    """Handles training for a single Atari game"""
    
    def __init__(self, config: Dict, logic: TrainingLogic, agent_cls: BaseAtariAgent, atari_manager=None):
        
        super().__init__(config, logic, agent_cls, atari_manager=atari_manager)
        
        # Initialize agent
        self.policy_net = self.agent
        self.target_net = agent_cls(config, self.max_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = self.logic.configure_optimizer(self.policy_net, lr=config.get('lr'))

    def train(self, game_name: str, episodes: int):
        """Train on a single game"""
        if game_name not in self.atari_manager.working_games:
            raise ValueError(f"Game {game_name} not in working games list")
            
        if self.config.get('render', True):
            self.keyboard.start()
            
        env = self._make_env(game_name)
        
        try:
            # Initialize tqdm progress bar
            progress_bar = tqdm(range(1, episodes + 1), desc=f"Training {game_name}", 
                               unit="episode", dynamic_ncols=True)
            
            for episode in progress_bar:
                total_reward, metrics = self.logic.run_episode(
                    env, self.agent, self.memory, episode
                )
                self.all_rewards.append(total_reward)
                
                # Update progress bar with the total reward from this episode
                progress_bar.set_postfix({
                    'reward': f"{total_reward:.2f}",
                    'avg_reward': f"{np.mean(self.all_rewards[-100:]):.2f}"
                })
                
                if episode % 100 == 0:
                    self.logic.on_checkpoint(episode)
                    self.save_checkpoint(game_name, episode)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            env.close()
            self.keyboard.stop()
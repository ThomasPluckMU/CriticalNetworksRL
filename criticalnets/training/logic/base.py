from abc import ABC, abstractmethod
import torch.optim as optim
from typing import Any, Tuple, Dict

class TrainingLogic(ABC):
    """Abstract base class for swappable training episode logic"""
    
    @abstractmethod
    def run_episode(self, 
                   env: Any, 
                   agent: Any, 
                   memory: Any, 
                   episode_idx: int) -> Tuple[float, Dict]:
        """Execute one training episode
        
        Args:
            env: Gym environment instance
            agent: RL agent instance
            memory: Replay memory buffer
            episode_idx: Current episode number
            
        Returns:
            Tuple of (total_reward, metrics_dict)
        """
        pass

    def configure_optimizer(self, network):
        """Default optimizer configuration"""
        params = list(network.parameters())
        if not params:
            return None
        return optim.SGD(params, lr=0.001)

    def on_checkpoint(self, episode: int):
        """Callback for checkpoint events"""
        pass

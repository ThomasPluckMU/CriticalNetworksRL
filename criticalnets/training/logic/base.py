from abc import ABC, abstractmethod
import torch.optim as optim
from typing import Any, Tuple, Dict, Optional

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

    def configure_optimizer(self, network) -> Optional[optim.Optimizer]:
        """Configure optimizer for the network
        
        Args:
            network: Neural network with parameters to optimize
            
        Returns:
            Configured optimizer or None if no parameters
        """
        params = list(network.parameters())
        if not params:
            return None
        
        self.optimizer.add_param_group(params)
        return self.optimizer
    
    def step_optimizer(self, loss):
        """Perform a single optimization step
        
        Args:
            loss: Loss tensor to backpropagate
        """
        if self.optimizer is None:
            return
            
        self.optimizer.zero_grad()
        loss.backward()
        
        # Print gradient norms for all parameters
        total_norm = 0
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        print(f" Total gradient norm: {total_norm}, Loss: {loss}")


    def on_checkpoint(self, episode: int):
        """Callback for checkpoint events"""
        pass
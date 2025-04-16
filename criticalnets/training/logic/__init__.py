# Expose the base interface
from importlib import import_module

from abc import ABC, abstractmethod
import torch.optim as optim
from pathlib import Path
from typing import Type, Any, Tuple, Dict, Optional

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
        raise NotImplementedError("Core episode logic missing!")

    def configure_optimizer(self, network) -> Optional[optim.Optimizer]:
        """Configure optimizer for the network
        
        Args:
            network: Neural network with parameters to optimize
            
        Returns:
            Configured optimizer or None if no parameters
        """
        params = list(network.parameters)
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
        # print(f" Total gradient norm: {total_norm}, Loss: {loss}")


    def on_checkpoint(self, episode: int):
        """Callback for checkpoint events"""
        pass

# Auto-discover and register agents
AGENT_REGISTRY = {}

agents_dir = Path(__file__).parent
for file in agents_dir.glob("*.py"):
    if file.name.startswith("_"):
        continue
    module = import_module(f"criticalnets.training.logic.{file.stem}")
    for name, cls in vars(module).items():
        if isinstance(cls, type) and issubclass(cls, TrainingLogic) and cls != TrainingLogic:
            AGENT_REGISTRY[name] = cls

def get_logic_class(name: str) -> Type[TrainingLogic]:
    """Get logic class by name from registry"""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown logic: {name}. Available logic: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name]
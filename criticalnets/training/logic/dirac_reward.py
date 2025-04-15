from .base import TrainingLogic
import random
from typing import Any, Dict, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim


class DiracRewardLogic(TrainingLogic):
    
    def run_episode(self, env, agent, memory, episode_idx) -> Tuple[float, Dict]:
        """Execute one training episode for single game"""
        total_reward = 0.0
        state, _ = env.reset()  # Unpack (observation, info) tuple
        done = False
        
        while not done:
            # Convert numpy array to PyTorch tensor and move to device
            state_tensor = torch.from_numpy(state).float().to(agent.device)
            action = agent.act(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push(state, action, next_state, reward, done, 'single')
            state = next_state
            total_reward += reward
            
            # Only compute loss and update if there's a reward
            if reward != 0:
                # Re-compute q_values for training (creating a new computation graph)
                q_values = agent.forward(state_tensor)
                normalized_q = q_values/torch.norm(q_values)
                target = reward*normalized_q.detach()  # Detach target to avoid double backprop
                loss = self.loss_fn(normalized_q, target)
                self.step_optimizer(loss)
    
        return total_reward, {
            'game': 'single',
            'steps': episode_idx,
            'reward': total_reward
        }
        
    def on_checkpoint(self, episode: int):
        """Callback for checkpoint events"""
        pass
    
    def configure_optimizer(self, network, **kwargs) -> Optional[optim.Optimizer]:
        """Configure optimizer for the network
        
        Args:
            network: Neural network with parameters to optimize
            
        Returns:
            Configured optimizer or None if no parameters
        """
        params = list(network.parameters())
        if not params:
            return None
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(params, **kwargs)
        return self.optimizer
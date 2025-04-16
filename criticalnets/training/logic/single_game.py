from . import TrainingLogic
from typing import Any, Dict, Tuple
import torch

class SingleGameLogic(TrainingLogic):
    """Concrete implementation for single-game training"""
    
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
            
        return total_reward, {
            'game': 'single',
            'steps': episode_idx,
            'reward': total_reward
        }
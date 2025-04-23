from . import TrainingLogic
import random
from typing import Any, Dict, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim


class DiracRewardLogic(TrainingLogic):

    def __init__(self, config):

        self.rescale = config.get("rescale", 1e1)
        self.learning_rate = config.get("lr", 1e-2)

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
            memory.push(state, action, next_state, reward, done, "single")
            state = next_state
            total_reward += reward
            # Re-compute q_values for training (creating a new computation graph)
            q_values = agent.forward(state_tensor)
            normalized_q = q_values / torch.norm(q_values)
            target = (
                2 * reward + 1
            ) * normalized_q.detach()  # Detach target to avoid double backprop
            reg_loss = agent.get_metrics().get("criticality_loss",0.0)
            loss = self.loss_fn(normalized_q, target) * self.rescale + reg_loss
            # print(loss)
            self.step_optimizer(loss)

        return total_reward, {
            "game": "single",
            "steps": episode_idx,
            "reward": total_reward,
            "loss": loss.detach().item(),
            "metrics": agent.get_metrics(),
        }

    def on_checkpoint(self, episode: int):
        """Callback for checkpoint events"""
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

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(params, lr=self.learning_rate)
        return self.optimizer


# class ExpRewardLogic(TrainingLogic):

#     def run_episode(self, env, agent, memory, episode_idx) -> Tuple[float, Dict]:
#         """Execute one training episode for single game"""
#         total_reward = 0.0
#         gamma = 0.99
#         state, _ = env.reset()  # Unpack (observation, info) tuple
#         done = False

#         while not done:
#             # Convert numpy array to PyTorch tensor and move to device
#             loss = 0
#             state_tensor = torch.from_numpy(state).float().to(agent.device)
#             action = agent.act(state_tensor)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             memory.push(state, action, next_state, reward, done, 'single')
#             state = next_state
#             total_reward += reward
#             q_values = agent.forward(state_tensor)
#             normalized_q = q_values/torch.norm(q_values)
#             target = reward*normalized_q.detach()  # Detach target to avoid double backprop
#             new_loss = self.loss_fn(normalized_q, target)
#             reg_loss = self.agent.get_metrics().get('criticality_loss')
#             if reg_loss is not None:
#                 new_loss += reg_loss
#             loss *= gamma
#             loss += (1-gamma)*new_loss
#             self.step_optimizer(loss)

#         return total_reward, {
#             'game': 'single',
#             'steps': episode_idx,
#             'reward': total_reward,
#             'loss': loss.detach(),
#             'metrics': agent.get_metrics()
#         }

#     def on_checkpoint(self, episode: int):
#         """Callback for checkpoint events"""
#         pass

#     def configure_optimizer(self, network, **kwargs) -> Optional[optim.Optimizer]:
#         """Configure optimizer for the network

#         Args:
#             network: Neural network with parameters to optimize

#         Returns:
#             Configured optimizer or None if no parameters
#         """
#         params = list(network.parameters())
#         if not params:
#             return None

#         self.loss_fn = nn.MSELoss()
#         self.optimizer = optim.Adam(params, **kwargs)
#         return self.optimizer

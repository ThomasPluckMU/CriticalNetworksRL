from . import TrainingLogic
import random
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TDLogic(TrainingLogic):
    """
    Standard Temporal Difference learning implementation
    adapted to work with your existing framework.
    """
    
    def __init__(self, learning_rate=0.01, gamma=0.99, batch_size=1):
        """
        Initialize the TD learning parameters.
        
        Args:
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            batch_size: Mini-batch size for experience replay
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = None
        self.optimizer = None
    
    def run_episode(self, env, agent, memory, episode_idx) -> Tuple[float, Dict]:
        """Execute one training episode using standard TD learning (Q-learning)"""
        total_reward = 0.0
        state, _ = env.reset()  # Unpack (observation, info) tuple
        done = False
        steps = 0
        
        while not done:
            steps += 1
            # Convert numpy array to PyTorch tensor and move to device
            state_tensor = torch.from_numpy(state).float().to(agent.device)
            
            # Select action using epsilon-greedy policy
            action = agent.act(state_tensor)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay memory
            memory.push(state, action, next_state, reward, done, 'single')
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Train the network if we have enough samples
            if len(memory) >= self.batch_size:
                loss = self._update_network(agent, memory)
        
        return total_reward, {
            'game': 'single',
            'steps': steps,
            'reward': total_reward,
            'loss': loss,
            'metrics': agent.get_metrics()
        }
    
    def _update_network(self, agent, memory):
        """
        Update the network using standard TD learning (Q-learning).
        This implements the core TD update: Q(s,a) ← Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        """
        # Sample a batch of transitions from memory
        transitions = memory.sample(self.batch_size)
        
        # Extract components from transitions - compatible with your ReplayMemory
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for t in transitions:
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            next_states.append(t.next_state)
            dones.append(t.done)
        
        # Create tensors for each element
        state_batch = torch.FloatTensor(np.array(states)).to(agent.device)
        action_batch = torch.LongTensor(np.array(actions)).to(agent.device)
        reward_batch = torch.FloatTensor(np.array(rewards)).to(agent.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(agent.device)
        done_batch = torch.FloatTensor(np.array(dones)).to(agent.device)
        
        # Compute current Q values: Q(s,a)
        current_q_values = agent.forward(state_batch)
        # Get the Q-values for the actions that were taken
        q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        
        # Compute next Q values (for TD target): max_a' Q(s',a')
        with torch.no_grad():
            # For Q-learning (off-policy), use the maximum Q-value for the next state
            next_q_values = agent.forward(next_state_batch).max(1)[0]
            # Set Q-value to 0 for terminal states
            next_q_values = next_q_values * (1 - done_batch)
            
            # Compute TD target: r + γ * max_a' Q(s', a')
            td_targets = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss
        loss = self.loss_fn(q_values, td_targets)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: clip gradients to prevent explosion
        for param in agent.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.detach()
    
    def step_optimizer(self, loss):
        """Helper method to step optimizer with a given loss"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
        
        # Use Mean Squared Error loss for TD learning
        self.loss_fn = nn.MSELoss()
        
        # Override kwargs with our learning rate if it wasn't explicitly provided
        if 'lr' not in kwargs:
            kwargs['lr'] = self.learning_rate
            
        # Adam optimizer often works better than SGD for deep Q-networks
        self.optimizer = optim.Adam(params, **kwargs)
        return self.optimizer
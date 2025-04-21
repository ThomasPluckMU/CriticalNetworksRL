from . import TrainingLogic
import random
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SARSALogic(TrainingLogic):
    """
    SARSA: State-Action-Reward-State-Action (on-policy TD learning)
    adapted to work with your existing framework.
    
    The key difference from Q-learning is that SARSA uses the actual next action
    taken rather than the maximum Q-value action for the TD target.
    """
    
    def __init__(self, learning_rate=0.01, gamma=0.99, batch_size=1):
        """
        Initialize the SARSA learning parameters.
        
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
        """Execute one training episode using SARSA TD learning"""
        total_reward = 0.0
        state, _ = env.reset()  # Unpack (observation, info) tuple
        
        # Convert initial state to tensor and select first action
        state_tensor = torch.from_numpy(state).float().to(agent.device)
        action = agent.act(state_tensor)
        
        done = False
        steps = 0
        
        while not done:
            steps += 1
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Select next action using the same policy (important for SARSA)
            next_state_tensor = torch.from_numpy(next_state).float().to(agent.device)
            next_action = agent.act(next_state_tensor)
            
            # Store transition in replay memory with both current and next actions
            # Since your memory may not have a slot for next_action, we'll handle it separately
            memory.push(state, action, next_state, reward, done, 'single')
            
            # Keep track of action pairs for SARSA updates
            if len(memory) >= self.batch_size:
                loss = self._update_network(agent, memory, next_state, next_action)
            # Move to next state and action
            state = next_state
            action = next_action
            total_reward += reward
        
        return total_reward, {
            'game': 'single',
            'steps': steps,
            'reward': total_reward,
            'loss': loss,
            'metrics': agent.get_metrics()
        }
    
    def _update_network(self, agent, memory, current_next_state=None, current_next_action=None):
        """
        Update the network using SARSA TD learning.
        This implements the core SARSA update: Q(s,a) ← Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        where a' is the action actually taken in state s' (not the max Q-value action)
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
        
        # For each next state, select the next action using the current policy
        # This is the key difference between SARSA and Q-learning
        next_action_batch = torch.LongTensor(np.array([
            agent.act(torch.FloatTensor(s).to(agent.device))
            for s in next_states
        ])).to(agent.device)
        
        # Compute current Q values: Q(s,a)
        current_q_values = agent.forward(state_batch)
        # Get the Q-values for the actions that were taken
        q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        
        # Compute next Q values (for SARSA target): Q(s',a')
        with torch.no_grad():
            # For SARSA (on-policy), use the Q-value of the actual next action
            next_q_values = agent.forward(next_state_batch)
            # Gather the Q-values for the next actions
            next_q_values = next_q_values.gather(1, next_action_batch.unsqueeze(1)).squeeze()
            # Set Q-value to 0 for terminal states
            next_q_values = next_q_values * (1 - done_batch)
            
            # Compute SARSA target: r + γ * Q(s', a')
            sarsa_targets = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss
        loss = self.loss_fn(q_values, sarsa_targets)
        reg_loss = agent.get_metrics().get('criticality_loss')
        if reg_loss is not None:
            loss += reg_loss
        
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


# TODO: Fix.

# You can also implement Expected SARSA, which uses expected values rather than samples
# class ExpectedSARSALogic(SARSALogic):
#     """
#     Expected SARSA: A variant of SARSA that uses the expected value of Q(s',a')
#     under the current policy, rather than a sample.
    
#     This often has lower variance than vanilla SARSA.
#     """
    
#     def _update_network(self, agent, memory, current_next_state=None, current_next_action=None):
#         """
#         Update the network using Expected SARSA TD learning.
#         This implements: Q(s,a) ← Q(s,a) + α[r + γ*E[Q(s',a')] - Q(s,a)]
#         where E[Q(s',a')] is the expected value of the next state under the policy
#         """
#         # Sample a batch of transitions from memory
#         transitions = memory.sample(self.batch_size)
        
#         # Extract components from transitions
#         states = []
#         actions = []
#         rewards = []
#         next_states = []
#         dones = []
        
#         for t in transitions:
#             states.append(t.state)
#             actions.append(t.action)
#             rewards.append(t.reward)
#             next_states.append(t.next_state)
#             dones.append(t.done)
        
#         # Create tensors for each element
#         state_batch = torch.FloatTensor(np.array(states)).to(agent.device)
#         action_batch = torch.LongTensor(np.array(actions)).to(agent.device)
#         reward_batch = torch.FloatTensor(np.array(rewards)).to(agent.device)
#         next_state_batch = torch.FloatTensor(np.array(next_states)).to(agent.device)
#         done_batch = torch.FloatTensor(np.array(dones)).to(agent.device)
        
#         # Compute current Q values
#         current_q_values = agent.forward(state_batch)
#         q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        
#         # Compute expected next Q values (for Expected SARSA target)
#         with torch.no_grad():
#             next_q_values = agent.forward(next_state_batch)
            
#             # Assuming the agent has a get_policy method that returns action probabilities
#             # If not available, we can estimate it from epsilon-greedy
#             expected_q_values = torch.zeros(self.batch_size).to(agent.device)
            
#             # For each next state
#             for i in range(self.batch_size):
#                 # Get all Q-values for this state
#                 state_q_values = next_q_values[i]
                
#                 # Simple epsilon-greedy policy calculation
#                 epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 0.1
#                 n_actions = state_q_values.size(0)
                
#                 # Probability of random action is epsilon/n_actions for each action
#                 # Probability of greedy action is (1-epsilon) + epsilon/n_actions
#                 probs = torch.ones(n_actions) * epsilon / n_actions
#                 best_action = state_q_values.argmax().item()
#                 probs[best_action] += (1 - epsilon)
                
#                 # Expected value is sum of q_values * probabilities
#                 expected_q_values[i] = (state_q_values * probs.to(agent.device)).sum()
            
#             # Set expected Q-value to 0 for terminal states
#             expected_q_values = expected_q_values * (1 - done_batch)
            
#             # Compute Expected SARSA target: r + γ * E[Q(s', a')]
#             expected_sarsa_targets = reward_batch + (self.gamma * expected_q_values)
        
#         # Compute loss and update
#         loss = self.loss_fn(q_values, expected_sarsa_targets)
#         reg_loss = agent.get_metrics().get('criticality_loss')
#         if reg_loss is not None:
#             loss += reg_loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         for param in agent.parameters():
#             if param.grad is not None:
#                 param.grad.data.clamp_(-1, 1)
#         self.optimizer.step()
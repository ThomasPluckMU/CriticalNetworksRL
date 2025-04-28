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

    def __init__(self, config):
        """
        Initialize the TD learning parameters.

        Args:
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            batch_size: Mini-batch size for experience replay
            reg_strength: Strength of the criticality regularization
        """
        self.learning_rate = config.get("lr", 0.01)
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.reg_strength = config.get("reg_strength", 0.01)
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

            # Execute action in environment (convert to int32 for envpool)
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy().astype('int32'))
            done = terminated or truncated

            # Store transition in replay memory
            memory.push(state, action, next_state, reward, done, "single")

            # Move to next state
            state = next_state
            total_reward += reward

            # Train the network if we have enough samples
            if len(memory) >= self.batch_size:
                loss = self._update_network(agent, memory)

        return total_reward, {
            "game": "single",
            "steps": steps,
            "reward": total_reward,
            "loss": loss.detach().item(),
            "metrics": agent.get_metrics(),
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
            # Convert action to CPU numpy array if it's a tensor
            action = t.action.cpu().numpy() if torch.is_tensor(t.action) else t.action
            actions.append(action)
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
        action_batch = action_batch.view(-1, 1)  # Ensure proper shape for gather
        q_values = current_q_values.gather(1, action_batch).squeeze(1)

        # Compute next Q values (for TD target): max_a' Q(s',a')
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)
            # For Q-learning (off-policy), use the maximum Q-value for the next state
            next_max = next_q.max(dim=1)[0]
            # Set Q-value to 0 for terminal states
            next_max = next_max * (1 - done_batch)
            # Compute TD target: r + γ * max_a' Q(s', a')
            td_targets = reward_batch + (self.gamma * next_max)
        # Compute loss
        loss = self.loss_fn(q_values, td_targets) * 1e3
        reg_loss = agent.get_metrics().get("criticality_loss", 0.0)
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

        # Use Mean Squared Error loss for TD learning
        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss()

        # Adam optimizer often works better than SGD for deep Q-networks
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        return self.optimizer

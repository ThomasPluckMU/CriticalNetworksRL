from . import TrainingLogic
import random
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SARSALogic(TrainingLogic):
    """
    Stabilized SARSA implementation (State-Action-Reward-State-Action)
    with proper target handling and improved stability.
    """

    def __init__(self, config):
        """
        Initialize the SARSA learning parameters.

        Args:
            config: Configuration dictionary containing parameters
        """
        self.learning_rate = config.get("lr", 1e-4)  # Reduced learning rate for stability
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        
        # New parameters for stability
        self.target_update_freq = config.get("target_update_freq", 10)  # Update target network every N episodes
        # self.gradient_clip = config.get("gradient_clip", 1.0)
        self.update_steps = config.get("update_steps", 4)  # Number of updates per episode
        self.min_memory_size = config.get("min_memory_size", 1000)  # Min buffer size before learning
        
        # Initialize loss and optimizer to None
        self.loss_fn = None
        self.optimizer = None
        self.target_network = None
        self.episode_counter = 0

    def configure_optimizer(self, network) -> Optional[optim.Optimizer]:
        """
        Configure optimizer and create target network for stability.

        Args:
            network: Neural network with parameters to optimize

        Returns:
            Configured optimizer or None if no parameters
        """
        params = list(network.parameters())
        if not params:
            return None

        # Use Huber loss instead of MSE for better robustness to outliers
        self.loss_fn = nn.MSELoss()
        
        # Create target network (a copy of the main network)
        self.target_network = type(network)(network.config, network.action_space).to(network.device)
        self.target_network.load_state_dict(network.state_dict())
        self.target_network.eval()  # Set to evaluation mode
        
        # Lower learning rate for stability
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        return self.optimizer

    def run_episode(self, env, agent, memory, episode_idx) -> Tuple[float, Dict]:
        """
        Execute one training episode using SARSA TD learning with stability improvements.
        """
        # Track metrics
        total_reward = 0.0
        episode_loss = 0.0
        update_count = 0
        
        # Reset environment and get initial state
        state, _ = env.reset()
        
        # Convert initial state to tensor and select first action
        state_tensor = torch.from_numpy(np.array([state])).float().to(agent.device)
        
        # Use epsilon-greedy policy for exploration during training
        with torch.no_grad():
            action = agent.act(state_tensor).cpu().numpy()[0]
        
        done = False
        steps = 0
        
        # Store state-action pairs for coherent SARSA updates
        trajectory = []
        
        # Execute episode
        while not done:
            steps += 1
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated
            
            # Select next action using current policy
            next_state_tensor = torch.from_numpy(np.array([next_state])).float().to(agent.device)
            with torch.no_grad():
                next_action = agent.act(next_state_tensor).cpu().numpy()[0]
            
            # Record both current action and next action (needed for SARSA)
            transition = (state, action, reward[0], next_state, next_action, done)
            trajectory.append(transition)
            
            # Store in replay memory 
            memory.push(state, action, next_state, reward[0], done, "single")
            
            # Update state and action for next step
            state = next_state
            action = next_action
            total_reward += reward[0]
        
        # Only update if we have enough samples
        if len(memory) >= self.min_memory_size:
            # Perform multiple updates per episode for better stability
            for _ in range(self.update_steps):
                loss = self._update_network(agent, memory, trajectory)
                if loss is not None:
                    episode_loss += loss
                    update_count += 1
        
        # Average loss if updates were performed
        avg_loss = episode_loss / max(update_count, 1)
        
        # Update target network periodically
        self.episode_counter += 1
        if self.episode_counter % self.target_update_freq == 0:
            self._update_target_network(agent)
        
        return total_reward, {
            "game": "single",
            "steps": steps,
            "reward": total_reward,
            "loss": avg_loss,
            "metrics": agent.get_metrics(),
        }

    def _update_network(self, agent, memory, trajectory):
        """
        Update the network using SARSA TD learning with target network.
        This ensures more stable learning compared to the original implementation.
        """
        if len(memory) < self.batch_size:
            return None
            
        # Sample a batch of transitions from memory
        transitions = memory.sample(self.batch_size)
        
        # Prepare batch arrays
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_next_actions = []
        batch_dones = []
        
        # Extract batch components
        for t in transitions:
            batch_states.append(t.state)
            batch_actions.append(t.action)
            batch_rewards.append(t.reward)
            batch_next_states.append(t.next_state)
            batch_dones.append(float(t.done))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch_states)).to(agent.device)
        action_batch = torch.LongTensor(np.array(batch_actions)).to(agent.device)
        reward_batch = torch.FloatTensor(np.array(batch_rewards)).to(agent.device)
        next_state_batch = torch.FloatTensor(np.array(batch_next_states)).to(agent.device)
        done_batch = torch.FloatTensor(np.array(batch_dones)).to(agent.device)
        
        # Get actions for next states using current policy (SARSA is on-policy)
        with torch.no_grad():
            # Force evaluation mode during action selection
            agent.eval()
            next_actions = agent.act(next_state_batch).cpu()
            agent.train()  # Return to training mode
            next_action_batch = next_actions.to(agent.device)
            
        # Compute current Q values
        q_values = agent.forward(state_batch)
        
        # Reshape actions for proper indexing
        action_batch = action_batch.view(-1, 1)
        current_q = q_values.gather(1, action_batch).squeeze(1)
        
        # Compute next Q values using TARGET network for stability
        with torch.no_grad():
            next_q_values = self.target_network.forward(next_state_batch)
            next_action_batch = next_action_batch.view(-1, 1)
            next_q = next_q_values.gather(1, next_action_batch).squeeze(1)
            
            # Set Q-value to 0 for terminal states
            next_q = next_q * (1 - done_batch)
            
            # Compute SARSA target: r + γ * Q(s', a')
            sarsa_targets = reward_batch + (self.gamma * next_q)
        
        # Compute loss
        loss = self.loss_fn(current_q, sarsa_targets)
        
        # Add regularization if available
        reg_loss = agent.get_metrics().get("criticality_loss", 0.0)
        if isinstance(reg_loss, torch.Tensor):
            loss += reg_loss
        else:
            loss += torch.tensor(reg_loss, device=agent.device)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients for stability
        # nn.utils.clip_grad_value_(agent.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        return loss.item()

    def _update_target_network(self, agent):
        """Update target network by copying parameters from online network"""
        self.target_network.load_state_dict(agent.state_dict())
        
    def on_checkpoint(self, episode: int):
        """Callback for checkpoint events"""
        pass
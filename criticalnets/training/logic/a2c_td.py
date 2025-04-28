from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from . import TrainingLogic


class A2CLogic(TrainingLogic):
    """
    Memory-optimized Synchronous Advantage Actor-Critic logic:
      - Uses fixed-length segments instead of full episodes
      - Processes data in mini-batches
      - More aggressive tensor detaching
      - More efficient return/advantage calculation
      - Truncated backpropagation through time
    """

    def __init__(self, config):
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 7e-4)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # New parameters for memory optimization
        self.segment_length = config.get("segment_length", 128)  # Fixed segment length
        self.batch_size = config.get("batch_size", 32)  # Mini-batch size for updates
        self.update_epochs = config.get("update_epochs", 1)  # Number of epochs per segment

    def configure_optimizer(self, network: Any) -> Adam:
        self.optimizer = Adam(network.parameters(), lr=self.lr)
        return self.optimizer

    def run_episode(self, env, agent, memory, episode_idx: int) -> Tuple[float, Dict]:
        # Statistics tracking
        total_reward = 0.0
        total_steps = 0
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0
        
        # Initial observation
        obs, _ = env.reset()
        done = False

        # For memory efficient processing
        while not done:
            # Process in fixed-length segments
            segment_rewards = 0.0
            segment_steps = 0
            
            # Pre-allocate fixed-size buffers for the segment
            obs_buf = np.zeros((self.segment_length,) + obs.shape, dtype=np.float32)
            act_buf = np.zeros((self.segment_length,), dtype=np.int64)
            rew_buf = np.zeros(self.segment_length, dtype=np.float32)
            val_buf = np.zeros(self.segment_length, dtype=np.float32)
            logp_buf = np.zeros(self.segment_length, dtype=np.float32)
            done_buf = np.zeros(self.segment_length, dtype=bool)
            
            # Collect segment data
            for step in range(self.segment_length):
                # Convert single observation to tensor
                state_tensor = torch.from_numpy(np.expand_dims(obs, 0)).float().to(agent.device)
                
                # Get action and value
                with torch.no_grad():  # No need to track gradients during data collection
                    logits, value = agent.forward(state_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                
                # Execute action
                action_np = action.cpu().numpy()[0]
                next_obs, reward, terminated, truncated, _ = env.step(np.array([action_np]))
                current_done = terminated or truncated
                
                # Store transition
                obs_buf[step] = obs
                act_buf[step] = action_np
                rew_buf[step] = reward[0]  # Assuming scalar reward
                val_buf[step] = value.cpu().numpy()[0]
                logp_buf[step] = logp.cpu().numpy()[0]
                done_buf[step] = current_done
                
                # Update episode stats
                obs = next_obs
                segment_rewards += reward[0]
                segment_steps += 1
                total_steps += 1
                
                # Break if episode ends
                if current_done:
                    done = True
                    break
            
            # If we collected any steps, process the segment
            if segment_steps > 0:
                # Compute returns and advantages more efficiently
                returns, advantages = self._compute_returns_advantages(
                    agent, 
                    obs,
                    rew_buf[:segment_steps], 
                    val_buf[:segment_steps], 
                    done_buf[:segment_steps]
                )
                
                # Process collected segment data in mini-batches
                segment_loss, policy_loss, value_loss, entropy = self._process_segment(
                    agent,
                    obs_buf[:segment_steps],
                    act_buf[:segment_steps],
                    returns,
                    advantages,
                    logp_buf[:segment_steps]
                )
                
                # Update statistics
                total_reward += segment_rewards
                total_loss += segment_loss
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                update_count += 1
        
        # Calculate average losses
        avg_loss = total_loss / max(update_count, 1)
        avg_policy_loss = total_policy_loss / max(update_count, 1)
        avg_value_loss = total_value_loss / max(update_count, 1)
        avg_entropy = total_entropy / max(update_count, 1)
        
        return total_reward, {
            "game": "single",
            "steps": total_steps,
            "reward": total_reward,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "metrics": agent.get_metrics(),
            "loss": avg_loss,
        }
    
    def _compute_returns_advantages(self, agent, next_obs, rewards, values, dones):
        """
        Efficiently compute returns and advantages using vectorized operations
        """
        # Get bootstrap value for incomplete episode
        if dones[-1]:
            last_value = 0.0
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.expand_dims(next_obs, 0)).float().to(agent.device)
                _, last_value = agent.forward(state_tensor)
                last_value = last_value.cpu().numpy()[0]
        
        # Pre-allocate arrays for returns and advantages
        segment_size = len(rewards)
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # Efficient vectorized calculation (avoiding repetitive list insertions)
        next_return = last_value
        next_advantage = 0
        
        for i in reversed(range(segment_size)):
            # Compute return (more efficient than inserting at beginning of list)
            returns[i] = next_return = rewards[i] + self.gamma * next_return * (1 - dones[i])
            
            # Compute advantage
            advantages[i] = next_advantage = returns[i] - values[i]
        
        # Convert to tensors (once, at the end)
        returns_tensor = torch.from_numpy(returns).float().to(agent.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(agent.device)
        
        return returns_tensor, advantages_tensor
    
    def _process_segment(self, agent, observations, actions, returns, advantages, old_logps):
        """
        Process a segment of data in mini-batches
        """
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        batch_count = 0
        
        segment_size = len(observations)
        
        # Convert segment data to tensors
        obs_tensor = torch.from_numpy(observations).float().to(agent.device)
        act_tensor = torch.from_numpy(actions).long().to(agent.device)
        
        # Multiple epochs can help with convergence while keeping memory usage low
        for _ in range(self.update_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(segment_size)
            
            # Process in mini-batches
            for start_idx in range(0, segment_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, segment_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = obs_tensor[batch_indices]
                batch_acts = act_tensor[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass and compute losses
                values, logps, entropy = agent.evaluate_actions(batch_obs, batch_acts)
                
                # Clear gradients before each mini-batch
                self.optimizer.zero_grad()
                
                # Compute losses
                policy_loss = -(logps * batch_advantages.detach()).mean()
                value_loss = (batch_returns - values).pow(2).mean()
                ent_loss = entropy.mean()
                reg_loss = agent.get_metrics().get("criticality_loss", 0.0)
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * ent_loss
                    + reg_loss
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update statistics (detach to prevent memory leaks)
                total_loss += loss.detach().item()
                total_policy_loss += policy_loss.detach().item()
                total_value_loss += value_loss.detach().item()
                total_entropy += ent_loss.detach().item()
                batch_count += 1
                
                # Clean up tensors to free memory
                del batch_obs, batch_acts, batch_returns, batch_advantages
                del values, logps, entropy, policy_loss, value_loss, ent_loss, loss
        
        # Calculate average losses
        avg_loss = total_loss / max(batch_count, 1)
        avg_policy_loss = total_policy_loss / max(batch_count, 1)
        avg_value_loss = total_value_loss / max(batch_count, 1)
        avg_entropy = total_entropy / max(batch_count, 1)
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy
from typing import Any, Dict, Tuple, List
import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from . import TrainingLogic


class A2CLogic(TrainingLogic):
    """
    Standard Synchronous Advantage Actor-Critic logic:
      - Runs multiple environments in parallel
      - Collects fixed-length segments for n-step returns
      - Processes batched data from all environments
      - Computes proper n-step advantages
    """

    def __init__(self, config):
        self.gamma = config.get("gamma", 0.9)
        self.lr = config.get("lr", 7e-4)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.1)
        self.n_steps = config.get("n_steps", 32)  # Number of steps for n-step returns
        self.num_envs = config.get("num_envs", 16)  # Number of parallel environments
        self.game = config.get("game")

    def configure_optimizer(self, network: Any) -> Adam:
        self.optimizer = Adam(network.parameters(), lr=self.lr)
        return self.optimizer

    def run_episode(self, envs, agent, memory, update_idx: int) -> Tuple[float, Dict]:
        # Statistics tracking
        total_reward = 0.0
        total_steps = 0
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Storage for trajectory data
        states = []
        actions = []
        rewards = []
        values = []
        logps = []
        dones = []
        
        # Initial observation - now handling multiple environments
        obs, _ = envs.reset()  # Expect batch of observations with shape [num_envs, *obs_shape]
        done, step_count = False, 0
        # Collect n_steps of experience from all environments
        while not done:
                        
            state_tensor = torch.from_numpy(obs).float().to(agent.device)
            
            # Get actions and values for all environments
            with torch.no_grad():
                logits, value = agent.forward(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                
            # Execute actions in all environments
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = envs.step(action_np)
            done_status = np.logical_or(terminated, truncated)
            done = done_status.all()
            
            # Store trajectory data
            states.append(obs)
            actions.append(action_np)
            rewards.append(reward)
            values.append(value.cpu().numpy())
            logps.append(logp.cpu().numpy())
            dones.append(done_status)
            
            # Update stats
            total_reward += reward.mean()
            total_steps += self.num_envs
            
            # Move to next state
            obs = next_obs
            
            if step_count % self.n_steps == self.n_steps - 1:
        
                # Compute returns and advantages for the collected trajectories
                returns, advantages = self._compute_returns_advantages(
                    agent, 
                    states,
                    actions,
                    rewards,
                    values,
                    dones,
                    next_obs
                )
                
                # Process collected data
                loss, policy_loss, value_loss, entropy = self._process_batch(
                    agent,
                    states,
                    actions,
                    returns,
                    advantages,
                    logps
                )
                
                states = []
                actions = []
                rewards = []
                values = []
                logps = []
                dones = []
                
                # Update statistics
                total_loss = loss
                total_policy_loss = policy_loss
                total_value_loss = value_loss
                total_entropy = entropy
                
                step_count = 0
                
            else:
                
                step_count += 1
                
        return total_reward, {
            "game": f"{self.game}",
            "steps": total_steps,
            "reward": total_reward,
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "entropy": total_entropy,
            "metrics": agent.get_metrics(),
            "loss": total_loss,
        }
    
    def _compute_returns_advantages(self, agent, states, actions, rewards, values, dones, last_obs):
        """
        Compute n-step returns and advantages using proper bootstrapping
        """
        # Get bootstrap value from the final state
        with torch.no_grad():
            state_tensor = torch.from_numpy(last_obs).float().to(agent.device)
            _, last_value = agent.forward(state_tensor)
            last_value = last_value.cpu().numpy()
        
        # Convert lists to numpy arrays for vectorized operations
        rewards_arr = np.array(rewards)  # [n_steps, num_envs]
        values_arr = np.array(values)    # [n_steps, num_envs]
        dones_arr = np.array(dones)      # [n_steps, num_envs]
        
        # Pre-allocate arrays for returns and advantages
        returns = np.zeros_like(rewards_arr)
        advantages = np.zeros_like(rewards_arr)
        
        # Compute returns and advantages with n-step bootstrapping
        next_return = last_value.squeeze() if type(last_value) is not float else last_value
        for t in reversed(range(self.n_steps)):
            returns[t] = rewards_arr[t] + self.gamma * next_return * (1 - dones_arr[t])
            temp_values = values_arr[t].squeeze() if type(last_value) is not float else values_arr[t]
            advantages[t] = returns[t] - temp_values
            next_return = returns[t]
        
        # Flatten and convert to tensors
        returns_tensor = torch.from_numpy(returns.reshape(-1)).float().to(agent.device)
        advantages_tensor = torch.from_numpy(advantages.reshape(-1)).float().to(agent.device)
        
        return returns_tensor, advantages_tensor
    
    def _process_batch(self, agent, states, actions, returns, advantages, old_logps):
        """
        Process batched data from all environments
        """
        # Prepare batched data
        batch_states = np.array(states).reshape(-1, *states[0].shape[1:])  # [n_steps*num_envs, *obs_shape]
        batch_actions = np.array(actions).reshape(-1)                      # [n_steps*num_envs]
        # Convert to tensors
        state_tensor = torch.from_numpy(batch_states).float().to(agent.device)
        action_tensor = torch.from_numpy(batch_actions).to(agent.device)
        
        # Forward pass and compute losses
        values, logps, entropy = agent.evaluate_actions(state_tensor, action_tensor)
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Compute losses
        policy_loss = -(logps * advantages.detach()).mean()
        value_loss = (returns - values).pow(2).mean()
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
        
        return loss.item(), policy_loss.item(), value_loss.item(), ent_loss.item()
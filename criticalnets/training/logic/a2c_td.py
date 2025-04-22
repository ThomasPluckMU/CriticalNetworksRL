from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from . import TrainingLogic


class A2CLogic(TrainingLogic):
    """
    Synchronous Advantage Actor-Critic logic:
      - Play until the end of each episode
      - Compute discounted returns and advantages
      - Single optimization step per full episode
    """

    def __init__(self, config):
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 7e-4)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)

    def configure_optimizer(self, network: Any) -> Adam:
        self.optimizer = Adam(network.parameters(), lr=self.lr)
        return self.optimizer

    def run_episode(self, env, agent, memory, episode_idx: int) -> Tuple[float, Dict]:
        # Buffers for trajectory
        obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []

        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # Play until game terminates
        while not done:
            state_tensor = torch.from_numpy(obs).float().to(agent.device)
            logits, value = agent.forward(state_tensor.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action).to(agent.device))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Record step
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(value.item())
            rew_buf.append(reward)

            obs = next_obs
            total_reward += reward

        # No bootstrap beyond terminal
        last_val = 0.0

        # Compute returns and advantages
        ret_buf, adv_buf = [], []
        R = last_val
        for r, v in zip(reversed(rew_buf), reversed(val_buf)):
            R = r + self.gamma * R
            adv = R - v
            ret_buf.insert(0, R)
            adv_buf.insert(0, adv)

        # Convert to tensors
        obs_array = np.stack(obs_buf)
        obs_tensor = torch.from_numpy(obs_array).float().to(agent.device)
        act_tensor = torch.LongTensor(act_buf).to(agent.device)
        old_logp = torch.stack(logp_buf).detach()
        ret_tensor = torch.FloatTensor(ret_buf).to(agent.device)
        adv_tensor = torch.FloatTensor(adv_buf).to(agent.device)

        # Single update
        values, logp, entropy = agent.evaluate_actions(obs_tensor, act_tensor)
        policy_loss = -(logp * adv_tensor.detach()).mean()
        value_loss = (ret_tensor - values).pow(2).mean()
        ent_loss = entropy.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return total_reward, {
            "game": "single",
            "steps": len(obs_buf),
            "reward": total_reward,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": ent_loss.item(),
            "metrics": agent.get_metrics(),
            "loss": policy_loss.item() + value_loss.item(),
        }

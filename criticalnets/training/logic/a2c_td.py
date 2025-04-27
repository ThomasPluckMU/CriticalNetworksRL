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
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # Play until game terminates
        while not done:
            state_tensor = torch.from_numpy(obs).float().to(agent.device)
            logits, value = agent.forward(state_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logps = dist.log_prob(actions)

            next_obs, reward, terminated, truncated, _ = env.step(actions.cpu().numpy())
            done = terminated or truncated

            # Record step
            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logps)
            val_buf.append(value)
            rew_buf.append(reward)
            done_buf.append(done)

            obs = next_obs
            total_reward += reward.mean().item()

        # No bootstrap beyond terminal
        last_val = torch.zeros_like(val_buf[-1])

        # Compute returns and advantages
        ret_buf, adv_buf = [], []
        R = last_val
        for r, v in zip(reversed(rew_buf), reversed(val_buf)):
            R = torch.tensor(r).to(agent.device) + self.gamma * R * (~torch.tensor(done_buf[-1]).to(agent.device)).float()
            adv = R - v
            ret_buf.insert(0, R)
            adv_buf.insert(0, adv)

        # Convert to tensors
        obs_tensor = torch.from_numpy(np.stack(obs_buf)).float().to(agent.device)
        act_tensor = torch.stack(act_buf).to(agent.device)
        old_logp = torch.stack(logp_buf).detach()
        ret_tensor = torch.stack(ret_buf).to(agent.device)
        adv_tensor = torch.stack(adv_buf).to(agent.device)

        # Single update
        values, logp, entropy = agent.evaluate_actions(obs_tensor, act_tensor)
        policy_loss = -(logp * adv_tensor.detach()).mean()
        value_loss = (ret_tensor - values).pow(2).mean()
        ent_loss = entropy.mean()
        reg_loss = agent.get_metrics().get("criticality_loss", 0.0)
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * ent_loss
            + reg_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return total_reward, {
            "game": "single",
            "steps": len(obs_buf),
            "reward": total_reward,
            "policy_loss": policy_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "entropy": ent_loss.detach().item(),
            "metrics": agent.get_metrics(),
            "loss": policy_loss.detach().item() + value_loss.detach().item(),
        }

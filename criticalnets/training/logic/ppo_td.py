import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from . import TrainingLogic
from ...agents.ppo import PPOAgent


class PPOLogic(TrainingLogic):
    """
    Proximal Policy Optimization training logic
    """

    def __init__(self, config):
        self.lr = config.get("lr", 25e-5)
        self.gamma = config.get("gamma", 0.99)
        self.clip_eps = config.get("clip_eps", 0.1)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.rollout_length = config.get("rollout_length", 128)
        self.epochs = config.get("epochs", 4)
        self.batch_size = config.get("batch_size", 32)
        self.optimizer = None

    def configure_optimizer(self, agent: PPOAgent):
        # take override if user passed lr in config, else use self.lr
        self.optimizer = optim.Adam(agent.parameters(), lr=self.lr)
        return self.optimizer

    def run_episode(
        self, env, agent: PPOAgent, memory, episode_idx
    ) -> Tuple[float, Dict]:
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        while not done:
            # Run rollout_length steps (or less if episode ends earlier)
            for _ in range(self.rollout_length):
                st = torch.from_numpy(state).float().to(agent.device)
                logits, val = agent.forward(st.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

                # Prevent invalid action if needed (specific Atari constraint)
                action = min(action, env.action_space.n - 1)

                logp = dist.log_prob(torch.tensor(action).to(agent.device)).detach()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(st)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(logp)
                values.append(val.squeeze(0).detach())

                state = next_state
                total_reward += reward

                if done:
                    break  # Exit inner loop immediately on actual termination

            # Compute GAE and returns for collected rollout
            last_val = (
                0
                if done
                else agent.forward(
                    torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                )[1].item()
            )

            returns, advs = [], []
            gae = 0
            for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
                mask = 1.0 - float(d)
                delta = r + self.gamma * last_val * mask - v.item()
                gae = delta + self.gamma * 0.95 * mask * gae
                advs.insert(0, gae)
                last_val = v.item()

            returns = [a + v.item() for a, v in zip(advs, values)]

            states_t = torch.stack(states)
            actions_t = torch.tensor(actions, dtype=torch.int64).to(agent.device)
            old_logp_t = torch.stack(log_probs)
            returns_t = torch.tensor(returns).float().to(agent.device)
            advs_t = (
                torch.tensor(advs).float().to(agent.device) - returns_t.mean()
            ) / (returns_t.std() + 1e-8)

            # PPO update epochs
            for _ in range(self.epochs):
                idxs = np.random.permutation(len(states))
                for start in range(0, len(states), self.batch_size):
                    batch_idx = idxs[start : start + self.batch_size]
                    b_states = states_t[batch_idx]
                    b_actions = actions_t[batch_idx]
                    b_old_logp = old_logp_t[batch_idx]
                    b_returns = returns_t[batch_idx]
                    b_advs = advs_t[batch_idx]

                    logits, vals = agent.forward(b_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(b_actions)
                    ratio = torch.exp(logp - b_old_logp)

                    surr1 = ratio * b_advs
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * b_advs
                    )

                    policy_loss = -torch.min(surr1, surr2).mean()
                    # Ensure both vals and b_returns have the same shape
                    vals = vals.view(-1)
                    b_returns = b_returns.view(-1)

                    # Confirm sizes match exactly before computing loss
                    assert (
                        vals.shape == b_returns.shape
                    ), f"vals shape {vals.shape} vs b_returns shape {b_returns.shape}"

                    value_loss = F.mse_loss(vals, b_returns)

                    entropy = dist.entropy().mean()
                    reg_loss = agent.get_metrics().get('criticality_loss',0.0)
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                        + reg_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    # print("loss:", loss.item())

            # Clear buffers after update
            states.clear()
            actions.clear()
            rewards.clear()
            dones.clear()
            log_probs.clear()
            values.clear()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
        return total_reward, {
            "metrics": metrics,
            "steps": len(states),
            "reward": total_reward,
            "loss": loss.detach().item(),
        }

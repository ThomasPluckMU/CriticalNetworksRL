import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from dynamic_bias_layers import DynamicBiasCNN
import time

# Register Atari environments
gym.register_envs(ale_py)

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000
LR = 0.01
FRAME_STACK = 4
RENDER = True  # Flag to enable/disable rendering

# Frame preprocessing
def preprocess_frame(frame):
    frame = frame.mean(axis=2)  # Convert to grayscale
    frame = frame[34:194, 8:152]  # Crop
    frame = frame[::2, ::2]  # Downsample
    return frame.astype(np.float32) / 255.0

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN with Dynamic Bias layers
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = DynamicBiasCNN(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = DynamicBiasCNN(32, 64, kernel_size=4, stride=2)
        self.conv3 = DynamicBiasCNN(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(1920,512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.head(x)
    
    def reset_bias(self):
        for module in self.children():
            if hasattr(module, 'reset_bias'):
                module.reset_bias()

# Training setup
env = gym.make('ALE/Breakout-v5', render_mode='human' if RENDER else None)
n_actions = env.action_space.n
policy_net = DQN(n_actions)
print(policy_net)
target_net = DQN(n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

# Training loop
def train():
    # Initialize state with stacked frames
    frame = env.reset()[0]
    frame = preprocess_frame(frame)
    state = np.stack([frame] * FRAME_STACK, axis=0)
    state = torch.FloatTensor(state).unsqueeze(0)
    
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    
    for episode in tqdm(range(1000)):
        frame = env.reset()[0]
        frame = preprocess_frame(frame)
        state = np.stack([frame] * FRAME_STACK, axis=0)
        state = torch.FloatTensor(state).unsqueeze(0)
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            if RENDER:
                # Small delay to make visualization watchable
                time.sleep(0.01)
            
            # Epsilon-greedy action selection
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                np.exp(-1. * total_steps / EPS_DECAY)
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
            
            # Step environment
            next_frame, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            next_frame = preprocess_frame(next_frame)
            
            # Create next state by shifting frames and adding new one
            next_state = state.clone()
            next_state = next_state.squeeze(0)
            next_state = torch.cat([next_state[1:], torch.FloatTensor(next_frame).unsqueeze(0).to(torch.float32)], dim=0)
            next_state = next_state.unsqueeze(0)
            
            # Store transition
            memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([done]))
            
            # Move to next state
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Optimize model
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                
                # Compute loss
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                next_state_batch = torch.cat(batch.next_state)
                done_batch = torch.cat(batch.done).to(torch.float32)
                
                # Compute Q(s_t, a)
                state_action_values = policy_net(state_batch).gather(1, action_batch)
                
                # Compute V(s_{t+1})
                next_state_values = target_net(next_state_batch).max(1)[0].detach()
                expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
                
                # Compute loss
                loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Reset dynamic biases
                policy_net.reset_bias()
            
            # Update target network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Plot results
        if episode % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.subplot(122)
            plt.plot(episode_lengths)
            plt.title('Episode Lengths')
            plt.savefig(f'training_progress_episode_{episode}.png')  # Save the figure instead of showing it
            plt.close()
    
    env.close()
    return episode_rewards, episode_lengths

# Function to watch the trained agent play
def watch_agent(episodes=5):
    # Create a separate environment for visualization
    watch_env = gym.make('ALE/Breakout-v5', render_mode='human')
    
    for episode in range(episodes):
        frame = watch_env.reset()[0]
        frame = preprocess_frame(frame)
        state = np.stack([frame] * FRAME_STACK, axis=0)
        state = torch.FloatTensor(state).unsqueeze(0)
        
        done = False
        total_reward = 0
        
        while not done:
            # Use the policy network to choose actions
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            
            # Take the action
            next_frame, reward, terminated, truncated, _ = watch_env.step(action.item())
            done = terminated or truncated
            next_frame = preprocess_frame(next_frame)
            
            # Update state
            next_state = state.clone()
            next_state = next_state.squeeze(0)
            next_state = torch.cat([next_state[1:], torch.FloatTensor(next_frame).unsqueeze(0)], dim=0)
            next_state = next_state.unsqueeze(0)
            
            state = next_state
            total_reward += reward
            
            # Add a small delay to make the visualization watchable
            time.sleep(0.03)
        
        print(f"Episode {episode+1} finished with reward {total_reward}")
    
    watch_env.close()

# Start training with visualization
rewards, lengths = train()

# After training, watch the agent play
print("Training complete. Watching the trained agent play...")
watch_agent()
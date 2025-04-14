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
import os
import signal
import sys
import traceback

# Get all available single-player Atari games
import ale_py
import gymnasium as gym
from gymnasium.envs.registration import registry

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

# Define a list of Atari games known to work reliably
RELIABLE_GAMES = [
    "ALE/Pong-v5",
    "ALE/Breakout-v5", 
    "ALE/SpaceInvaders-v5",
    "ALE/MsPacman-v5", 
    "ALE/Asterix-v5",
    "ALE/Boxing-v5",
    "ALE/Freeway-v5", 
    "ALE/Seaquest-v5",
    "ALE/Assault-v5",
    "ALE/BeamRider-v5",
    "ALE/Centipede-v5",
    "ALE/DemonAttack-v5",
    "ALE/Enduro-v5",
    "ALE/Phoenix-v5",
    "ALE/Qbert-v5",
    "ALE/Riverraid-v5",
    "ALE/RoadRunner-v5",
    "ALE/Solaris-v5",
    "ALE/TimePilot-v5",
    "ALE/Tutankham-v5",
    "ALE/UpNDown-v5",
    "ALE/Venture-v5",
    "ALE/VideoPinball-v5",
    "ALE/WizardOfWor-v5",
    "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5"
]

# Actually verify which games work on this system
ATARI_GAMES = []
print("Verifying working Atari games...")
for game in RELIABLE_GAMES:
    try:
        env = gym.make(game)
        env.reset()  # Actually try to reset to verify it works
        env.close()
        ATARI_GAMES.append(game)
        print(f"✓ {game} works!")
    except Exception as e:
        print(f"✗ {game} failed: {e}")

if not ATARI_GAMES:
    print("FATAL ERROR: No working Atari games found!")
    print("Please install the Atari ROMs properly and try again.")
    sys.exit(1)
    
print(f"Found {len(ATARI_GAMES)} working Atari games for training")

# Get maximum action space size across all games
def get_max_action_space():
    max_actions = 0
    games_tried = 0
    
    # Try to sample games until we find some that work
    for _ in range(min(50, len(ATARI_GAMES))):  # Try up to 50 games or all games, whichever is smaller
        game = random.choice(ATARI_GAMES)
        try:
            env = gym.make(game)
            n_actions = env.action_space.n
            max_actions = max(max_actions, n_actions)
            env.close()
            games_tried += 1
            print(f"Successfully checked game: {game}, actions: {n_actions}")
        except Exception as e:
            # Just print error and continue with next game
            print(f"Error checking action space for {game}: {e}")
    
    # Ensure we have at least some default max actions if all games failed
    if max_actions == 0:
        max_actions = 18  # Default max for Atari games
        print("Warning: Could not determine max action space. Using default of 18.")
    else:
        print(f"Successfully checked {games_tried} games. Maximum action space: {max_actions}")
    
    return max_actions

MAX_ACTIONS = get_max_action_space()
print(f"Maximum action space across all games: {MAX_ACTIONS}")

# Hyperparameters
BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000
LR = 0.01
FRAME_STACK = 4
RENDER = True  # Flag is set to True by default to ensure rendering stays on for all games
SAVE_DIR = 'model_checkpoints'
RENDER_DELAY = 0.01  # Initial render delay - can be adjusted during runtime

# For keyboard control of rendering speed
import threading
import msvcrt  # Windows-specific keyboard module

# Keyboard control thread
def keyboard_control():
    global RENDER_DELAY
    print("\nKeyboard controls:")
    print("  + : Increase render delay (slower)")
    print("  - : Decrease render delay (faster)")
    print("  0 : Reset to default delay")
    print("  f : Toggle super fast mode (no delay)")
    
    fast_mode = False
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore')
            if key == '+':
                RENDER_DELAY += 0.005
                print(f"Render delay: {RENDER_DELAY:.3f}s")
            elif key == '-':
                RENDER_DELAY = max(0, RENDER_DELAY - 0.005)
                print(f"Render delay: {RENDER_DELAY:.3f}s")
            elif key == '0':
                RENDER_DELAY = 0.01
                print(f"Render delay reset to default: {RENDER_DELAY:.3f}s")
            elif key == 'f':
                fast_mode = not fast_mode
                RENDER_DELAY = 0 if fast_mode else 0.01
                print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
        time.sleep(0.1)  # Check for key presses 10 times per second

# Create save directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Frame preprocessing
def preprocess_frame(frame):
    frame = frame.mean(axis=2)  # Convert to grayscale
    frame = frame[34:194, 8:152]  # Crop
    frame = frame[::2, ::2]  # Downsample
    return frame.astype(np.float32) / 255.0

# Experience replay memory with game ID
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'game_id'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN with Dynamic Bias layers - unified action space
class UnifiedDQN(nn.Module):
    def __init__(self, max_actions, layer_type='gated'):
        super(UnifiedDQN, self).__init__()
        if layer_type == 'gated':
            from dynamic_bias_layers import GatedDynamicBiasCNN as DynamicLayer
        else:
            from dynamic_bias_layers import DeadWeightDynamicBiasCNN as DynamicLayer
            
        self.conv1 = DynamicLayer(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv1._initialize_parameters((1, 32, 19, 17))
        self.conv2 = DynamicLayer(32, 64, kernel_size=4, stride=2)
        self.conv2._initialize_parameters((1, 64, 8, 7))
        self.conv3 = DynamicLayer(64, 64, kernel_size=3, stride=1)
        self.conv3._initialize_parameters((1, 64, 6, 5))
        self.fc = nn.Linear(1920, 512)
        self.head = nn.Linear(512, max_actions)
        
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

# Game-specific action masks
class ActionMaskManager:
    def __init__(self):
        self.game_action_masks = {}
        
    def get_mask(self, game_id, device='cpu'):
        """Get action mask for a specific game"""
        if game_id in self.game_action_masks:
            return self.game_action_masks[game_id]
        
        # Create new mask for this game
        try:
            env = gym.make(game_id)
            n_actions = env.action_space.n
            mask = torch.zeros(MAX_ACTIONS, device=device)
            mask[:n_actions] = 1.0  # Set valid actions to 1
            self.game_action_masks[game_id] = mask
            env.close()
            return mask
        except Exception as e:
            print(f"Error creating action mask for {game_id}: {e}")
            # Default to full mask if error
            mask = torch.ones(MAX_ACTIONS, device=device)
            self.game_action_masks[game_id] = mask
            return mask

# Save model weights
def save_model(policy_net, game_name, episode_count):
    save_path = os.path.join(SAVE_DIR, f"{game_name.replace('/', '_')}_{episode_count}.pt")
    torch.save({
        'state_dict': policy_net.state_dict(),
        'max_actions': MAX_ACTIONS,
        'games': ATARI_GAMES,
        'timestamp': datetime.now().isoformat()
    }, save_path)
    print(f"Model checkpoint saved to {save_path}")

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print('KeyboardInterrupt detected! Saving model before exit...')
    if 'policy_net' in globals() and 'current_game' in globals() and 'total_episodes' in globals():
        save_model(policy_net, current_game, total_episodes)
    print('Model saved. Exiting...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Get valid action for epsilon-greedy selection
def get_valid_action(q_values, action_mask, eps_threshold):
    """
    Select a valid action using epsilon-greedy strategy with action masking
    """
    if random.random() > eps_threshold:
        # Mask out invalid actions by setting their q-values to -infinity
        masked_q_values = q_values + (action_mask - 1) * 1e9
        return masked_q_values.max(1)[1].view(1, 1)
    else:
        # For random action, only sample from valid actions
        valid_actions = torch.nonzero(action_mask).squeeze()
        if valid_actions.dim() == 0:  # Handle case where only one valid action
            return torch.tensor([[valid_actions.item()]], dtype=torch.long)
        else:
            random_idx = random.randint(0, len(valid_actions) - 1)
            return torch.tensor([[valid_actions[random_idx]]], dtype=torch.long)

# Training loop
def train_infinite():
    global policy_net, current_game, total_episodes
    
    # Initialize environment with a random game
    if not ATARI_GAMES:
        print("Error: No working Atari games available!")
        sys.exit(1)
        
    current_game = random.choice(ATARI_GAMES)
    print(f"Starting with game: {current_game}")
    
    # Since we've verified all games in ATARI_GAMES work, this should succeed
    env = gym.make(current_game, render_mode='human' if RENDER else None)
    print(f"Successfully loaded {current_game} environment")
    
    # Initialize the action mask manager
    mask_manager = ActionMaskManager()
    
    # Initialize networks with specified layer type
    policy_net = UnifiedDQN(MAX_ACTIONS, args.layer_type)
    target_net = UnifiedDQN(MAX_ACTIONS, args.layer_type)
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        if isinstance(checkpoint, dict):  # New format with metadata
            policy_net.load_state_dict(checkpoint['state_dict'], 
                                     strict=not args.ignore_shape_mismatch)
        else:  # Old format (just state_dict)
            policy_net.load_state_dict(checkpoint, 
                                     strict=not args.ignore_shape_mismatch)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    total_steps = 0
    total_episodes = 0
    game_episodes = 0
    all_rewards = []
    game_switch_points = []
    
    try:
        while True:  # Infinite loop
            # Check if we need to switch to a new game
            if game_episodes >= 10:
                # Close current environment
                env.close()
                
                # Choose a new random game
                prev_game = current_game
                while current_game == prev_game and len(ATARI_GAMES) > 1:
                    current_game = random.choice(ATARI_GAMES)
                
                print(f"\n===== Switching from {prev_game} to {current_game} after {game_episodes} episodes =====\n")
                
                # Create new environment
                try:
                    env.close()  # Close current environment first
                except Exception:
                    pass  # Ignore errors when closing the environment
                
                # Create new environment with a verified working game
                current_game = random.choice([g for g in ATARI_GAMES if g != prev_game or len(ATARI_GAMES) == 1])
                env = gym.make(current_game, render_mode='human' if RENDER else None)
                print(f"Successfully switched to {current_game}")
                
                # Get action mask for the new game
                action_mask = mask_manager.get_mask(current_game)
                print(f"Action space for {current_game}: {int(action_mask.sum())} of {MAX_ACTIONS}")
                
                # Record game switch point for plotting
                game_switch_points.append(len(all_rewards))
                
                # Reset episode counter for the new game
                game_episodes = 0
            
            # Get current game's action mask
            action_mask = mask_manager.get_mask(current_game)
            
            # Reset environment with verified game
            try:
                frame = env.reset()[0]
                frame = preprocess_frame(frame)
                state = np.stack([frame] * FRAME_STACK, axis=0)
                state = torch.FloatTensor(state).unsqueeze(0)
            except Exception as e:
                print(f"Unexpected error resetting environment: {e}")
                # This shouldn't happen with verified games, but just in case
                try:
                    env.close()
                except:
                    pass
                
                # Find another game
                prev_game = current_game
                current_game = random.choice([g for g in ATARI_GAMES if g != prev_game or len(ATARI_GAMES) == 1])
                
                # Create new environment
                env = gym.make(current_game, render_mode='human' if RENDER else None)
                print(f"Switched to {current_game} after reset error")
                
                # Reset the new environment
                frame = env.reset()[0]
                frame = preprocess_frame(frame)
                state = np.stack([frame] * FRAME_STACK, axis=0)
                state = torch.FloatTensor(state).unsqueeze(0)
                
                # Reset game episodes counter
                game_episodes = 0
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                if RENDER:
                    # Use dynamic rendering delay
                    time.sleep(RENDER_DELAY)
                
                # Epsilon-greedy action selection with action masking
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.exp(-1. * total_steps / EPS_DECAY)
                
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = get_valid_action(q_values, action_mask, eps_threshold)
                
                # Step environment with error handling
                try:
                    next_frame, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    next_frame = preprocess_frame(next_frame)
                    
                    # Create next state by shifting frames and adding new one
                    next_state = state.clone()
                    next_state = next_state.squeeze(0)
                    next_state = torch.cat([next_state[1:], torch.FloatTensor(next_frame).unsqueeze(0).to(torch.float32)], dim=0)
                    next_state = next_state.unsqueeze(0)
                    
                    # Store transition with game ID
                    memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([done]), current_game)
                    
                    # Move to next state
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    total_steps += 1
                except Exception as e:
                    print(f"Unexpected error during step execution: {e}")
                    # This should not happen with our verified games, but just in case:
                    done = True  # End this episode
                
                # Optimize model
                if len(memory) > BATCH_SIZE:
                    try:
                        transitions = memory.sample(BATCH_SIZE)
                        batch = Transition(*zip(*transitions))
                        
                        # Compute loss
                        state_batch = torch.cat(batch.state)
                        action_batch = torch.cat(batch.action)
                        reward_batch = torch.cat(batch.reward)
                        next_state_batch = torch.cat(batch.next_state)
                        done_batch = torch.cat(batch.done).to(torch.float32)
                        game_ids = batch.game_id
                        
                        # Get masks for all games in the batch
                        masks = []
                        for game_id in game_ids:
                            masks.append(mask_manager.get_mask(game_id))
                        action_masks = torch.stack(masks)
                        
                        # Compute Q(s_t, a)
                        state_action_values = policy_net(state_batch).gather(1, action_batch)
                        
                        # Compute V(s_{t+1}) with action masking
                        if args.no_target_model:
                            next_q_values = policy_net(next_state_batch)
                        else:
                            next_q_values = target_net(next_state_batch)
                        # Apply masks to exclude invalid actions
                        next_q_values = next_q_values + (action_masks - 1) * 1e9
                        next_state_values = next_q_values.max(1)[0].detach()
                        
                        if args.reward_mode == 'instant':
                            expected_state_action_values = reward_batch
                        else:
                            expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
                        
                        # Compute loss
                        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                        
                        # Optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Reset dynamic biases
                        policy_net.reset_bias()
                    except Exception as e:
                        print(f"Error during optimization: {e}")
                        # Save model in case of crash
                        save_model(policy_net, current_game, total_episodes)
                        raise
                
                # Update target network if not in no-target-model mode
                if not args.no_target_model and total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            # Episode complete
            all_rewards.append(episode_reward)
            total_episodes += 1
            game_episodes += 1
            
            print(f"Game: {current_game} | Episode: {total_episodes} | " +
                  f"Game Episodes: {game_episodes}/10 | Reward: {episode_reward}")
            
            # Plot results periodically
            if total_episodes % 10 == 0:
                plt.figure(figsize=(15, 5))
                plt.plot(all_rewards)
                plt.title(f'Episode Rewards (Total Episodes: {total_episodes})')
                
                # Mark game switches
                for switch_point in game_switch_points:
                    plt.axvline(x=switch_point, color='r', linestyle='--', alpha=0.5)
                
                plt.savefig(f'training_progress_episode_{total_episodes}.png')
                plt.close()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_model(policy_net, current_game, total_episodes)
    
    except Exception as e:
        print(f"\nTraining terminated due to error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        save_model(policy_net, current_game, total_episodes)
    
    finally:
        # Always try to close the environment
        try:
            env.close()
        except:
            pass

def train_single_game(game_name, total_episodes=1000):
    """Train the agent on a single specified Atari game."""
    global policy_net
    
    # Verify game is available
    if game_name not in ATARI_GAMES:
        print(f"Error: {game_name} is not in the list of working Atari games")
        print(f"Available games: {ATARI_GAMES}")
        sys.exit(1)
    
    # Initialize environment
    env = gym.make(game_name, render_mode='human' if RENDER else None)
    print(f"Training on single game: {game_name} for {total_episodes} episodes")
    
    # Initialize networks with specified layer type
    policy_net = UnifiedDQN(MAX_ACTIONS, args.layer_type)
    target_net = UnifiedDQN(MAX_ACTIONS, args.layer_type)
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        if isinstance(checkpoint, dict):  # New format with metadata
            policy_net.load_state_dict(checkpoint['state_dict'],
                                     strict=not args.ignore_shape_mismatch)
        else:  # Old format (just state_dict)
            policy_net.load_state_dict(checkpoint,
                                     strict=not args.ignore_shape_mismatch)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    mask_manager = ActionMaskManager()
    action_mask = mask_manager.get_mask(game_name)
    
    # Training statistics
    all_rewards = []
    total_steps = 0
    
    try:
        for episode in range(1, total_episodes + 1):
            # Reset environment
            frame = env.reset()[0]
            frame = preprocess_frame(frame)
            state = np.stack([frame] * FRAME_STACK, axis=0)
            state = torch.FloatTensor(state).unsqueeze(0)
            
            done = False
            episode_reward = 0
            
            while not done:
                if RENDER:
                    time.sleep(RENDER_DELAY)
                
                # Epsilon-greedy action selection
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.exp(-1. * total_steps / EPS_DECAY)
                
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = get_valid_action(q_values, action_mask, eps_threshold)
                
                # Step environment
                next_frame, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                next_frame = preprocess_frame(next_frame)
                
                # Store transition
                next_state = state.clone()
                next_state = next_state.squeeze(0)
                next_state = torch.cat([next_state[1:], torch.FloatTensor(next_frame).unsqueeze(0)], dim=0)
                next_state = next_state.unsqueeze(0)
                
                memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([done]), game_name)
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # Optimize model
                if len(memory) > BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    next_state_batch = torch.cat(batch.next_state)
                    done_batch = torch.cat(batch.done).to(torch.float32)
                    
                    # Compute loss
                    state_action_values = policy_net(state_batch).gather(1, action_batch)
                    if args.no_target_model:
                        next_q_values = policy_net(next_state_batch)
                    else:
                        next_q_values = target_net(next_state_batch)
                    next_q_values = next_q_values + (action_mask - 1) * 1e9
                    next_state_values = next_q_values.max(1)[0].detach()
                    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
                    
                    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    policy_net.reset_bias()
                
                # Update target network if not in no-target-model mode
                if not args.no_target_model and total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            # Episode complete
            all_rewards.append(episode_reward)
            print(f"Episode: {episode}/{total_episodes} | Reward: {episode_reward}")
            
            # Save model periodically
            if episode % 100 == 0:
                save_path = os.path.join(SAVE_DIR, f"{game_name.replace('/', '_')}_single_{episode}.pt")
                torch.save(policy_net.state_dict(), save_path)
                print(f"Saved model checkpoint to {save_path}")
                
                # Plot training progress
                plt.figure(figsize=(15, 5))
                plt.plot(all_rewards)
                plt.title(f'Single-Game Training: {game_name} (Episode {episode})')
                plt.savefig(f'training_progress_{game_name.replace("/", "_")}_{episode}.png')
                plt.close()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_path = os.path.join(SAVE_DIR, f"{game_name.replace('/', '_')}_single_interrupted.pt")
        torch.save(policy_net.state_dict(), save_path)
    
    except Exception as e:
        print(f"\nTraining error: {e}")
        traceback.print_exc()
        save_path = os.path.join(SAVE_DIR, f"{game_name.replace('/', '_')}_single_error.pt")
        torch.save(policy_net.state_dict(), save_path)
    
    finally:
        env.close()


# Function to watch the trained agent play
def watch_agent(model_path, game_name, episodes=5):
    # Create environment for visualization
    watch_env = gym.make(game_name, render_mode='human')
    
    # Load model with unified action space
    model = UnifiedDQN(MAX_ACTIONS)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get action mask for this game
    mask_manager = ActionMaskManager()
    action_mask = mask_manager.get_mask(game_name)
    
    for episode in range(episodes):
        frame = watch_env.reset()[0]
        frame = preprocess_frame(frame)
        state = np.stack([frame] * FRAME_STACK, axis=0)
        state = torch.FloatTensor(state).unsqueeze(0)
        
        done = False
        total_reward = 0
        
        while not done:
            # Use the policy network to choose actions with masking
            with torch.no_grad():
                q_values = model(state)
                # Apply action mask
                masked_q_values = q_values + (action_mask - 1) * 1e9
                action = masked_q_values.max(1)[1].view(1, 1)
            
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

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    # Start keyboard control thread
    if RENDER:
        keyboard_thread = threading.Thread(target=keyboard_control, daemon=True)
        keyboard_thread.start()
        print("Keyboard control thread started")

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train RL agent on Atari games')
    parser.add_argument('--mode', choices=['multi', 'single'], required=True,
                      help='Training mode: "multi" for multiple games, "single" for one game')
    parser.add_argument('--game', help='Game name (required for single mode)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to train (default: 1000)')
    parser.add_argument('--checkpoint', 
                      help='Path to checkpoint file to resume training')
    parser.add_argument('--ignore_shape_mismatch', action='store_true',
                      help='Allow loading checkpoints with different network architecture')
    parser.add_argument('--layer-type', choices=['gated', 'deadweight'], default='gated',
                      help='Dynamic bias layer variant (default: gated)')
    parser.add_argument('--reward-mode', choices=['instant', 'discounted'], default='discounted',
                      help='Reward calculation mode (default: discounted)')
    parser.add_argument('--no-target-model', action='store_true',
                      help='Bypass target network and use policy net directly')
    args = parser.parse_args()

    if args.mode == 'single':
        if not args.game:
            print("Error: Must specify --game for single mode")
            sys.exit(1)
        print(f"Starting single-game training on {args.game} for {args.episodes} episodes...")
        print(f"Press Ctrl+C to stop training and save the model")
        print(f"Rendering is {'ON' if RENDER else 'OFF'}")
        print(f"Using unified action space with {MAX_ACTIONS} actions")
        train_single_game(args.game, args.episodes)
    else:
        print("Starting training with dynamic game selection...")
        print(f"Press Ctrl+C to stop training and save the model")
        print(f"Rendering is {'ON' if RENDER else 'OFF'}")
        print(f"Games will switch every 10 episodes, weights will be saved on interruption or error")
        print(f"Using unified action space with {MAX_ACTIONS} actions")
        train_infinite()

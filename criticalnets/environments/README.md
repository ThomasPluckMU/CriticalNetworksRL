# Environments Module

## Supported Games
The following Atari games are supported:
- ALE/Pong-v5
- ALE/Breakout-v5
- ALE/SpaceInvaders-v5
- ALE/MsPacman-v5
- ALE/Asterix-v5
- ALE/Boxing-v5
- ALE/Freeway-v5
- ALE/Seaquest-v5
- ALE/Assault-v5
- ALE/BeamRider-v5
- ALE/Centipede-v5
- ALE/DemonAttack-v5
- ALE/Enduro-v5
- ALE/Phoenix-v5
- ALE/Qbert-v5
- ALE/Riverraid-v5
- ALE/RoadRunner-v5
- ALE/Solaris-v5
- ALE/TimePilot-v5
- ALE/Tutankham-v5
- ALE/UpNDown-v5
- ALE/Venture-v5
- ALE/VideoPinball-v5
- ALE/WizardOfWor-v5
- ALE/YarsRevenge-v5
- ALE/Zaxxon-v5

## AtariManager Features
- Game verification on initialization
- Action space utilities
- Current game tracking
- Centralized game configuration

## Usage Examples
```python
from criticalnets.environments import AtariManager
import gymnasium as gym

# Initialize manager (auto-verifies games)
manager = AtariManager()

# Get working games
print(f"Available games: {manager.working_games}")

# Get max action space needed
print(f"Max actions: {manager.get_max_action_space()}")

# Set current game
manager.set_current_game("ALE/Pong-v5")

# Create environment
env = gym.make(manager.get_current_game())
```

## Adding New Games
1. Add game ID to `RELIABLE_GAMES` in `atari_manager.py`:
```python
RELIABLE_GAMES = [
    # Existing games...
    "ALE/NewGame-v5",  # Add new game IDs here
]
```
2. The manager will verify it works during initialization
3. Verified games will be available in `working_games`

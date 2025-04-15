# Environments Module

## 1. What is this module?
This module handles Atari game environment management. Key features:
- `AtariManager` class for environment setup and verification
- Game compatibility checking
- Action space utilities
- Centralized game configuration

## 2. How to use it
```python
from criticalnets.environments import AtariManager

# Create environment manager
manager = AtariManager()

# Get list of working games
working_games = manager.working_games

# Get maximum action space needed
max_actions = manager.get_max_action_space()

# Create an environment
env = gym.make(working_games[0])
```

## 3. How to add new games
To add support for new Atari games:
1. Add the game ID to `RELIABLE_GAMES` list in `atari_manager.py`
2. The manager will automatically verify it works during initialization
3. The game will be available if verification succeeds

Example game format:
```python
RELIABLE_GAMES = [
    # Existing games...
    "ALE/NewGame-v5",  # Add new game IDs here
]

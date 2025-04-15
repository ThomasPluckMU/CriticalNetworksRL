import gymnasium as gym
import ale_py
import numpy as np
from typing import List
import sys

class AtariManager:
    """Handles Atari environment setup and verification"""
    
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

    def __init__(self):
        self.working_games = self._verify_games()
        
    def _verify_games(self) -> List[str]:
        """Verify which Atari games work on this system"""
        working = []
        print("Verifying working Atari games...")
        for game in self.RELIABLE_GAMES:
            try:
                # Ensure we're using the full game name with version
                full_game = game if '-' in game else f"{game}-v4"
                env = gym.make(full_game)
                env.reset()
                env.close()
                working.append(full_game)
                print(f"[OK] {full_game} works!")
            except Exception as e:
                print(f"[X] {game} failed: {e}")
        
        if not working:
            print("FATAL ERROR: No working Atari games found!")
            sys.exit(1)
            
        print(f"Found {len(working)} working Atari games")
        return working

    def get_max_action_space(self) -> int:
        """Get maximum action space across all games"""
        max_actions = 0
        for game in self.working_games:
            try:
                env = gym.make(game)
                max_actions = max(max_actions, env.action_space.n)
                env.close()
            except:
                continue
        return max_actions if max_actions > 0 else 18  # Default fallback

    def get_current_game(self) -> str:
        """Get the current game being played"""
        return self.current_game if hasattr(self, 'current_game') else self.working_games[0]

    def set_current_game(self, game: str):
        """Set the current game being played"""
        self.current_game = game

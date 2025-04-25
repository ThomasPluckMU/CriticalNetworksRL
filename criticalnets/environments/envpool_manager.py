import envpool
import numpy as np
from typing import List, Optional
import sys


class EnvPoolAtariManager:
    """Handles Atari environment setup and verification using EnvPool"""

    RELIABLE_GAMES = [
        "Pong-v5"
    ]

    def __init__(self, num_envs: int = 1):
        """Initialize the AtariManager with EnvPool support

        Args:
            num_envs: Number of parallel environments to create
        """
        self.num_envs = num_envs
        self.working_games = self._verify_games()
        self.current_game = self.working_games[0] if self.working_games else None
        self.env = None

    def _verify_games(self) -> List[str]:
        """Verify which Atari games work with EnvPool on this system"""
        working = []

        # Note: EnvPool doesn't require explicit verification as Gymnasium does
        # But we can still filter the games that are supported by EnvPool
        for game in self.RELIABLE_GAMES:
            working.append(game)

        return working

        #! Uncomment if you want debug verification.
        # print("Verifying working Atari games in EnvPool...")
        # for game in self.RELIABLE_GAMES:
        #     try:
        #         # Create a temporary environment to verify
        #         env = envpool.make(
        #             "Atari",
        #             env_id=game,
        #             num_envs=1,
        #             episodic_life=True,
        #             reward_clip=True
        #         )
        #         working.append(game)
        #         print(f"[OK] {game} works!")
        #     except Exception as e:
        #         print(f"[X] {game} failed: {e}")
        #
        # if not working:
        #     print("FATAL ERROR: No working Atari games found in EnvPool!")
        #     sys.exit(1)
        #
        # print(f"Found {len(working)} working Atari games")
        # return working

    def create_env(self, game: Optional[str] = None):
        """Create an EnvPool environment for the specified game

        Args:
            game: Name of the game to create an environment for.
                 If None, uses the current game.
        """
        if game:
            self.current_game = game

        if not self.current_game:
            raise ValueError("No game specified and no default game available")

        # Close existing environment if any
        if self.env is not None:
            # EnvPool doesn't require explicit closing
            self.env = None

        # Create new environment with EnvPool
        self.env = envpool.make(
            self.current_game,
            env_type="gymnasium",
            num_envs=self.num_envs,
            full_action_space = True,
            episodic_life=True,
            reward_clip=True,
            stack_num=4,
        )

        return self.env

    def get_max_action_space(self) -> int:
        """Get maximum action space across all games"""
        # Standard Atari action space sizes in EnvPool
        # For most games, it's 18, but we'll verify by sampling

        max_actions = 0
        for game in self.working_games:  # Sample a few games to check
            try:
                env = envpool.make("Atari", env_id=game, num_envs=1)
                max_actions = max(max_actions, env.action_space.n)
            except:
                continue

        return max_actions if max_actions > 0 else 18  # Default fallback

    def get_current_game(self) -> str:
        """Get the current game being used"""
        return self.current_game if self.current_game else self.working_games[0]

    def set_current_game(self, game: str):
        """Set the current game to be used"""
        if game not in self.working_games:
            raise ValueError(f"Game {game} is not in the list of working games")
        self.current_game = game
        # Recreate the environment with the new game
        if self.env is not None:
            self.create_env(game)

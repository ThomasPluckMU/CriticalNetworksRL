import unittest
import subprocess
from pathlib import Path
import sys
import json
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLIConfigurations(unittest.TestCase):
    """Test different CLI configurations for the Atari training script"""

    TEST_CONFIGS = [
        # Basic configurations
        {
            "game": "Breakout-v4",
            "agent": "GatedAtariUDQN",
            "logic": "criticalnets.training.logic.single_game.SingleGameLogic",
            "episodes": 1,
            "render": False,
        },
        {
            "game": "Pong-v4",
            "agent": "GatedAtariUDQN",
            "logic": "criticalnets.training.logic.single_game.SingleGameLogic",
            "episodes": 1,
            "render": False,
        },
        # Different hyperparameters
        {
            "game": "Breakout-v4",
            "agent": "GatedAtariUDQN",
            "logic": "criticalnets.training.logic.single_game.SingleGameLogic",
            "episodes": 1,
            "render": False,
            "lr": 0.0001,
            "memory-size": 50000,
        },
        # Multi-game configuration - temporarily disabled due to timeout issues
        # TODO: Implement proper mocking for multi-game CLI testing
        # {
        #     'multi': True,
        #     'agent': 'GatedAtariUDQN',
        #     'logic': 'criticalnets.training.logic.multi_game.MultiGameLogic',
        #     'episodes': 1,
        #     'render': False
        # }
    ]

    def test_all_configurations(self):
        """Test all configurations in sequence"""
        root_dir = Path(__file__).parent.parent
        script_path = root_dir / "scripts" / "train_atari.py"

        results = {}
        for config in self.TEST_CONFIGS:
            test_name = f"{config.get('game', 'multi')}_{config['agent']}"
            try:
                cmd = ["python", str(script_path)]

                if "multi" in config:
                    cmd.append("--multi")
                else:
                    cmd.extend(["--single", config["game"]])

                cmd.extend(
                    [
                        "--agent",
                        config["agent"],
                        "--logic",
                        config["logic"],
                        "--episodes",
                        str(config["episodes"]),
                    ]
                )

                if config.get("render", False):
                    cmd.append("--render")

                if "lr" in config:
                    cmd.extend(["--lr", str(config["lr"])])

                if "memory-size" in config:
                    cmd.extend(["--memory-size", str(config["memory-size"])])

                # Run the command with timeout and pipes
                timeout = 60  # 60 second timeout per test
                result = subprocess.run(
                    cmd,
                    cwd=root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                # Print captured output for debugging
                print(f"\n=== {test_name} OUTPUT ===")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

                if result.returncode == 0:
                    results[test_name] = {"status": "PASSED"}
                else:
                    results[test_name] = {"status": "FAILED", "error": result.stderr}

            except subprocess.TimeoutExpired as e:
                results[test_name] = {
                    "status": "TIMEOUT",
                    "error": f"Test timed out after {timeout} seconds: {str(e)}",
                }
            except Exception as e:
                results[test_name] = {"status": "ERROR", "error": str(e)}

        # Save results to file
        with open(root_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Check if all tests passed
        failed = [k for k, v in results.items() if v["status"] != "PASSED"]
        if failed:
            self.fail(f"Some configurations failed: {failed}")


if __name__ == "__main__":
    unittest.main()

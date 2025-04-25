#!/usr/bin/env python3
"""
Atari Training CLI Entry Point

Usage:
  train_atari.py [--config <path>] [options]

Options:
  --config <path>      Path to YAML config file (overrides other options)
  --trainer            Choose a valid trainer class from the training directory
  --game               Game from the py-ALE library
  --episodes <count>   Number of episodes [default: 1000]
  --render             Enable rendering
  --checkpoint <path>  Load existing checkpoint
  --save-dir <path>    Model save directory [default: model_checkpoints]
  --lr <rate>          Learning rate [default: 0.001]
  --memory-size <size> Replay memory size [default: 100000]
  --agent <name>       Agent class name from agents directory
  --logic <path>       Training logic class path
  --log-dir <path>     Debug log directory [default: logs]
  --debug              Enable debug logging of activations/gradients
"""
import argparse
import os
import sys
from pathlib import Path
import gymnasium as gym
import hashlib, json
from criticalnets.configs import ConfigHandler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from criticalnets.environments.envpool_manager import EnvPoolAtariManager
from criticalnets.agents import get_agent_class
from criticalnets.training.logic import get_logic_class
from criticalnets.training import get_trainer_class


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Atari games")
    parser.add_argument(
        "--config", type=str, help="Path to YAML config file (overrides other options)"
    )
    parser.add_argument(
        "--trainer", type=str, help="Trainer class from training directory"
    )
    parser.add_argument("--game", type=str, help="Choose a game available in py-ALE")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--save-dir", type=str, default="model_checkpoints", help="Model save directory"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--memory-size", type=int, default=100000, help="Replay memory size"
    )
    parser.add_argument(
        "--agent", type=str, help="Agent class name from agents directory"
    )
    parser.add_argument(
        "--logic",
        type=str,
        help="Training logic class name from training/logic/ directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save debug logs"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging of activations/gradients",
    )
    return parser.parse_args()


def run_training(config: dict, run_id: int = 0):
    """Run training with given config"""
    import logging
    from datetime import datetime

    # Setup logging - Reset root logger first
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = os.path.join(config["save_dir"], f"run_{run_id}.log")
    logging.basicConfig(
        level=logging.DEBUG if config.get("debug", False) else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    config["run_id"] = run_id

    try:
        logger.info(f"Starting training run {run_id}")
        logger.info(f"Config: {config}")

        trainer_cls = get_trainer_class(config["trainer"])
        agent_cls = get_agent_class(config["agent"])
        logic_cls = get_logic_class(config["logic"])

        # Convert learning_rate from string to float if needed
        if "hyperparameters" in config and "learning_rate" in config["hyperparameters"]:
            try:
                config["hyperparameters"]["learning_rate"] = float(
                    config["hyperparameters"]["learning_rate"]
                )
            except (ValueError, TypeError):
                raise ValueError("Invalid learning_rate format - must be numeric")

        # Handle both single game and multi-game cases
        games = config.get("games", [config.get("game")])
        if not games or not any(games):
            raise ValueError("No game specified in config")

        # Train on all specified games
        for game in games:
            if not game:
                continue

            env = gym.make(game)
            config["action_space"] = env.action_space.n
            env.close()

            trainer = trainer_cls(
                config, logic_cls, agent_cls, atari_manager=EnvPoolAtariManager()
            )
            trainer.train(game, config.get("episodes", 1000))

        logger.info(f"Successfully completed training run {run_id}")
        return True
    except Exception as e:
        logger.error(
            f"Failed training {config['agent']} with {config['logic']}: {str(e)}",
            exc_info=True,
        )
        if config.get("debug", False):
            raise
        return False


def config_to_hex(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def main():
    args = parse_args()

    if args.config:
        # Use config file mode
        config_handler = ConfigHandler(args.config)
        configs = config_handler.generate_configs()
        total_runs = len(configs)
        successful_runs = 0

        for idx, config in enumerate(configs):

            run_training(config, run_id=config_to_hex(config))
            print(f"\n=== Completed run {idx+1}/{total_runs} ===\n")

    else:
        # Legacy CLI args mode
        if not all([args.trainer, args.game, args.agent, args.logic]):
            raise ValueError("Missing required arguments when not using --config")

        config = {
            "trainer": args.trainer,
            "game": args.game,
            "agent": args.agent,
            "logic": args.logic,
            "render": args.render,
            "save_dir": args.save_dir,
            "lr": args.lr,
            "memory_size": args.memory_size,
            "log_dir": args.log_dir,
            "debug": args.debug,
            "episodes": args.episodes,
        }
        os.makedirs(args.save_dir, exist_ok=True)
        run_training(config, run_id=config_to_hex(config))


if __name__ == "__main__":
    main()

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
  --parallel           Enable parallel training across GPUs
  --max-jobs <num>     Maximum number of parallel jobs [default: 8]
"""
import argparse
import os
import sys
from pathlib import Path
import hashlib, json
import torch
import concurrent.futures
import multiprocessing as mp
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
    # New options for parallel training
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel training across GPUs"
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=8,
        help="Maximum number of parallel jobs"
    )
    return parser.parse_args()


def train_game(config, game, gpu_id, run_id):
    """Train a single game with specified config on a specific GPU"""

    import logging
    from datetime import datetime

    # Setup logging - Reset root logger first
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = os.path.join(config["save_dir"], f"run_{run_id}_{game}.log")
    logging.basicConfig(
        level=logging.DEBUG if config.get("debug", False) else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting training for {game} on GPU {gpu_id}, run {run_id}")
        
        trainer_cls = get_trainer_class(config["trainer"])
        agent_cls = get_agent_class(config["agent"])
        logic_cls = get_logic_class(config["logic"])

        # Get action space
        env = EnvPoolAtariManager().create_env(game)
        game_config = config.copy()
        game_config["action_space"] = env.action_space.n
        game_config["game"] = game
        env.close()

        # Create trainer with game-specific config
        trainer = trainer_cls(
            game_config, logic_cls, agent_cls, atari_manager=EnvPoolAtariManager()
        )
        
        # Train
        trainer.train(game, config.get("episodes", 1000))
        
        logger.info(f"Successfully completed training for {game} on GPU {gpu_id}")
        return True
    except Exception as e:
        logger.error(
            f"Failed training {game} with {config['agent']}: {str(e)}",
            exc_info=True,
        )
        if config.get("debug", False):
            raise
        return False


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

        # Check if parallel mode is enabled
        if config.get("parallel", False):
            run_parallel_training(config, games, run_id)
        else:
            # Original sequential training
            for game in games:
                if not game:
                    continue

                # Get action space using EnvPool
                atari_manager = EnvPoolAtariManager()
                env = atari_manager.create_env(game)
                config["action_space"] = env.action_space.n

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


def run_parallel_training(config, configs_to_run):
    """Run training on multiple configurations in parallel across available workers
    
    Args:
        config: Base configuration with general settings
        configs_to_run: List of configurations to distribute across workers
        run_id_base: Base identifier for this training run
    """
    import logging
    import queue
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    logger = logging.getLogger(__name__)
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPUs detected! Falling back to sequential training on CPU.")
        for idx, train_config in enumerate(configs_to_run):
            train_game(train_config, train_config.get("game"), 0, f"{train_config['run_id']}")
        return
    
    # Determine max concurrent jobs
    max_jobs = min(config.get("max_jobs", num_gpus * 2), len(configs_to_run))
    logger.info(f"Running parallel training across {num_gpus} GPUs with max {max_jobs} concurrent jobs")
    
    # Create a queue of training configurations
    config_queue = queue.Queue()
    for idx, train_config in enumerate(configs_to_run):
        config_queue.put((idx, train_config))
    
    results = []
    
    # Use process pool for parallel execution
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        # Dictionary to keep track of running tasks
        running_tasks = {}
        completed_count = 0
        
        # Start initial batch of tasks
        for worker_id in range(min(max_jobs, config_queue.qsize())):
            if not config_queue.empty():
                idx, train_config = config_queue.get()
                gpu_id = worker_id % num_gpus  # Distribute across available GPUs
                
                # Copy config and set device
                job_config = train_config.copy()
                job_config['device'] = f"cuda:{gpu_id}"
                
                game = job_config.get("game")
                logger.info(f"Scheduling config {idx} ({game}) on GPU {gpu_id}")
                
                # Submit job
                future = executor.submit(
                    train_game,
                    job_config,
                    game,
                    gpu_id,
                    f"{job_config['run_id']}"
                )
                running_tasks[future] = (idx, game, gpu_id)
        
        # Process results as they complete and schedule new jobs
        while running_tasks:
            # Wait for a task to complete
            done, _ = concurrent.futures.wait(
                running_tasks.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for future in done:
                idx, game, gpu_id = running_tasks.pop(future)
                try:
                    result = future.result()
                    status = "Success" if result else "Failed"
                    logger.info(f"{status}: Config {idx} ({game}) on GPU {gpu_id}")
                    results.append((idx, game, status))
                    completed_count += 1
                except Exception as e:
                    logger.error(f"Exception in worker: {str(e)}")
                    results.append((idx, game, "Error"))
                    completed_count += 1
                
                # Schedule a new task if there are configs left
                if not config_queue.empty():
                    next_idx, next_config = config_queue.get()
                    next_game = next_config.get("game")
                    
                    # Copy config and set device
                    job_config = next_config.copy()
                    job_config['device'] = f"cuda:{gpu_id}"  # Reuse the GPU
                    
                    logger.info(f"Scheduling config {next_idx} ({next_game}) on GPU {gpu_id}")
                    
                    # Submit new job
                    new_future = executor.submit(
                        train_game,
                        job_config,
                        next_game,
                        gpu_id,
                        f"{job_config['run_id']}"
                    )
                    running_tasks[new_future] = (next_idx, next_game, gpu_id)
    
    logger.info(f"All training jobs completed. Total: {completed_count}")
    return results


def main():
    # On Linux, use 'spawn' method for better CUDA compatibility
    if sys.platform == 'linux':
        mp.set_start_method('spawn', force=True)
        
    from datetime import datetime
        
    args = parse_args()

    if args.config:
        # Use config file mode
        config_handler = ConfigHandler(args.config)
        configs = config_handler.generate_configs()
        total_configs = len(configs)
        
        # Base configuration for shared settings
        base_config = {
            "save_dir": args.save_dir,
            "log_dir": args.log_dir,
            "debug": args.debug,
            "max_jobs": args.max_jobs
        }
        
        # If parallel mode is enabled, run all configs in parallel
        if args.parallel:
            print(f"Running {total_configs} configurations in parallel mode")
            run_parallel_training(base_config, configs)
        else:
            # Run configs sequentially
            for idx, config in enumerate(configs):
                run_id = config['run_id']
                run_training(config, run_id=run_id)
                print(f"\n=== Completed run {idx+1}/{total_configs} ===\n")

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
            "parallel": args.parallel,
            "max_jobs": args.max_jobs,
        }
        os.makedirs(args.save_dir, exist_ok=True)
        run_training(config, run_id=config['run_id'])

if __name__ == "__main__":
    main()
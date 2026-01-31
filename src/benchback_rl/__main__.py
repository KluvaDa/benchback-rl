"""Main entry point for benchback_rl package.

Usage:
    python -m benchback_rl --run_yaml config.yaml      # Run single benchmark from YAML
    python -m benchback_rl --run_index 42              # Run benchmark at index 42
    python -m benchback_rl --n_index                   # Print total number of benchmarks
    python -m benchback_rl --run_all                   # Run all benchmarks in-process
"""

import argparse
import dataclasses
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.rl_common.benchmark import (
    run_ppo_benchmark,
    run_all_benchmarks,
    get_benchmark_count,
    get_benchmark_config,
)


def main() -> None:
    """Parse arguments and run appropriate benchmark command."""
    parser = argparse.ArgumentParser(
        description="PPO Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchback_rl --run_yaml configs/torch.yaml
  python -m benchback_rl --run_index 0
  python -m benchback_rl --n_index
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_yaml", type=str, metavar="PATH",
                       help="Run a single benchmark from a YAML config file")
    group.add_argument("--run_index", type=int, metavar="N",
                       help="Run the benchmark at index N (0-indexed)")
    group.add_argument("--n_index", action="store_true",
                       help="Print the total number of benchmark configurations")
    group.add_argument("--run_all", action="store_true",
                       help="Run all benchmarks in-process (use run_all_benchmarks.sh for isolation)")
    
    parser.add_argument("--notes", type=str, default="",
                        help="User notes to log in wandb")
    
    args = parser.parse_args()

    # Load environment variables from .env file at project root
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env", override=False)

    if args.n_index:
        print(get_benchmark_count())
    elif args.run_index is not None:
        config = get_benchmark_config(args.run_index)
        assert isinstance(config, PPOConfig)
        if args.notes:
            config = dataclasses.replace(config, notes_user=args.notes)
        run_ppo_benchmark(config)
    elif args.run_yaml:
        config = PPOConfig.from_yaml(args.run_yaml)
        if args.notes:
            config = dataclasses.replace(config, notes_user=args.notes)
        run_ppo_benchmark(config)
    elif args.run_all:
        run_all_benchmarks()


if __name__ == "__main__":
    main()
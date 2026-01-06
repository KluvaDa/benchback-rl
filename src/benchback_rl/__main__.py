"""Main entry point for benchback_rl package.

Usage:
    python -m benchback_rl config.yaml
"""

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.benchmarks.runner import run_ppo_benchmark


def main() -> None:
    """Load config from YAML file and run PPO benchmark."""
    if len(sys.argv) != 2:
        print("Usage: python -m benchback_rl <config.yaml>", file=sys.stderr)
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load environment variables from .env file at project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path, override=False)
    
    # Load YAML config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Create PPOConfig with YAML values overriding defaults
    config = PPOConfig(**config_dict)
    
    run_ppo_benchmark(config)


if __name__ == "__main__":
    main()
"""Main entry point for benchback_rl package.

Usage:
    python -m benchback_rl config.yaml
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.rl_common.benchmark import run_ppo_benchmark


def main() -> None:
    """Load config from YAML file and run PPO benchmark."""
    if len(sys.argv) != 2:
        print("Usage: python -m benchback_rl <config.yaml>", file=sys.stderr)
        sys.exit(1)
    
    # Load environment variables from .env file at project root
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root/".env", override=False)
    
    config = PPOConfig.from_yaml(sys.argv[1])
    run_ppo_benchmark(config)


if __name__ == "__main__":
    main()
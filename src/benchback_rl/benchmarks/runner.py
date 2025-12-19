"""Benchmark runner for PPO training."""

import os
from pathlib import Path

from dotenv import load_dotenv
import wandb

from benchback_rl.environment.torch_env import TorchEnv
from benchback_rl.rl_common.config import PPOConfig
from benchback_rl import rl_torch


def run_ppo_benchmark(
    config: PPOConfig) -> None:
    """Run a PPO training benchmark."""
    if config.use_wandb:
        # config is saved in the train method
        wandb.init(
            project=config.wandb_project,
        )

    if config.framework == "torch":
        env = TorchEnv(config.env_name, config.num_envs)

        agent = rl_torch.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_dim=config.hidden_dim,
        )
        torch_ppo = rl_torch.PPO(env=env, agent=agent, config=config)

        torch_ppo.train_from_scratch()

    elif config.framework == "linen":
        raise NotImplementedError()
    elif config.framework == "nnx":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")

    # Finish WandB run if active
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    # Load environment variables from .env file at project root
    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path, override=False)

    config = PPOConfig(
        framework="torch",
        env_name="Acrobot-v1",
        num_envs=32,
        num_steps=256,
        num_minibatches=8,
        update_epochs=10,
        num_iterations=100,
        use_wandb=True,
        sync_for_timing=True
    )

    run_ppo_benchmark(config)

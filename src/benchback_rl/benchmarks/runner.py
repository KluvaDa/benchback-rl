"""Benchmark runner for PPO training."""

import os
from pathlib import Path

from dotenv import load_dotenv
import wandb

from benchback_rl.environment.torch_env import TorchEnv
from benchback_rl.rl_torch.models import DefaultActorCritic
from benchback_rl.rl_torch.ppo import PPO, PPOHyperparameters


def run_ppo_benchmark(
    env_name: str = "Acrobot-v1",
    num_envs: int = 64,
    num_steps: int = 1028,
    num_iterations: int = 64,
    num_minibatches: int = 64,
    update_epochs: int = 16,
    hidden_dim: int = 64,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "benchback_rl",
) -> None:
    """Run a PPO training benchmark.

    Args:
        env_name: Name of the gymnax environment.
        num_envs: Number of parallel environments.
        num_steps: Number of steps per rollout per environment.
        num_iterations: Total number of training iterations.
        num_minibatches: Number of minibatches per update.
        update_epochs: Number of epochs per update.
        hidden_dim: Hidden dimension for the actor-critic network.
        seed: Random seed for reproducibility.
        use_wandb: Whether to enable WandB logging.
        wandb_project: WandB project name (only used if use_wandb=True).
    """
    # Create environment
    env = TorchEnv(env_name, num_envs)

    # Create agent
    agent = DefaultActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.num_actions,
        hidden_dim=hidden_dim,
    )

    # Create hyperparameters
    hparams = PPOHyperparameters(
        num_envs=num_envs,
        obs_dim=env.obs_dim,
        num_steps=num_steps,
        num_iterations=num_iterations,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        seed=seed,
    )

    # Initialize WandB if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={
                "env_name": env_name,
                "hidden_dim": hidden_dim,
            },
        )

    # Create PPO and train
    ppo = PPO(env, agent, hparams)
    ppo.train()

    # Finish WandB run if active
    if wandb.run is not None:
        wandb.finish()

    print(f"\n✓ Benchmark completed: {env_name} with {num_iterations} iterations")


if __name__ == "__main__":
    # Load environment variables from .env file at project root
    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path, override=False)

    run_ppo_benchmark()

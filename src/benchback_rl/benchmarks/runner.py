"""Benchmark runner for PPO training."""

from typing import Any, cast

import gymnax
from gymnax.environments.spaces import Discrete
import torch
import wandb

from benchback_rl.environment.torch_env import TorchEnv
from benchback_rl.rl_common.config import PPOConfig
from benchback_rl import rl_torch
from benchback_rl import rl_linen
from benchback_rl.rl_linen.models import ModelParams
from benchback_rl.rl_linen.ppo.rollout import EnvParamsVmapped

# Type alias for gymnax environment (to work around complex union types)
Env = Any


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

        agent: rl_torch.ActorCritic = rl_torch.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_dim=config.hidden_dim,
        )
        # torch.compile provides ~2x speedup for the model forward pass
        agent = torch.compile(agent)  # type: ignore[assignment]
        torch_ppo = rl_torch.PPO(env=env, agent=agent, config=config)

        torch_ppo.train_from_scratch()

    elif config.framework == "linen":
        # Create gymnax environment directly for JAX/Flax
        env_: Env
        env_params: EnvParamsVmapped
        env_, env_params = gymnax.make(config.env_name)
        
        # Get action dimension from environment
        action_space = env_.action_space(env_params)
        action_dim = cast(Discrete, action_space).n
        
        # Create model and initialize parameters
        model = rl_linen.DefaultActorCritic(
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
        )
        
        # Initialize model parameters with dummy input
        import jax
        import jax.numpy as jnp
        obs_space = env_.observation_space(env_params)
        obs_dim: int = obs_space.shape[0]
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_rng = jax.random.PRNGKey(0)
        model_params: ModelParams = dict(model.init(dummy_rng, dummy_obs))
        
        # Create PPO trainer
        linen_ppo = rl_linen.PPO(
            env=env_,
            env_params=env_params,
            model=model,
            model_params=model_params,
            config=config,
        )
        
        linen_ppo.train_from_scratch()
    elif config.framework == "nnx":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")

    # Finish WandB run if active
    if wandb.run is not None:
        wandb.finish()

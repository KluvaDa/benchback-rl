"""Benchmark runner for PPO training."""

import time
from typing import Any, cast, Literal

import gymnax
from gymnax.environments.spaces import Discrete
import wandb

from benchback_rl.rl_common.config import PPOConfig

# Type alias for gymnax environment (to work around complex union types)
Env = Any


def create_config(model: Literal["CartPole-v1", "Acrobot-v1!", "MountainCar-v0"]):
    """ Creates the PPOConfig for the specific benchmark experiments that we will run.
    Relies on the defaults from PPOConfig for the unspecified parameters.
    """


def run_all_benchmarks() -> None:
    """Run all predefined PPO benchmarks."""
    environments = (
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "DiscountingChain-bsuite",
        "MemoryChain-bsuite",
        "UmbrellaChain-bsuite",
        "DeepSea-bsuite",
        "Catch-bsuite",
        "SimpleBandit-bsuite",
        "BernoulliBandit-misc",
        "GaussianBandit-misc"
    )
    models_hidden_sizes_lookup = {
        "small": (64, 64),
        "medium": (256, 256, 256),
        "large": (1024, 1024, 1024, 1024, 1024, 1024),
    }

    # warmup
    for _ in range(2):
        config_template = lambda framework, compile: PPOConfig(
            framework=framework,
            compile=compile,
            env_name="Acrobot-v1",
            hidden_sizes=models_hidden_sizes_lookup["small"],
            sync_for_timing=False,
            use_wandb=False,
            notes="warmup"
        )
        run_ppo_benchmark(config_template("torch", "torch.compile"))
        run_ppo_benchmark(config_template("linen", "jax.jit"))
        run_ppo_benchmark(config_template("nnx", "nnx.cached_partial"))

    # experiment 1: various envs
    for _ in range(2):
        for env_name in environments:
            config_template = lambda framework, compile: PPOConfig(
                framework=framework,
                compile=compile,
                env_name=env_name,
                hidden_sizes=models_hidden_sizes_lookup["small"],
                sync_for_timing=False,
                use_wandb=True,
                notes="experiment 1: various envs"
            )
            run_ppo_benchmark(config_template("torch", "torch.compile"))
            run_ppo_benchmark(config_template("linen", "jax.jit"))
            run_ppo_benchmark(config_template("nnx", "nnx.cached_partial"))
    
    # experiment 2: various model sizes on Acrobot-v1
    for _ in range(4):
        for model_size in ["small", "medium", "large"]:
            config_template = lambda framework, compile: PPOConfig(
                framework=framework,
                compile=compile,
                env_name="Acrobot-v1",
                hidden_sizes=models_hidden_sizes_lookup[model_size],
                sync_for_timing=False,
                use_wandb=True,
                notes="experiment 2: various model sizes"
            )
            run_ppo_benchmark(config_template("torch", "torch.compile"))
            run_ppo_benchmark(config_template("linen", "jax.jit"))
            run_ppo_benchmark(config_template("nnx", "nnx.cached_partial"))
    
    # experiment 3: various compilation settings on Acrobot-v1 and synced timing
    for _ in range(4):
        for model_size in ["small", "medium", "large"]:
            config_template = lambda framework, compile: PPOConfig(
                framework=framework,
                compile=compile,
                env_name="Acrobot-v1",
                hidden_sizes=models_hidden_sizes_lookup[model_size],
                sync_for_timing=True,
                use_wandb=True,
                notes="experiment 3: various compilation settings with synced timing"
            )
            run_ppo_benchmark(config_template("torch", "torch.compile"))
            run_ppo_benchmark(config_template("linen", "jax.jit"))
            run_ppo_benchmark(config_template("nnx", "nnx.cached_partial"))
            run_ppo_benchmark(config_template("torch", "none"))
            run_ppo_benchmark(config_template("linen", "none"))
            run_ppo_benchmark(config_template("nnx", "none"))
            run_ppo_benchmark(config_template("torch", "torch.compile"))
            run_ppo_benchmark(config_template("torch", "nnx.jit"))


def run_ppo_benchmark(
    config: PPOConfig) -> None:
    """Run a PPO training benchmark."""
    start_time = time.perf_counter()  # Capture at very beginning for initial overhead
    
    if config.use_wandb:
        # config is saved in the train method
        wandb.init(
            project=config.wandb_project,
        )

    if config.framework == "torch":
        import torch
        from benchback_rl import rl_torch
        env = rl_torch.TorchEnv(config.env_name, config.num_envs)

        agent: rl_torch.ActorCritic = rl_torch.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_sizes=config.hidden_sizes,
        )
        # torch.compile provides ~2x speedup for the model forward pass
        agent = torch.compile(agent)  # type: ignore[assignment]
        torch_ppo = rl_torch.PPO(env=env, agent=agent, config=config, start_time=start_time)

        torch_ppo.train_from_scratch()

    elif config.framework == "linen":
        from benchback_rl import rl_linen
        from benchback_rl.rl_linen.models import ModelParams
        from benchback_rl.rl_linen.ppo.rollout import EnvParamsVmapped
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
            hidden_sizes=config.hidden_sizes,
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
            start_time=start_time,
        )
        
        linen_ppo.train_from_scratch()
    elif config.framework == "nnx":
        from benchback_rl import rl_nnx
        from flax import nnx
        import jax

        # Create rngs with all required streams from a single seed
        seed = config.seed if config.seed is not None else 0
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)
        rngs = nnx.Rngs(
            params=keys[0],     # used in model initialization (nnx.Linear)
            env=keys[1],        # used in env.reset and env.step
            action=keys[2],     # used in model action sampling
            minibatch=keys[3],  # used in PPO minibatch shuffling
        )
        
        # Create NnxVecEnv which wraps gymnax environment
        env = rl_nnx.NnxVecEnv(
            env_name=config.env_name,
            num_envs=config.num_envs,
            rngs=rngs,
        )
        
        # Create model with NNX
        model = rl_nnx.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_sizes=config.hidden_sizes,
            rngs=rngs,
        )
        
        # Create PPO trainer
        nnx_ppo = rl_nnx.PPO(
            config=config,
            env=env,
            model=model,
            rngs=rngs,
            start_time=start_time,
        )
        
        nnx_ppo.train_from_scratch()
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")

    # Finish WandB run if active
    if wandb.run is not None:
        wandb.finish()

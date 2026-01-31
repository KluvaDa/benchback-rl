"""Benchmark runner for PPO training."""

import gc
import math
import os
import time
from collections.abc import Sequence
from typing import Any, cast

import gymnax
import psutil
from gymnax.environments.spaces import Discrete
import jax
import jax.numpy as jnp
from flax import nnx
import torch
import wandb

from benchback_rl import rl_linen, rl_nnx, rl_torch
from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.rl_linen.models import ModelParams
from benchback_rl.rl_linen.ppo.rollout import EnvParamsVmapped

# Type alias for gymnax environment (to work around complex union types)
Env = Any


def _cleanup_memory() -> None:
    """Clean up memory after a benchmark run to prevent accumulation.

    OBSERVED ISSUES (from memory_test.py isolation tests):
    - NNX leaks JAX GPU VRAM (0 -> 0.62 GB over 15 runs without cleanup)
    - Linen does NOT leak JAX GPU VRAM (stays at 0.00 GB)
    - All frameworks (torch, linen, nnx) accumulate RAM (~0.7-1.0 GB growth)

    CLEANUP EFFECTIVENESS (from diagnose_*.log):
    - jax.clear_caches() clears NNX VRAM leak: reduces growth from +0.24 GB to +0.12 GB
    - gc.collect() does NOT help JAX VRAM (0.24 GB with or without)
    - gc.collect() removes RAM accumulation across all frameworks
      (unknown if RAM accumulation causes crashes)
    - ~0.1 GB residual VRAM persists with NNX even after cleanup

    SUSPICIONS:
    - NNX cached_partial does not get cleared with jax.clear_caches() - it may still accumulate ram or vram

    We keep all cleanup calls for safety across all frameworks.
    """
    # gc.collect() - removes RAM accumulation across all frameworks
    gc.collect()
    
    # jax.clear_caches() - clears NNX VRAM leak (linen doesn't leak)
    jax.clear_caches()
    
    # Clear PyTorch's CUDA memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Clear torch.compile caches
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    
    gc.collect()


def _log_memory(label: str) -> dict[str, float]:
    """Log current memory usage for debugging.
    
    Args:
        label: A descriptive label for this memory snapshot (e.g., "before_run", "after_run")
    
    Returns:
        Dictionary of memory metrics that can be logged to wandb.
    """
    metrics: dict[str, float] = {}
    
    # RAM usage
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    metrics[f"memory/{label}/ram_gb"] = ram_gb
    print(f"[Memory {label}] RAM: {ram_gb:.2f} GB")
    
    # PyTorch GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(i) / (1024 ** 3)
            metrics[f"memory/{label}/torch_gpu{i}_allocated_gb"] = allocated_gb
            metrics[f"memory/{label}/torch_gpu{i}_reserved_gb"] = reserved_gb
            print(f"[Memory {label}] PyTorch GPU {i}: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")
    
    # JAX GPU memory
    try:
        for device in jax.devices():
            if device.platform == "gpu":
                stats = device.memory_stats()
                if stats:
                    bytes_in_use = stats.get("bytes_in_use", 0)
                    peak_bytes = stats.get("peak_bytes_in_use", 0)
                    jax_gb = bytes_in_use / (1024 ** 3)
                    jax_peak_gb = peak_bytes / (1024 ** 3)
                    metrics[f"memory/{label}/jax_{device.id}_gb"] = jax_gb
                    metrics[f"memory/{label}/jax_{device.id}_peak_gb"] = jax_peak_gb
                    print(f"[Memory {label}] JAX {device}: {jax_gb:.2f} GB in use, {jax_peak_gb:.2f} GB peak")
    except Exception as e:
        print(f"[Memory {label}] Could not get JAX memory stats: {e}")
    
    return metrics


def _pick_from_loops(index: int, *sequences: Sequence) -> tuple:
    """Decode a flat index into elements from nested for-loops over sequences."""
    result = []
    for seq in reversed(sequences):
        result.append(seq[index % len(seq)])
        index //= len(seq)
    return tuple(reversed(result))


def get_benchmark_config(index: int | None, repeats: int | None = None) -> PPOConfig | int:
    """Get benchmark config at index, or total count if index is None.
    
    Args:
        index: 0-based index of the benchmark to retrieve, or None to get count.
        repeats: Optional override for the number of repeats per experiment.
                 If None, uses the default repeats defined for each experiment.
        
    Returns:
        PPOConfig at the given index, or total count if index is None.
        
    Raises:
        IndexError: If index is out of range.
    """
    # Model size definitions
    model_sizes_2_hidden_sizes = {
        "small": (64, 64),
        "medium": (256, 256, 256),
        "large": (1024, 1024, 1024, 1024, 1024, 1024),
    }

    # Note: Only 1D observation envs are supported (no images)
    # Excluded: DeepSea-bsuite (8,8), Catch-bsuite (10,5), SimpleBandit-bsuite (1,1)
    
    counter = 0
    
    # =========================================================================
    # Warmup
    # =========================================================================
    # sequences to iterate over like in a nested for-loop
    repeats_range = range(repeats if repeats is not None else 1)
    frameworks = (("torch", "torch.compile"), ("linen", "jax.jit"), ("nnx", "nnx.cached_partial"))
    
    experiment_size = len(repeats_range) * len(frameworks)
    if index is not None and index < counter + experiment_size:
        _, (framework, compile) = _pick_from_loops(index - counter, repeats_range, frameworks)
        return PPOConfig(
            framework=framework,  # type: ignore[arg-type]
            compile=compile,  # type: ignore[arg-type]
            env_name="Acrobot-v1",
            hidden_sizes=model_sizes_2_hidden_sizes["small"],
            sync_for_timing=False,
            use_wandb=False,
            notes_config="warmup",
        )
    else:
        counter += experiment_size
    
    # =========================================================================
    # Experiment 1: various envs
    # =========================================================================
    repeats_range = range(repeats if repeats is not None else 2)
    environments = ("CartPole-v1",
                    "Acrobot-v1",
                    "MountainCar-v0",
                    "DiscountingChain-bsuite",
                    "MemoryChain-bsuite",
                    "UmbrellaChain-bsuite",
                    "BernoulliBandit-misc",
                    "GaussianBandit-misc",)
    frameworks = (("torch", "torch.compile"),
                  ("linen", "jax.jit"),
                  ("nnx", "nnx.cached_partial"),)
    
    experiment_size = len(repeats_range) * len(environments) * len(frameworks)
    if index is not None and index < counter + experiment_size:
        _, env_name, (framework, compile) = _pick_from_loops(index - counter, repeats_range, environments, frameworks)
        return PPOConfig(
            framework=framework,  # type: ignore[arg-type]
            compile=compile,  # type: ignore[arg-type]
            env_name=env_name,  # type: ignore[arg-type]
            hidden_sizes=model_sizes_2_hidden_sizes["small"],
            sync_for_timing=False,
            use_wandb=True,
            notes_config="experiment 1: various envs",
        )
    else:
        counter += experiment_size
    
    # =========================================================================
    # Experiment 2: various model sizes
    # =========================================================================
    repeats_range = range(repeats if repeats is not None else 4)
    model_sizes = ("large", "medium", "small")
    frameworks = (("torch", "torch.compile"), 
                  ("linen", "jax.jit"),
                  ("nnx", "nnx.cached_partial"),)
    
    experiment_size = len(repeats_range) * len(model_sizes) * len(frameworks)
    if index is not None and index < counter + experiment_size:
        _, model_size, (framework, compile) = _pick_from_loops(index - counter, repeats_range, model_sizes, frameworks)
        return PPOConfig(
            framework=framework,  # type: ignore[arg-type]
            compile=compile,  # type: ignore[arg-type]
            env_name="Acrobot-v1",
            hidden_sizes=model_sizes_2_hidden_sizes[model_size],
            sync_for_timing=False,
            use_wandb=True,
            notes_config="experiment 2: various model sizes",
        )
    else:
        counter += experiment_size
    
    # =========================================================================
    # Experiment 3: various compilation settings with synced timing
    # =========================================================================
    repeats_range = range(repeats if repeats is not None else 4)
    model_sizes = ("small", "medium", "large")
    frameworks = (
        # fully compiled
        ("torch", "torch.compile"), ("linen", "jax.jit"), ("nnx", "nnx.cached_partial"),
        # no compilation  
        ("torch", "none"), ("linen", "none"), ("nnx", "none"),
        # partial compilation
        ("nnx", "nnx.jit"), ("torch", "torch.nocompile/env.jit"),
    )
    
    experiment_size = len(repeats_range) * len(model_sizes) * len(frameworks)
    if index is not None and index < counter + experiment_size:
        _, model_size, (framework, compile) = _pick_from_loops(index - counter, repeats_range, model_sizes, frameworks)
        return PPOConfig(
            framework=framework,
            compile=compile,  # type: ignore[arg-type]
            env_name="Acrobot-v1",
            hidden_sizes=model_sizes_2_hidden_sizes[model_size],
            sync_for_timing=True,
            use_wandb=True,
            notes_config="experiment 3: various compilation settings with synced timing",
        )
    else:
        counter += experiment_size
    
    # =========================================================================
    # End - return count or raise error
    # =========================================================================
    if index is None:
        return counter
    raise IndexError(f"Benchmark index {index} out of range [0, {counter - 1}]")


def get_benchmark_count(repeats: int | None = None) -> int:
    """Get the total number of benchmark configurations.
    
    Args:
        repeats: Optional override for the number of repeats per experiment.
    """
    result = get_benchmark_config(None, repeats=repeats)
    assert isinstance(result, int)
    return result


def run_all_benchmarks(repeats: int | None = None) -> None:
    """Run all predefined PPO benchmarks sequentially in-process.
    
    Args:
        repeats: Optional override for the number of repeats per experiment.
    
    Note: For better isolation between runs, use run_all_benchmarks.sh
    which runs each benchmark in a separate Docker container.
    """
    total = get_benchmark_count(repeats=repeats)
    for i in range(total):
        config = get_benchmark_config(i, repeats=repeats)
        assert isinstance(config, PPOConfig)
        run_ppo_benchmark(config)


def run_ppo_benchmark(
    config: PPOConfig) -> None:
    """Run a PPO training benchmark."""
    start_time = time.perf_counter()  # Capture at very beginning for initial overhead
    
    # Log memory before run
    pre_metrics = _log_memory("before_run")
    
    if config.use_wandb:
        # config is saved in the train method
        wandb.init(
            project=config.wandb_project,
        )
        # Log pre-run memory to wandb
        wandb.log(pre_metrics, step=0)

    if config.framework == "torch":
        env = rl_torch.TorchEnv(config.env_name, config.num_envs, jit=(config.compile != "none"), seed=config.seed)
        agent: rl_torch.ActorCritic = rl_torch.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_sizes=config.hidden_sizes,
        )
        # torch.compile provides ~2x speedup for the model forward pass
        if config.compile == "torch.compile":
            agent = torch.compile(agent)  # type: ignore[assignment]
        torch_ppo = rl_torch.PPO(env=env, agent=agent, config=config, start_time=start_time)
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
            hidden_sizes=config.hidden_sizes,
        )
        
        # Initialize model parameters with dummy input
        obs_space = env_.observation_space(env_params)
        obs_dim: int = math.prod(obs_space.shape)
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
        # Create rngs with all required streams from a single seed
        # Use provided seed or generate one based on time (consistent with linen/torch)
        seed = config.seed if config.seed is not None else int(time.time_ns())
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

    # Log memory after run (before cleanup)
    post_metrics = _log_memory("after_run")
    if wandb.run is not None:
        wandb.log(post_metrics)

    # Finish WandB run if active
    if wandb.run is not None:
        wandb.finish()

    # Clean up to prevent memory accumulation across benchmark runs
    _cleanup_memory()
    
    # Log memory after cleanup to verify cleanup effectiveness
    _log_memory("after_cleanup")

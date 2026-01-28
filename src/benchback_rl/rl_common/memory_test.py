"""Memory stress test for debugging memory leaks across benchmark runs.

This module provides tools to test whether memory accumulates across
sequential benchmark runs with different model sizes and frameworks.

Usage:
    # Test WITHOUT cleanup (to see memory growth)
    python -c "from benchback_rl.rl_common.memory_test import run_memory_test; run_memory_test(cleanup_enabled=False)"
    
    # Test WITH cleanup (to verify it helps)
    python -c "from benchback_rl.rl_common.memory_test import run_memory_test; run_memory_test(cleanup_enabled=True)"
"""

import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Literal, cast

import gymnax
from gymnax.environments.spaces import Discrete
import jax
import jax.numpy as jnp
from flax import nnx
import psutil
import torch

from benchback_rl import rl_linen, rl_nnx, rl_torch
from benchback_rl.rl_common.config import PPOConfig
from benchback_rl.rl_linen.models import ModelParams
from benchback_rl.rl_linen.ppo.rollout import EnvParamsVmapped

# Type alias for gymnax environment
Env = Any


def _cleanup_memory() -> None:
    """Clean up memory after a benchmark run to prevent accumulation."""
    gc.collect()
    jax.clear_caches()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    gc.collect()


def _get_memory_snapshot() -> dict[str, float]:
    """Get current memory usage as a simple dict (no printing)."""
    snapshot: dict[str, float] = {}
    
    # RAM
    process = psutil.Process(os.getpid())
    snapshot["ram_gb"] = process.memory_info().rss / (1024 ** 3)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        snapshot["torch_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024 ** 3)
        snapshot["torch_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024 ** 3)
    
    # JAX GPU
    try:
        for device in jax.devices():
            if device.platform == "gpu":
                stats = device.memory_stats()
                if stats:
                    snapshot["jax_in_use_gb"] = stats.get("bytes_in_use", 0) / (1024 ** 3)
                    snapshot["jax_peak_gb"] = stats.get("peak_bytes_in_use", 0) / (1024 ** 3)
                    break
    except Exception:
        pass
    
    return snapshot


def _run_ppo_benchmark_no_wandb(config: PPOConfig) -> None:
    """Run a PPO benchmark without any wandb logging.
    
    Stripped down version of run_ppo_benchmark for memory testing.
    """
    start_time = time.perf_counter()

    if config.framework == "torch":
        env = rl_torch.TorchEnv(config.env_name, config.num_envs, jit=(config.compile != "none"), seed=config.seed)
        agent: rl_torch.ActorCritic = rl_torch.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_sizes=config.hidden_sizes,
        )
        if config.compile == "torch.compile":
            agent = torch.compile(agent)  # type: ignore[assignment]
        torch_ppo = rl_torch.PPO(env=env, agent=agent, config=config, start_time=start_time)
        torch_ppo.train_from_scratch()

    elif config.framework == "linen":
        env_: Env
        env_params: EnvParamsVmapped
        env_, env_params = gymnax.make(config.env_name)
        action_space = env_.action_space(env_params)
        action_dim = cast(Discrete, action_space).n
        model = rl_linen.DefaultActorCritic(
            action_dim=action_dim,
            hidden_sizes=config.hidden_sizes,
        )
        obs_space = env_.observation_space(env_params)
        obs_dim: int = obs_space.shape[0]
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_rng = jax.random.PRNGKey(0)
        model_params: ModelParams = dict(model.init(dummy_rng, dummy_obs))
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
        # Use provided seed or generate one based on time (consistent with linen/torch)
        seed = config.seed if config.seed is not None else int(time.time_ns())
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)
        rngs = nnx.Rngs(
            params=keys[0],
            env=keys[1],
            action=keys[2],
            minibatch=keys[3],
        )
        env = rl_nnx.NnxVecEnv(
            env_name=config.env_name,
            num_envs=config.num_envs,
            rngs=rngs,
        )
        model = rl_nnx.DefaultActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.num_actions,
            hidden_sizes=config.hidden_sizes,
            rngs=rngs,
        )
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


def run_memory_test(cleanup_enabled: bool = True) -> None:
    """Run a minimal memory stress test to verify cleanup effectiveness.
    
    Tests all frameworks × model sizes × repeats with num_iterations=1.
    Prints memory stats and JSON results to console.
    
    Args:
        cleanup_enabled: If True, run _cleanup_memory() after each benchmark.
    """
    models_hidden_sizes = {
        "small": (64, 64),
        "medium": (256, 256, 256),
        "large": (1024, 1024, 1024, 1024, 1024, 1024),
    }
    frameworks_and_compile: list[tuple[Literal["torch", "linen", "nnx"], Literal["torch.compile", "jax.jit", "nnx.cached_partial"]]] = [
        ("torch", "torch.compile"),
        ("linen", "jax.jit"),
        ("nnx", "nnx.cached_partial"),
    ]
    num_repeats = 3
    
    results: list[dict[str, Any]] = []
    
    print("=" * 80)
    print(f"MEMORY TEST - cleanup_enabled={cleanup_enabled}")
    print("=" * 80)
    print(f"{'#':<3} {'Framework':<8} {'Size':<8} {'Rep':<4} {'RAM GB':<8} {'Torch Alloc':<12} {'JAX Use':<10}")
    print("-" * 80)
    
    step = 0
    for repeat in range(num_repeats):
        for model_size in ["small", "medium", "large"]:
            for framework, compile_mode in frameworks_and_compile:
                # Get memory before
                mem_before = _get_memory_snapshot()
                
                # Create config with wandb disabled
                config = PPOConfig(
                    framework=framework,
                    compile=compile_mode,
                    env_name="Acrobot-v1",
                    hidden_sizes=models_hidden_sizes[model_size],
                    num_iterations=1,  # Minimal iterations
                    use_wandb=False,   # No wandb logging
                    sync_for_timing=False,
                )
                
                # Run the benchmark
                try:
                    _run_ppo_benchmark_no_wandb(config)
                except Exception as e:
                    print(f"[ERROR] Benchmark failed: {e}")
                    continue
                
                # Cleanup if enabled
                if cleanup_enabled:
                    _cleanup_memory()
                
                # Get memory after
                mem_after = _get_memory_snapshot()
                
                # Record results
                result = {
                    "step": step,
                    "repeat": repeat,
                    "framework": framework,
                    "model_size": model_size,
                    "cleanup_enabled": cleanup_enabled,
                    "mem_before": mem_before,
                    "mem_after": mem_after,
                }
                results.append(result)
                
                # Print summary line
                print(f"{step:<3} {framework:<8} {model_size:<8} {repeat:<4} "
                      f"{mem_after.get('ram_gb', 0):<8.2f} "
                      f"{mem_after.get('torch_allocated_gb', 0):<12.2f} "
                      f"{mem_after.get('jax_in_use_gb', 0):<10.2f}")
                
                step += 1
    
    print("=" * 80)
    print(f"Completed {step} benchmarks")
    
    # Print JSON to console for easy capture
    print("\n" + "=" * 80)
    print("JSON RESULTS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))


def run_framework_isolation_test(framework: Literal["torch", "linen", "nnx"], num_repeats: int = 5) -> None:
    """Run memory test for a single framework to isolate memory behavior.
    
    Cycles through model sizes multiple times WITHOUT cleanup to see
    how much memory each framework accumulates on its own.
    
    Args:
        framework: Which framework to test ("torch", "linen", or "nnx")
        num_repeats: Number of cycles through all model sizes
    """
    models_hidden_sizes = {
        "small": (64, 64),
        "medium": (256, 256, 256),
        "large": (1024, 1024, 1024, 1024, 1024, 1024),
    }
    compile_modes = {
        "torch": "torch.compile",
        "linen": "jax.jit",
        "nnx": "nnx.cached_partial",
    }
    compile_mode = compile_modes[framework]
    
    results: list[dict[str, Any]] = []
    
    print("=" * 80)
    print(f"FRAMEWORK ISOLATION TEST - {framework} (no cleanup)")
    print("=" * 80)
    print(f"{'#':<3} {'Size':<8} {'Rep':<4} {'RAM GB':<8} {'Torch Alloc':<12} {'JAX Use':<10} {'JAX Peak':<10}")
    print("-" * 80)
    
    step = 0
    for repeat in range(num_repeats):
        for model_size in ["small", "medium", "large"]:
            mem_before = _get_memory_snapshot()
            
            config = PPOConfig(
                framework=framework,
                compile=compile_mode,  # type: ignore[arg-type]
                env_name="Acrobot-v1",
                hidden_sizes=models_hidden_sizes[model_size],
                num_iterations=1,
                use_wandb=False,
                sync_for_timing=False,
            )
            
            try:
                _run_ppo_benchmark_no_wandb(config)
            except Exception as e:
                print(f"[ERROR] Benchmark failed: {e}")
                continue
            
            # NO cleanup - we want to see accumulation
            
            mem_after = _get_memory_snapshot()
            
            result = {
                "step": step,
                "repeat": repeat,
                "framework": framework,
                "model_size": model_size,
                "mem_before": mem_before,
                "mem_after": mem_after,
            }
            results.append(result)
            
            print(f"{step:<3} {model_size:<8} {repeat:<4} "
                  f"{mem_after.get('ram_gb', 0):<8.2f} "
                  f"{mem_after.get('torch_allocated_gb', 0):<12.2f} "
                  f"{mem_after.get('jax_in_use_gb', 0):<10.2f} "
                  f"{mem_after.get('jax_peak_gb', 0):<10.2f}")
            
            step += 1
    
    print("=" * 80)
    print(f"Completed {step} benchmarks for {framework}")
    
    # Summary
    if results:
        start_ram = results[0]["mem_before"]["ram_gb"]
        end_ram = results[-1]["mem_after"]["ram_gb"]
        start_jax = results[0]["mem_before"].get("jax_in_use_gb", 0)
        end_jax = results[-1]["mem_after"].get("jax_in_use_gb", 0)
        print(f"\nRAM growth: {start_ram:.2f} GB -> {end_ram:.2f} GB (+{end_ram - start_ram:.2f} GB)")
        print(f"JAX GPU growth: {start_jax:.2f} GB -> {end_jax:.2f} GB (+{end_jax - start_jax:.2f} GB)")
    
    print("\n" + "=" * 80)
    print("JSON RESULTS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))


def run_cleanup_diagnosis_test(cleanup_mode: Literal["none", "gc", "jax", "both"]) -> None:
    """Diagnose which cleanup command actually clears JAX VRAM.
    
    Each mode should be run in a SEPARATE process (fresh container) to isolate effects.
    
    Args:
        cleanup_mode: Which cleanup to apply after each benchmark:
            - "none": No cleanup (baseline)
            - "gc": gc.collect() only
            - "jax": jax.clear_caches() only  
            - "both": gc.collect() + jax.clear_caches()
    """
    models_hidden_sizes = {
        "small": (64, 64),
        "medium": (256, 256, 256),
        "large": (1024, 1024, 1024, 1024, 1024, 1024),
    }
    
    print("=" * 80)
    print(f"CLEANUP DIAGNOSIS TEST - Mode: {cleanup_mode.upper()}")
    print("=" * 80)
    print("NOTE: Run each mode in a separate container for valid comparison!")
    print("=" * 80)
    
    def run_nnx_benchmark(size_name: str) -> None:
        hidden_sizes = models_hidden_sizes[size_name]
        config = PPOConfig(
            env_name="CartPole-v1",
            num_iterations=1,
            num_envs=16,
            num_steps=64,
            num_minibatches=4,
            update_epochs=1,
            framework="nnx",
            compile="nnx.cached_partial",
            hidden_sizes=hidden_sizes,
        )
        _run_ppo_benchmark_no_wandb(config)
    
    def apply_cleanup(mode: str) -> None:
        if mode == "gc":
            gc.collect()
        elif mode == "jax":
            jax.clear_caches()
        elif mode == "both":
            gc.collect()
            jax.clear_caches()
        # "none" does nothing
    
    def get_jax_vram() -> float:
        """Get JAX GPU in-use memory in GB."""
        try:
            for device in jax.devices():
                if device.platform == "gpu":
                    stats = device.memory_stats()
                    if stats:
                        return stats.get("bytes_in_use", 0) / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    
    results: list[dict[str, Any]] = []
    sizes = ["small", "medium", "large", "small", "medium", "large"]  # 2 cycles
    
    print(f"\n{'#':<4} {'Size':<8} {'Before':<10} {'After Run':<12} {'After Cleanup':<14}")
    print("-" * 60)
    
    for i, size in enumerate(sizes):
        before = get_jax_vram()
        run_nnx_benchmark(size)
        after_run = get_jax_vram()
        apply_cleanup(cleanup_mode)
        after_cleanup = get_jax_vram()
        
        print(f"{i:<4} {size:<8} {before:<10.3f} {after_run:<12.3f} {after_cleanup:<14.3f}")
        results.append({
            "cleanup_mode": cleanup_mode,
            "step": i,
            "size": size,
            "before": before,
            "after_run": after_run,
            "after_cleanup": after_cleanup,
        })
    
    start_vram = results[0]["before"]
    end_vram = results[-1]["after_cleanup"]
    
    print("-" * 60)
    print(f"VRAM growth ({cleanup_mode}): {start_vram:.3f} GB -> {end_vram:.3f} GB (+{end_vram - start_vram:.3f} GB)")
    
    print("\n" + "=" * 80)
    print("JSON RESULTS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    # Run via docker-compose (use tee to save output to file):
    #   With cleanup (default):
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test 2>&1 | tee memory_test_with_cleanup.log
    #   Without cleanup (to see memory growth):
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --no-cleanup 2>&1 | tee memory_test_no_cleanup.log
    #   Single framework isolation (to identify which framework leaks):
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --framework torch 2>&1 | tee memory_test_torch.log
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --framework linen 2>&1 | tee memory_test_linen.log
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --framework nnx 2>&1 | tee memory_test_nnx.log
    #   Cleanup diagnosis - RUN EACH IN A SEPARATE CONTAINER (fresh process):
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --diagnose none 2>&1 | tee diagnose_none.log
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --diagnose gc 2>&1 | tee diagnose_gc.log
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --diagnose jax 2>&1 | tee diagnose_jax.log
    #     docker compose -f setup/docker/docker-compose.run.yml run --build --rm run python -m benchback_rl.rl_common.memory_test --diagnose both 2>&1 | tee diagnose_both.log
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--no-cleanup":
        run_memory_test(cleanup_enabled=False)
    elif len(sys.argv) > 2 and sys.argv[1] == "--diagnose":
        mode = sys.argv[2]
        if mode not in ("none", "gc", "jax", "both"):
            print(f"Unknown diagnose mode: {mode}. Use 'none', 'gc', 'jax', or 'both'.")
            sys.exit(1)
        run_cleanup_diagnosis_test(mode)  # type: ignore[arg-type]
    elif len(sys.argv) > 2 and sys.argv[1] == "--framework":
        framework = sys.argv[2]
        if framework not in ("torch", "linen", "nnx"):
            print(f"Unknown framework: {framework}. Use 'torch', 'linen', or 'nnx'.")
            sys.exit(1)
        run_framework_isolation_test(framework)  # type: ignore[arg-type]
    else:
        run_memory_test(cleanup_enabled=True)

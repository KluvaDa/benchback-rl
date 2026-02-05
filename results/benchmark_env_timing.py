"""Benchmark environment execution timing.

This script is fully vibe coded.

This script times the execution of Gymnax environments used in our PPO benchmarks.
It runs a vmap of environments for a single iteration with random actions,
using jax.jit for compilation.

The script includes:
- Cache clearing between environments for isolation
- Warmup runs to ensure JIT compilation is complete
- Multiple repetitions to build statistics (mean, std, min, max, median)
- Trimmed statistics (excluding outliers) for more robust measurements
- Proper synchronization with jax.block_until_ready() for accurate timing
- Results saved in human-readable format
"""

import gc
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import gymnax
from typing import Any

# Same environments used in benchmarks (from benchmark.py)
BENCHMARK_ENVIRONMENTS = (
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "DiscountingChain-bsuite",
    "MemoryChain-bsuite",
    "UmbrellaChain-bsuite",
    "BernoulliBandit-misc",
    "GaussianBandit-misc",
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
)

# Benchmark configuration
NUM_ENVS = 4096  # Larger vmap for more consistent GPU utilization
NUM_WARMUP_RUNS = 10  # More warmup to ensure stable state
NUM_TIMED_RUNS = 100  # More samples for better statistics
TRIM_PERCENT = 10  # Exclude top/bottom 10% for trimmed statistics


@dataclass
class EnvTimingResult:
    """Results for a single environment's timing benchmark."""
    env_name: str
    num_envs: int
    num_warmup_runs: int
    num_timed_runs: int
    obs_dim: int
    num_actions: int
    # Timing statistics in milliseconds (full data)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    # Trimmed statistics (excluding outliers)
    trimmed_mean_ms: float
    trimmed_std_ms: float
    trim_percent: int
    # All individual timings
    all_timings_ms: list[float]


def clear_caches() -> None:
    """Clear all caches to ensure isolation between benchmark runs."""
    gc.collect()
    jax.clear_caches()
    gc.collect()


def create_env_functions(env_name: str) -> tuple[Any, Any, Any, Any, int, int]:
    """Create vmapped and jitted environment functions.
    
    Args:
        env_name: Name of the gymnax environment.
        
    Returns:
        Tuple of (env, env_params, reset_fn, step_fn, obs_dim, num_actions)
        where reset_fn and step_fn are vmapped and jitted.
    """
    # Cast to Any to avoid Pylance errors with gymnax's complex union types
    env: Any
    env_params: Any
    env, env_params = gymnax.make(env_name)
    
    # Get observation and action dimensions
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    obs_dim: int = math.prod(obs_space.shape)
    num_actions: int = action_space.n
    
    # Check if we need to flatten observations (for MinAtar etc.)
    flatten = len(obs_space.shape) > 1
    
    if flatten:
        def _reset(rng, params):
            obs, state = env.reset(rng, params)
            return obs.reshape(-1), state
        def _step(rng, state, action, params):
            obs, state, reward, done, info = env.step(rng, state, action, params)
            return obs.reshape(-1), state, reward, done, info
    else:
        _reset = env.reset
        _step = env.step
    
    # Vmap over: keys (num_envs,), state (num_envs, ...), action (num_envs, ...)
    # params are shared (not vmapped)
    vmapped_reset = jax.vmap(_reset, in_axes=(0, None))
    vmapped_step = jax.vmap(_step, in_axes=(0, 0, 0, None))
    
    # JIT compile
    reset_fn = jax.jit(vmapped_reset)
    step_fn = jax.jit(vmapped_step)
    
    return env, env_params, reset_fn, step_fn, obs_dim, num_actions


def benchmark_env_step(
    env_name: str,
    num_envs: int = NUM_ENVS,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_timed: int = NUM_TIMED_RUNS,
    trim_percent: int = TRIM_PERCENT,
) -> EnvTimingResult:
    """Benchmark a single environment's step execution time.
    
    Args:
        env_name: Name of the gymnax environment.
        num_envs: Number of parallel environments to vmap.
        num_warmup: Number of warmup runs before timing.
        num_timed: Number of timed runs to collect statistics.
        trim_percent: Percentage to trim from each end for trimmed statistics.
        
    Returns:
        EnvTimingResult with timing statistics.
    """
    # Clear caches before starting this environment's benchmark
    clear_caches()
    
    print(f"\nBenchmarking: {env_name}")
    print(f"  Num envs: {num_envs}, Warmup: {num_warmup}, Timed runs: {num_timed}")
    
    # Create environment functions
    env, env_params, reset_fn, step_fn, obs_dim, num_actions = create_env_functions(env_name)
    print(f"  Obs dim: {obs_dim}, Num actions: {num_actions}")
    
    # Initialize RNG
    rng = jax.random.PRNGKey(42)
    
    # Reset environments
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, num_envs)
    obs, state = reset_fn(reset_keys, env_params)
    
    # Block until reset is complete
    jax.block_until_ready((obs, state))
    
    # Small delay to let GPU settle
    time.sleep(0.1)
    
    # =========================================================================
    # Warmup runs - ensure JIT compilation is complete
    # =========================================================================
    print(f"  Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        step_keys = jax.random.split(step_rng, num_envs)
        
        # Random actions
        actions = jax.random.randint(action_rng, (num_envs,), 0, num_actions)
        
        # Step all environments
        obs, state, rewards, dones, info = step_fn(step_keys, state, actions, env_params)
        
        # Block until complete
        jax.block_until_ready((obs, state, rewards, dones))
    
    print("  Warmup complete.")
    
    # Small delay after warmup
    time.sleep(0.1)
    
    # =========================================================================
    # Timed runs - collect statistics
    # =========================================================================
    print(f"  Running {num_timed} timed iterations...")
    timings_ms: list[float] = []
    
    for i in range(num_timed):
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        step_keys = jax.random.split(step_rng, num_envs)
        
        # Random actions
        actions = jax.random.randint(action_rng, (num_envs,), 0, num_actions)
        
        # Time the step
        start = time.perf_counter()
        obs, state, rewards, dones, info = step_fn(step_keys, state, actions, env_params)
        
        # CRITICAL: Block until JAX computation is complete before stopping timer
        jax.block_until_ready((obs, state, rewards, dones))
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000.0
        timings_ms.append(elapsed_ms)
    
    # Compute statistics
    timings_array = jnp.array(timings_ms)
    mean_ms = float(jnp.mean(timings_array))
    std_ms = float(jnp.std(timings_array))
    min_ms = float(jnp.min(timings_array))
    max_ms = float(jnp.max(timings_array))
    median_ms = float(jnp.median(timings_array))
    
    # Compute trimmed statistics (exclude outliers)
    sorted_timings = jnp.sort(timings_array)
    trim_count = int(num_timed * trim_percent / 100)
    if trim_count > 0:
        trimmed_timings = sorted_timings[trim_count:-trim_count]
    else:
        trimmed_timings = sorted_timings
    trimmed_mean_ms = float(jnp.mean(trimmed_timings))
    trimmed_std_ms = float(jnp.std(trimmed_timings))
    
    print(f"  Results: {mean_ms:.3f} ± {std_ms:.3f} ms (min: {min_ms:.3f}, max: {max_ms:.3f}, median: {median_ms:.3f})")
    print(f"  Trimmed ({trim_percent}%): {trimmed_mean_ms:.3f} ± {trimmed_std_ms:.3f} ms")
    
    return EnvTimingResult(
        env_name=env_name,
        num_envs=num_envs,
        num_warmup_runs=num_warmup,
        num_timed_runs=num_timed,
        obs_dim=obs_dim,
        num_actions=num_actions,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        median_ms=median_ms,
        trimmed_mean_ms=trimmed_mean_ms,
        trimmed_std_ms=trimmed_std_ms,
        trim_percent=trim_percent,
        all_timings_ms=timings_ms,
    )


def run_all_benchmarks() -> list[EnvTimingResult]:
    """Run benchmarks for all environments."""
    print("=" * 70)
    print("Environment Timing Benchmark")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Number of parallel environments (vmap): {NUM_ENVS}")
    print(f"  Warmup runs: {NUM_WARMUP_RUNS}")
    print(f"  Timed runs: {NUM_TIMED_RUNS}")
    print(f"  Trim percent (outlier exclusion): {TRIM_PERCENT}%")
    print(f"  Environments: {len(BENCHMARK_ENVIRONMENTS)}")
    print(f"  JAX devices: {jax.devices()}")
    print("=" * 70)
    
    results = []
    for env_name in BENCHMARK_ENVIRONMENTS:
        result = benchmark_env_step(env_name)
        results.append(result)
        # Small delay between environments
        time.sleep(0.2)
    
    return results


def save_results(results: list[EnvTimingResult], output_dir: Path) -> None:
    """Save benchmark results to human-readable files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Save as JSON (structured, machine-readable)
    # =========================================================================
    json_path = output_dir / "env_timing_results.json"
    
    # Convert to JSON-serializable format
    json_data = {
        "metadata": {
            "num_envs": NUM_ENVS,
            "num_warmup_runs": NUM_WARMUP_RUNS,
            "num_timed_runs": NUM_TIMED_RUNS,
            "trim_percent": TRIM_PERCENT,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "jax_devices": [str(d) for d in jax.devices()],
        },
        "results": [asdict(r) for r in results],
    }
    
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nSaved JSON results to: {json_path}")
    
    # =========================================================================
    # Save as human-readable text summary
    # =========================================================================
    txt_path = output_dir / "env_timing_results.txt"
    
    with open(txt_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("ENVIRONMENT TIMING BENCHMARK RESULTS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Number of parallel environments (vmap): {NUM_ENVS}\n")
        f.write(f"  Warmup runs: {NUM_WARMUP_RUNS}\n")
        f.write(f"  Timed runs: {NUM_TIMED_RUNS}\n")
        f.write(f"  Trim percent (outlier exclusion): {TRIM_PERCENT}%\n")
        f.write(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  JAX devices: {jax.devices()}\n\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"{'Environment':<28} {'Trimmed Mean':<14} {'Trimmed Std':<14} {'Mean (ms)':<12} {'Std (ms)':<12} {'Median (ms)':<12}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by trimmed mean time for easier comparison
        sorted_results = sorted(results, key=lambda r: r.trimmed_mean_ms)
        
        for r in sorted_results:
            f.write(f"{r.env_name:<28} {r.trimmed_mean_ms:<14.3f} {r.trimmed_std_ms:<14.3f} {r.mean_ms:<12.3f} {r.std_ms:<12.3f} {r.median_ms:<12.3f}\n")
        
        f.write("-" * 100 + "\n\n")
        
        # Additional details per environment
        f.write("Detailed Results:\n")
        f.write("=" * 100 + "\n\n")
        
        for r in sorted_results:
            f.write(f"{r.env_name}:\n")
            f.write(f"  Observation dim: {r.obs_dim}\n")
            f.write(f"  Number of actions: {r.num_actions}\n")
            f.write(f"  Trimmed Mean ± Std ({r.trim_percent}% excluded): {r.trimmed_mean_ms:.3f} ± {r.trimmed_std_ms:.3f} ms\n")
            f.write(f"  Full Mean ± Std: {r.mean_ms:.3f} ± {r.std_ms:.3f} ms\n")
            f.write(f"  Range: [{r.min_ms:.3f}, {r.max_ms:.3f}] ms\n")
            f.write(f"  Median: {r.median_ms:.3f} ms\n")
            f.write("\n")
    
    print(f"Saved text summary to: {txt_path}")
    
    # =========================================================================
    # Print summary table to console
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY (sorted by trimmed mean execution time)")
    print("=" * 100)
    print(f"{'Environment':<28} {'Trimmed Mean':<14} {'Trimmed Std':<14} {'Obs Dim':<10} {'Actions':<10}")
    print("-" * 100)
    
    for r in sorted_results:
        print(f"{r.env_name:<28} {r.trimmed_mean_ms:<14.3f} {r.trimmed_std_ms:<14.3f} {r.obs_dim:<10} {r.num_actions:<10}")
    
    print("=" * 100)


def main() -> None:
    """Main entry point."""
    # Run benchmarks
    results = run_all_benchmarks()
    
    # Save results
    output_dir = Path(__file__).parent
    save_results(results, output_dir)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()

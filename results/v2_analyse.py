"""
Analyse benchmark results from W&B experiment data (V2).

This module provides data loading utilities and statistical analysis
for the benchmark experiments.

This file is fully vibe coded
"""

from functools import lru_cache
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from scipy.stats import gmean
import wandb


# =============================================================================
# Configuration
# =============================================================================

CACHE_DIR = Path(__file__).parent
WANDB_PROJECT = "kluvada/benchback_rl"

# Experiment names (must match config.notes_config in benchmark.py)
EXP1_NAME = "V2 Exp1: async"
EXP2_NAME = "V2 Exp2: sync, envs"
EXP3_NAME = "V2 Exp3: sync, models"

# OS labels (must match config.notes_user in benchmark.py)
OS_LINUX = "linux"
OS_WINDOWS = "windows"

# Compiled settings for each framework
COMPILED_SETTINGS = {
    "linen": "jax.jit",
    "nnx": "nnx.cached_partial",
    "torch": "torch.compile",
}

# Model size definitions (keys match numpy array string representation)
MODEL_SIZES = {
    "[64 64]": "small",
    "[256 256 256]": "medium",
    "[1024 1024 1024 1024 1024 1024]": "large",
}

# Model size ordering
MODEL_SIZE_ORDER = ["small", "medium", "large"]

# Metrics for speedup comparison
METRIC_TOTAL = "summary.duration_total"
METRIC_ITER_AVG_7 = "summary.duration_iteration_avg_7:"
METRICS = [METRIC_TOTAL, METRIC_ITER_AVG_7]
METRIC_LABELS = {
    METRIC_TOTAL: "Total Duration",
    METRIC_ITER_AVG_7: "Iteration Average (last 7)",
}


# =============================================================================
# Data Loading
# =============================================================================

def _normalize_mixed_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns with mixed types caused by JSON serialization quirks."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        
        type_names = {type(x).__name__ for x in df[col]}
        numeric_types = {'int', 'float', 'NoneType'}
        string_types = {'str'}
        allowed_types = numeric_types | string_types
        
        if not type_names.issubset(allowed_types):
            continue
        
        has_strings = bool(type_names & string_types)
        has_numerics = bool(type_names & {'int', 'float'})
        
        if not has_numerics:
            continue
            
        if not has_strings:
            df[col] = pd.to_numeric(df[col])
            continue
        
        str_mask = df[col].apply(lambda x: isinstance(x, str))
        unique_strs = df.loc[str_mask, col].unique()
        unexpected = [s for s in unique_strs if s != 'NaN']
        if unexpected:
            df[col] = df[col].astype(str)
            print(f"  Converted mixed-type column to string: {col}")
            continue
        
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  Normalized mixed-type column: {col}")
    
    return df


def _fetch_wandb_runs(
    filters: dict[str, Any],
    cache_path: Path,
) -> pd.DataFrame:
    """Fetch runs from W&B API with server-side filtering, with local caching."""
    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Fetching from W&B API with filters: {filters}")
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters=filters)

    summary_list = []
    config_list = []
    name_list = []
    
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append(run.config)
        name_list.append(run.name)

    if not summary_list:
        print(f"Warning: No runs found for filters: {filters}")
        return pd.DataFrame()

    summary_df = pd.json_normalize(summary_list).add_prefix("summary.")
    config_df = pd.json_normalize(config_list).add_prefix("config.")

    runs_df = pd.concat([summary_df, config_df], axis=1)
    runs_df["name"] = name_list
    
    runs_df = _normalize_mixed_type_columns(runs_df)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    runs_df.to_parquet(cache_path, index=False)
    print(f"Cached {len(runs_df)} runs to: {cache_path}")

    return runs_df


@lru_cache(maxsize=1)
def get_exp1_df() -> pd.DataFrame:
    """Get DataFrame for V2 Exp1: async (various envs × frameworks × model sizes)."""
    filters: dict[str, Any] = {"config.notes_config": EXP1_NAME}
    cache_path = CACHE_DIR / "v2_exp1_cache.parquet"
    return _fetch_wandb_runs(filters, cache_path)


@lru_cache(maxsize=1)
def get_exp2_df() -> pd.DataFrame:
    """Get DataFrame for V2 Exp2: sync, envs (various envs × framework/compile combos)."""
    filters: dict[str, Any] = {"config.notes_config": EXP2_NAME}
    cache_path = CACHE_DIR / "v2_exp2_cache.parquet"
    return _fetch_wandb_runs(filters, cache_path)


@lru_cache(maxsize=1)
def get_exp3_df() -> pd.DataFrame:
    """Get DataFrame for V2 Exp3: sync, models (Acrobot × framework/compile × model sizes)."""
    filters: dict[str, Any] = {"config.notes_config": EXP3_NAME}
    cache_path = CACHE_DIR / "v2_exp3_cache.parquet"
    return _fetch_wandb_runs(filters, cache_path)


def filter_compiled(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only compiled runs."""
    mask = df.apply(
        lambda row: COMPILED_SETTINGS.get(row["config.framework"]) == row["config.compile"],
        axis=1,
    )
    return df[mask].copy()


# =============================================================================
# Environment Timing Data
# =============================================================================

ENV_TIMING_FILE = Path(__file__).parent / "env_timing_results.json"


@lru_cache(maxsize=1)
def get_relative_env_speeds() -> dict[str, float]:
    """Load environment timing data from JSON and compute relative speeds.
    
    Returns a dict mapping env_name to relative speed (fastest = 1.0).
    Uses trimmed_mean_ms from the timing results.
    """
    with open(ENV_TIMING_FILE) as f:
        data = json.load(f)
    
    # Extract trimmed mean timing for each environment
    env_times = {
        result["env_name"]: result["trimmed_mean_ms"]
        for result in data["results"]
    }
    
    # Compute relative speeds (fastest = 1.0)
    min_time = min(env_times.values())
    relative_env = {
        env_name: time_ms / min_time
        for env_name, time_ms in env_times.items()
    }
    
    return relative_env


# =============================================================================
# Exp1 Speedup Analysis
# =============================================================================

def compute_exp1_speedups(
    numerator_df: pd.DataFrame,
    denominator_df: pd.DataFrame,
    metric_col: str = "summary.duration_total",
) -> dict[str, list[float]]:
    """Compute speedup = numerator / denominator for matching (env, model_size) pairs.
    
    Args:
        numerator_df: DataFrame for the numerator framework
        denominator_df: DataFrame for the denominator (baseline) framework
        metric_col: Column to use for computing speedup
        
    Returns:
        Dictionary mapping model_size to list of speedup values
    """
    speedups_by_model: dict[str, list[float]] = {size: [] for size in MODEL_SIZE_ORDER}
    
    for _, num_row in numerator_df.iterrows():
        matching = denominator_df[
            (denominator_df["config.env_name"] == num_row["config.env_name"]) &
            (denominator_df["model_size"] == num_row["model_size"])
        ]
        model_size = num_row["model_size"]
        for _, den_row in matching.iterrows():
            if pd.notna(num_row[metric_col]) and pd.notna(den_row[metric_col]):
                speedup = num_row[metric_col] / den_row[metric_col]
                speedups_by_model[model_size].append(speedup)
    
    return speedups_by_model


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of speedup values."""
    if values:
        arr = np.array(values)
        return {
            "gmean": float(gmean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(values),
        }
    return {
        "gmean": float("nan"),
        "median": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "count": 0,
    }


def analyse_exp1_speedups(
    os_name: str = OS_LINUX,
    metric_col: str = METRIC_TOTAL,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute geometric mean and median speedups for Exp1 by model size.
    
    Args:
        os_name: OS filter (OS_LINUX or OS_WINDOWS)
        metric_col: Column to use for computing speedup
        
    Returns:
        Nested dict: {comparison_name: {model_size: {"gmean": value, "median": value, "count": n}}}
        
        model_size keys include "small", "medium", "large", and "overall".
        
        Comparison names:
        - "nnx_vs_linen": nnx / linen (>1 means linen is faster)
        - "torch_vs_linen": torch / linen (>1 means linen is faster)  
        - "torch_vs_nnx": torch / nnx (>1 means nnx is faster)
    """
    # Load and prepare data
    exp1_df = get_exp1_df()
    exp1_df = exp1_df[exp1_df["config.notes_user"] == os_name].copy()
    df = filter_compiled(exp1_df)
    
    # Add model_size column
    df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
    
    # Split by framework
    linen_df = df[df["config.framework"] == "linen"]
    nnx_df = df[df["config.framework"] == "nnx"]
    torch_df = df[df["config.framework"] == "torch"]
    
    # Define comparisons: (name, numerator_df, denominator_df)
    comparisons = [
        ("nnx_vs_linen", nnx_df, linen_df),
        ("torch_vs_linen", torch_df, linen_df),
        ("torch_vs_nnx", torch_df, nnx_df),
    ]
    
    results: dict[str, dict[str, dict[str, float]]] = {}
    
    for comp_name, num_df, den_df in comparisons:
        speedups_by_model = compute_exp1_speedups(num_df, den_df, metric_col=metric_col)
        
        results[comp_name] = {}
        all_values: list[float] = []
        for model_size in MODEL_SIZE_ORDER:
            values = speedups_by_model[model_size]
            all_values.extend(values)
            results[comp_name][model_size] = _compute_stats(values)
        
        # Overall stats (across all model sizes and environments)
        results[comp_name]["overall"] = _compute_stats(all_values)
    
    return results


def _print_stats_row(label: str, stats: dict[str, float]) -> None:
    """Print a single row of speedup statistics."""
    if stats["count"] > 0:
        print(
            f"{label:<12} "
            f"{stats['gmean']:>7.2f}x "
            f"{stats['median']:>7.2f}x "
            f"{stats['min']:>7.2f}x "
            f"{stats['max']:>7.2f}x "
            f"{stats['count']:>5d}"
        )
    else:
        print(f"{label:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>5d}")


def print_exp1_speedups(os_name: str = OS_LINUX) -> None:
    """Print formatted speedup analysis for Exp1 across all metrics."""
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    
    comparison_labels = {
        "nnx_vs_linen": "NNX vs Linen (NNX/Linen)",
        "torch_vs_linen": "Torch vs Linen (Torch/Linen)",
        "torch_vs_nnx": "Torch vs NNX (Torch/NNX)",
    }
    
    display_sizes = MODEL_SIZE_ORDER + ["overall"]
    
    for metric_col in METRICS:
        metric_label = METRIC_LABELS[metric_col]
        results = analyse_exp1_speedups(os_name, metric_col=metric_col)
        
        print(f"\n{'='*70}")
        print(f"V2 Exp1 Speedup Analysis ({os_label}) \u2014 {metric_label}")
        print(f"{'='*70}")
        print("\nSpeedup = numerator_duration / denominator_duration")
        print("Values > 1.0 mean the denominator (baseline) is faster")
        print()
        
        for comp_name, comp_label in comparison_labels.items():
            print(f"\n{comp_label}")
            print("-" * 55)
            print(f"{'Model Size':<12} {'G.Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'N':>5}")
            print("-" * 55)
            
            for model_size in display_sizes:
                stats = results[comp_name][model_size]
                label = "OVERALL" if model_size == "overall" else model_size
                _print_stats_row(label, stats)


def print_all_exp1_analysis() -> None:
    """Print Exp1 speedup analysis for both Linux and Windows."""
    print_exp1_speedups(OS_LINUX)
    print_exp1_speedups(OS_WINDOWS)


# =============================================================================
# Summary Statistics
# =============================================================================

def get_exp1_summary(
    os_name: str = OS_LINUX,
    metric_col: str = METRIC_TOTAL,
) -> pd.DataFrame:
    """Get summary statistics for Exp1 as a DataFrame.
    
    Args:
        os_name: OS filter (OS_LINUX or OS_WINDOWS)
        metric_col: Column to use for computing speedup
    
    Returns a DataFrame with columns:
    - comparison: comparison name
    - model_size: small/medium/large/overall
    - gmean: geometric mean speedup
    - median: median speedup
    - min: minimum speedup
    - max: maximum speedup
    - count: number of data points
    """
    results = analyse_exp1_speedups(os_name, metric_col=metric_col)
    
    rows = []
    for comp_name, model_stats in results.items():
        for model_size, stats in model_stats.items():
            rows.append({
                "comparison": comp_name,
                "model_size": model_size,
                **stats,
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run all analyses and print results."""
    print_all_exp1_analysis()


if __name__ == "__main__":
    main()

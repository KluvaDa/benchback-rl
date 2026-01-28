"""
Generate benchmark result charts from W&B experiment data.

This script creates visualizations for:
- Experiment 1: Various environments with small model (boxplot + slowdown violin)
- Experiment 2: Various model sizes with Acrobot-v1 (boxplot)
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import wandb


# =============================================================================
# Configuration
# =============================================================================

CACHE_PATH = Path(__file__).parent / "wandb_runs_cache.parquet"

# Color schemes
_tab10 = sns.color_palette("tab10")
_tab20c = sns.color_palette("tab20c")

COLORS = {
    "linen": _tab10[2],  # green
    "nnx": _tab10[1],    # orange
    "torch": _tab10[0],  # blue
}

# tab20c: groups of 4 shades (dark to light)
# indices 0-3: blue, 4-7: orange, 8-11: green
COLORS_SLOWDOWN = {
    "torch": {"dark": _tab20c[0], "light1": _tab20c[2], "light2": _tab20c[3]},
    "nnx": {"dark": _tab20c[4], "light1": _tab20c[6], "light2": _tab20c[7]},
    "linen": {"dark": _tab20c[8], "light1": _tab20c[10], "light2": _tab20c[11]},
}

FRAMEWORKS = ["linen", "nnx", "torch"]
OS_LABELS = ["linux docker", "windows docker"]
OS_DISPLAY = ["Linux", "Windows"]


# =============================================================================
# Data Loading
# =============================================================================

def get_wandb_runs_df(use_cache: bool = True, refresh_cache: bool = False) -> pd.DataFrame:
    """Fetch runs from W&B API or load from cache."""
    if use_cache and not refresh_cache and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    api = wandb.Api()
    runs = api.runs("kluvada/benchback_rl")  # type: ignore

    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append(run.config)
        name_list.append(run.name)

    summary_df = pd.json_normalize(summary_list).add_prefix("summary.")
    config_df = pd.json_normalize(config_list).add_prefix("config.")

    runs_df = pd.concat([summary_df, config_df], axis=1)
    runs_df["name"] = name_list
    runs_df.to_parquet(CACHE_PATH, index=False)

    return runs_df


# =============================================================================
# Shared Utilities
# =============================================================================

def save_figure(filename: str, tight: bool = True) -> None:
    """Save figure to results directory and close."""
    if tight:
        plt.tight_layout()
    plt.savefig(Path(__file__).parent / filename, dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close()


def style_boxplot(bp: dict[str, Any], color: Any) -> None:
    """Apply consistent coloring to a boxplot."""
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_edgecolor(color)
    bp["medians"][0].set_color(color)
    for whisker in bp["whiskers"]:
        whisker.set_color(color)
    for cap in bp["caps"]:
        cap.set_color(color)


# =============================================================================
# Experiment 1: Boxplot (Total Duration)
# =============================================================================

def visualise_exp_1_box_plot(runs_df: pd.DataFrame) -> None:
    """Boxplot for experiment 1: various environments with small model."""
    subset = runs_df[runs_df["config.notes_config"] == "experiment 1: various envs"]

    positions_linux = [0, 0.7, 1.4]
    positions_windows = [3, 3.7, 4.4]
    all_positions = positions_linux + positions_windows

    fig, ax = plt.subplots(figsize=(3, 6))

    # Plot boxes for each OS and framework
    for os_label, positions in zip(OS_LABELS, [positions_linux, positions_windows]):
        os_subset = subset[subset["config.notes_user"] == os_label]

        for fw, pos in zip(FRAMEWORKS, positions):
            data = os_subset[os_subset["config.framework"] == fw]["summary.duration_total"].dropna()
            if len(data) > 0:
                bp = ax.boxplot(
                    [data],
                    positions=[pos],
                    widths=0.6,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False,
                    whis=(0, 100),
                )
                style_boxplot(bp, COLORS[fw])

    # Y-axis configuration
    ax.set_yscale("log")
    ax.set_ylabel("total duration (smaller is better)")
    ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}s"))
    ax.yaxis.set_minor_locator(FixedLocator([30, 40, 60, 70, 80, 90, 150]))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # X-axis labels
    ax.set_xticks(all_positions)
    ax.set_xticklabels(FRAMEWORKS * 2, rotation=45, ha="center")
    ax.text(0.7, -0.12, OS_DISPLAY[0], ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold")
    ax.text(3.7, -0.12, OS_DISPLAY[1], ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold")

    # Vertical grid lines
    for pos in all_positions:
        ax.axvline(x=pos, linestyle="-", linewidth=0.5, alpha=0.2, color="gray", zorder=0)

    ax.set_title("Total duration\n(all environments, small model)")
    ax.set_xlim(-0.5, 4.9)
    fig.subplots_adjust(bottom=0.15)
    sns.despine(ax=ax)

    save_figure("exp1_boxplot.png")


# =============================================================================
# Experiment 2: Boxplot (Total Duration)
# =============================================================================

def visualise_exp_2_box_plot(runs_df: pd.DataFrame) -> None:
    """Boxplot for experiment 2: various model sizes with Acrobot-v1."""
    subset = runs_df[runs_df["config.notes_config"] == "experiment 2: various model sizes"].copy()
    subset["hidden_sizes_str"] = subset["config.hidden_sizes"].apply(str)

    model_sizes = ["[64 64]", "[256 256 256]", "[1024 1024 1024 1024 1024 1024]"]
    model_labels = ["s", "m", "l"]

    positions_linux = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    positions_windows = [14, 15, 16, 18, 19, 20, 22, 23, 24]
    all_positions = positions_linux + positions_windows

    fig, ax = plt.subplots(figsize=(4, 6))

    # Plot boxes for each OS, framework, and model size
    for os_label, positions_base in zip(OS_LABELS, [positions_linux, positions_windows]):
        os_subset = subset[subset["config.notes_user"] == os_label]

        pos_idx = 0
        for fw in FRAMEWORKS:
            for ms in model_sizes:
                data = os_subset[
                    (os_subset["config.framework"] == fw) &
                    (os_subset["hidden_sizes_str"] == ms)
                ]["summary.duration_total"].dropna()

                if len(data) > 0:
                    bp = ax.boxplot(
                        [data],
                        positions=[positions_base[pos_idx]],
                        widths=1.8,
                        patch_artist=True,
                        manage_ticks=False,
                        showfliers=False,
                        whis=(0, 100),
                    )
                    style_boxplot(bp, COLORS[fw])
                pos_idx += 1

    # Y-axis configuration
    ax.set_yscale("log")
    ax.set_ylabel("total duration (smaller is better)")
    ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}s"))
    ax.yaxis.set_minor_locator(FixedLocator([30, 40, 60, 70, 80, 90, 150]))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # X-axis labels
    ax.set_xticks(all_positions)
    ax.set_xticklabels(model_labels * 6, ha="center", fontsize=8)

    framework_label_positions = [(1, "linen"), (5, "nnx"), (9, "torch"),
                                 (15, "linen"), (19, "nnx"), (23, "torch")]
    for pos, label in framework_label_positions:
        ax.text(pos, -0.06, label, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9)

    ax.text(5, -0.11, OS_DISPLAY[0], ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold")
    ax.text(19, -0.11, OS_DISPLAY[1], ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold")

    # Vertical grid lines
    for pos in all_positions:
        ax.axvline(x=pos, linestyle="-", linewidth=0.5, alpha=0.2, color="gray", zorder=0)

    ax.set_title("Total duration\n(Acrobot-v1, s/m/l model sizes)")
    ax.set_xlim(-1.2, 25.2)
    fig.subplots_adjust(bottom=0.15)
    sns.despine(ax=ax)

    save_figure("exp2_boxplot.png")


# =============================================================================
# Experiment 1: Slowdown Violin Plot
# =============================================================================

def compute_slowdown_metrics(
    baseline: np.ndarray,
    comparison: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Compute slowdown metrics between baseline and comparison runs.

    Returns: (geo_mean_slowdown, best_case, worst_case, baseline_geo_mean)
    """
    baseline_geo_mean = float(np.exp(np.mean(np.log(baseline))))
    comparison_geo_mean = float(np.exp(np.mean(np.log(comparison))))

    geo_mean_slowdown = comparison_geo_mean / baseline_geo_mean
    best_case = float(np.min(comparison)) / float(np.max(baseline))
    worst_case = float(np.max(comparison)) / float(np.min(baseline))

    return geo_mean_slowdown, best_case, worst_case, baseline_geo_mean


# =============================================================================
# Slowdown Charts: Shared Components
# =============================================================================

# Shared comparison groups for slowdown charts
SLOWDOWN_GROUPS: list[tuple[str, dict[str, str], dict[str, str], str]] = [
    # Framework comparisons on Linux
    ("linen→nnx\n(Linux)",
     {"config.framework": "linen", "config.notes_user": "linux docker"},
     {"config.framework": "nnx", "config.notes_user": "linux docker"},
     "nnx"),
    ("linen→torch\n(Linux)",
     {"config.framework": "linen", "config.notes_user": "linux docker"},
     {"config.framework": "torch", "config.notes_user": "linux docker"},
     "torch"),
    # Framework comparisons on Windows
    ("linen→nnx\n(Windows)",
     {"config.framework": "linen", "config.notes_user": "windows docker"},
     {"config.framework": "nnx", "config.notes_user": "windows docker"},
     "nnx"),
    ("linen→torch\n(Windows)",
     {"config.framework": "linen", "config.notes_user": "windows docker"},
     {"config.framework": "torch", "config.notes_user": "windows docker"},
     "torch"),
    # OS comparisons per framework
    ("Linux→Win\n(linen)",
     {"config.framework": "linen", "config.notes_user": "linux docker"},
     {"config.framework": "linen", "config.notes_user": "windows docker"},
     "linen"),
    ("Linux→Win\n(nnx)",
     {"config.framework": "nnx", "config.notes_user": "linux docker"},
     {"config.framework": "nnx", "config.notes_user": "windows docker"},
     "nnx"),
    ("Linux→Win\n(torch)",
     {"config.framework": "torch", "config.notes_user": "linux docker"},
     {"config.framework": "torch", "config.notes_user": "windows docker"},
     "torch"),
]

SLOWDOWN_POSITIONS = [0, 1, 2, 3, 4.5, 5.5, 6.5]
SLOWDOWN_VLINE_POSITION = 3.75


def draw_slowdown_violin(
    ax: Axes,
    geo_means: np.ndarray,
    best_cases: np.ndarray,
    worst_cases: np.ndarray,
    x_positions: np.ndarray,
    position: float,
    color_key: str,
    width: float = 0.7,
) -> None:
    """Draw violin plots with overlaid scatter points for slowdown data."""
    colors = COLORS_SLOWDOWN[color_key]
    n_points = len(geo_means)

    # Draw violins (need at least 2 points)
    if n_points >= 2:
        for values, color in [
            (worst_cases, colors["light2"]),
            (best_cases, colors["light2"]),
            (geo_means, colors["light1"]),
        ]:
            vp = ax.violinplot(
                [values], positions=[position], widths=width,
                showmeans=False, showmedians=False, showextrema=False
            )
            for body in vp["bodies"]:  # type: ignore[union-attr]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(1.0)

    # Draw vertical lines from best to worst case
    for i in range(n_points):
        ax.plot(
            [x_positions[i], x_positions[i]],
            [best_cases[i], worst_cases[i]],
            color=colors["dark"], linewidth=1.5, zorder=10,
        )

    # Draw geo mean as filled circles
    ax.scatter(
        x_positions, geo_means,
        color=colors["dark"], s=30, zorder=11, edgecolors="none",
    )


def draw_slowdown_sub_axis(
    ax: Axes,
    x_positions: np.ndarray,
    y_values: np.ndarray,
    labels: list[str],
    position: float,
    width: float = 0.7,
) -> None:
    """Draw small x-axis labels underneath the violin."""
    y_min = float(y_values.min())
    y_label_pos = y_min / 1.03
    y_text_pos = y_min / 1.07

    x_left = position - width * 0.3
    x_right = position + width * 0.3
    ax.plot([x_left, x_right], [y_label_pos, y_label_pos],
            color="gray", linewidth=0.8, zorder=5)

    for i, label in enumerate(labels[:len(x_positions)]):
        ax.text(x_positions[i], y_text_pos, label,
                ha="center", va="top", fontsize=6, color="gray")


def setup_slowdown_axes(ax: Axes, title: str, y_max: float | None = None) -> None:
    """Configure common axis settings for slowdown charts."""
    # Vertical separator between framework and OS comparisons
    ax.axvline(x=SLOWDOWN_VLINE_POSITION, color="gray", linestyle="--",
               linewidth=1, alpha=0.7)

    # Y-axis: logarithmic with custom ticks
    ax.set_yscale("log", base=2)
    ax.set_ylabel("Slowdown (higher = slower)")
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1x", "2x", "4x", "8x"])
    ax.set_yticks([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                   3, 5, 6, 7, 9, 10, 11, 12, 13, 14], minor=True)
    ax.yaxis.set_minor_formatter(NullFormatter())
    if y_max is not None:
        ax.set_ylim(top=y_max)

    # Horizontal reference line at 1x (no slowdown)
    ax.axhline(y=1, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # X-axis labels
    ax.set_xticks(SLOWDOWN_POSITIONS)
    ax.set_xticklabels([g[0] for g in SLOWDOWN_GROUPS], fontsize=9)
    ax.text(1.5, -0.07, "Framework Comparisons", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight="bold")
    ax.text(5.5, -0.07, "OS Comparisons", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight="bold")

    ax.set_title(title)
    ax.set_xlim(-0.5, 7.0)


# =============================================================================
# Experiment 1: Slowdown Violin Plot (various environments)
# =============================================================================

def compute_slowdown_data(
    df: pd.DataFrame,
    baseline_filter: dict[str, str],
    comparison_filter: dict[str, str],
) -> list[dict[str, float]]:
    """Compute slowdown metrics per environment for exp1."""
    results: list[dict[str, float]] = []
    for env_name in df["config.env_name"].unique():
        baseline_mask = (df["config.env_name"] == env_name)
        comparison_mask = (df["config.env_name"] == env_name)
        for col, val in baseline_filter.items():
            baseline_mask &= (df[col] == val)
        for col, val in comparison_filter.items():
            comparison_mask &= (df[col] == val)

        baseline_vals = df[baseline_mask]["summary.duration_total"].values
        comparison_vals = df[comparison_mask]["summary.duration_total"].values

        if len(baseline_vals) == 0 or len(comparison_vals) == 0:
            continue

        geo_mean, best, worst, baseline_gm = compute_slowdown_metrics(
            np.asarray(baseline_vals, dtype=float),
            np.asarray(comparison_vals, dtype=float),
        )
        results.append({
            "geo_mean": geo_mean, "best_case": best,
            "worst_case": worst, "baseline_geo_mean": baseline_gm,
        })
    return results


def visualise_exp_1_slowdown(runs_df: pd.DataFrame) -> None:
    """Slowdown violin plot for experiment 1: various environments."""
    subset = runs_df[runs_df["config.notes_config"] == "experiment 1: various envs"].copy()

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_axes([0.1, 0.22, 0.85, 0.71])  # [left, bottom, width, height]

    for i, (_, baseline_filter, comparison_filter, color_key) in enumerate(SLOWDOWN_GROUPS):
        data = compute_slowdown_data(subset, baseline_filter, comparison_filter)
        if len(data) < 2:
            continue

        position = SLOWDOWN_POSITIONS[i]
        width = 0.7

        geo_means = np.array([d["geo_mean"] for d in data])
        best_cases = np.array([d["best_case"] for d in data])
        worst_cases = np.array([d["worst_case"] for d in data])
        baseline_gms = np.array([d["baseline_geo_mean"] for d in data])

        # X positions based on baseline duration
        bm_min, bm_max = float(baseline_gms.min()), float(baseline_gms.max())
        if bm_max > bm_min:
            x_offsets = (baseline_gms - bm_min) / (bm_max - bm_min) * (width * 0.6) - (width * 0.3)
        else:
            x_offsets = np.zeros_like(baseline_gms)
        x_positions = position + x_offsets

        draw_slowdown_violin(ax, geo_means, best_cases, worst_cases,
                             x_positions, position, color_key, width)

        # Sub-axis: baseline duration labels
        if bm_max > bm_min:
            y_all = np.concatenate([geo_means, best_cases, worst_cases])
            y_min = float(y_all.min())
            y_label_pos = y_min / 1.03
            y_text_pos = y_min / 1.05
            x_left, x_right = position - width * 0.3, position + width * 0.3
            ax.plot([x_left, x_right], [y_label_pos, y_label_pos],
                    color="gray", linewidth=0.8, zorder=5)
            ax.text(x_left, y_text_pos, f"{bm_min:.0f}s",
                    ha="center", va="top", fontsize=6, color="gray")
            ax.text(x_right, y_text_pos, f"{bm_max:.0f}s",
                    ha="center", va="top", fontsize=6, color="gray")

    setup_slowdown_axes(ax, "Slowdown (all environments, small model)")

    # Footer explanation
    footer = """Explanation:
- A = baseline runs, B = comparison runs (for each A→B comparison: same environment, different framework or OS).
- g_mean(X) = exp(mean(log(X))) — geometric mean of total duration.
- g_mean_slowdown = g_mean(B) / g_mean(A),  worst_slowdown = max(B) / min(A),  best_slowdown = min(B) / max(A)
Chart:
- Each violin shows the distribution across 8 environments.
- Violin fill: g_mean_slowdown (darker), best/worst_slowdown (lighter).
- Scatter overlay: one point per environment.
  - y: g_mean_slowdown (disc), range from best to worst (vertical line).
  - x position within violin: baseline g_mean(A) duration (left=fast, right=slow).
- Y-axis: log₂ scale, 1x = no slowdown, larger values = slower."""
    fig.text(0.02, 0.01, footer, fontsize=6, va="bottom", ha="left", family="monospace")

    sns.despine(ax=ax)
    save_figure("exp1_slowdown.png", tight=False)


# =============================================================================
# Experiment 2: Slowdown Violin Plot (s/m/l model sizes)
# =============================================================================

MODEL_SIZES = ["[64 64]", "[256 256 256]", "[1024 1024 1024 1024 1024 1024]"]
MODEL_SIZE_LABELS = ["s", "m", "l"]


def compute_slowdown_data_by_model_size(
    df: pd.DataFrame,
    baseline_filter: dict[str, str],
    comparison_filter: dict[str, str],
    model_size: str,
) -> dict[str, float] | None:
    """Compute slowdown metrics for a single model size."""
    baseline_mask = pd.Series(True, index=df.index)
    comparison_mask = pd.Series(True, index=df.index)

    for col, val in baseline_filter.items():
        baseline_mask &= (df[col] == val)
    for col, val in comparison_filter.items():
        comparison_mask &= (df[col] == val)

    baseline_mask &= (df["hidden_sizes_str"] == model_size)
    comparison_mask &= (df["hidden_sizes_str"] == model_size)

    baseline_vals = df[baseline_mask]["summary.duration_total"].values
    comparison_vals = df[comparison_mask]["summary.duration_total"].values

    if len(baseline_vals) == 0 or len(comparison_vals) == 0:
        return None

    geo_mean, best, worst, _ = compute_slowdown_metrics(
        np.asarray(baseline_vals, dtype=float),
        np.asarray(comparison_vals, dtype=float),
    )
    return {"geo_mean": geo_mean, "best_case": best, "worst_case": worst}


def visualise_exp_2_slowdown(runs_df: pd.DataFrame) -> None:
    """Slowdown violin plot for experiment 2: model sizes s/m/l."""
    subset = runs_df[runs_df["config.notes_config"] == "experiment 2: various model sizes"].copy()
    subset["hidden_sizes_str"] = subset["config.hidden_sizes"].apply(str)

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_axes([0.1, 0.22, 0.85, 0.71])  # [left, bottom, width, height]

    for i, (_, baseline_filter, comparison_filter, color_key) in enumerate(SLOWDOWN_GROUPS):
        # Compute data for each model size
        data_by_size = []
        for model_size in MODEL_SIZES:
            data = compute_slowdown_data_by_model_size(
                subset, baseline_filter, comparison_filter, model_size
            )
            if data:
                data_by_size.append(data)

        if len(data_by_size) < 2:
            continue

        position = SLOWDOWN_POSITIONS[i]
        width = 0.7

        geo_means = np.array([d["geo_mean"] for d in data_by_size])
        best_cases = np.array([d["best_case"] for d in data_by_size])
        worst_cases = np.array([d["worst_case"] for d in data_by_size])

        # X positions: evenly spread across the violin
        x_offsets = np.linspace(-width * 0.3, width * 0.3, len(data_by_size))
        x_positions = position + x_offsets

        draw_slowdown_violin(ax, geo_means, best_cases, worst_cases,
                             x_positions, position, color_key, width)

        # Sub-axis: s/m/l labels
        y_all = np.concatenate([geo_means, best_cases, worst_cases])
        draw_slowdown_sub_axis(ax, x_positions, y_all, MODEL_SIZE_LABELS, position, width)

    setup_slowdown_axes(ax, "Slowdown (Acrobot-v1, s/m/l model sizes)", y_max=12)

    # Footer explanation
    footer = """Explanation:
- A = baseline runs, B = comparison runs (for each A→B comparison: same model size, different framework or OS).
- g_mean(X) = exp(mean(log(X))) — geometric mean of total duration.
- g_mean_slowdown = g_mean(B) / g_mean(A),  worst_slowdown = max(B) / min(A),  best_slowdown = min(B) / max(A)
Chart:
- Each violin shows the distribution across 3 model sizes (s, m, l).
- Violin fill: g_mean_slowdown (darker), best/worst_slowdown (lighter).
- Scatter overlay: one point per model size.
  - y: g_mean_slowdown (disc), range from best to worst (vertical line).
  - x position within violin: model size (left=s, middle=m, right=l).
- Y-axis: log₂ scale, 1x = no slowdown, larger values = slower."""
    fig.text(0.02, 0.01, footer, fontsize=6, va="bottom", ha="left", family="monospace")

    sns.despine(ax=ax)
    save_figure("exp2_slowdown.png", tight=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    runs_df = get_wandb_runs_df()
    visualise_exp_1_box_plot(runs_df)
    visualise_exp_2_box_plot(runs_df)
    visualise_exp_1_slowdown(runs_df)
    visualise_exp_2_slowdown(runs_df)
    print("Done")


if __name__ == "__main__":
    main()

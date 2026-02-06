"""
Generate benchmark result charts from W&B experiment data (V2).

This file is fully vibe coded using Claude Opus 4.5.
The code is not quality controlled and does not necessarily follow best practices.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator, FixedLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns

# Import shared data loading and configuration from v2_analyse
from v2_analyse import (
    COMPILED_SETTINGS,
    MODEL_SIZES,
    MODEL_SIZE_ORDER,
    OS_LINUX,
    OS_WINDOWS,
    filter_compiled,
    get_exp1_df,
    get_exp2_df,
    get_exp3_df,
    get_relative_env_speeds,
)

# =============================================================================
# Chart-specific Configuration
# =============================================================================

# Markers by model size
MODEL_MARKERS = {
    "small": "o",   # circle
    "medium": "^",  # triangle
    "large": "s",   # square
}

# Colors from tab20c (three shades per framework: dark for large, medium, light for small)
_tab20c = sns.color_palette("tab20c")

FRAMEWORK_COLORS = {
    "linen": {
        "large": _tab20c[8],   # dark green
        "medium": _tab20c[9],  # medium green
        "small": _tab20c[10],  # light green
    },
    "nnx": {
        "large": _tab20c[4],   # dark orange
        "medium": _tab20c[5],  # medium orange
        "small": _tab20c[6],   # light orange
    },
    "torch": {
        "large": _tab20c[0],   # dark blue
        "medium": _tab20c[1],  # medium blue
        "small": _tab20c[2],   # light blue
    },
}

# Dark colors for violin plots and labels
FRAMEWORK_COLORS_DARK = {
    "linen": _tab20c[8],
    "nnx": _tab20c[4],
    "torch": _tab20c[0],
}

# Gray shades for legend (dark for large, light for small)
MODEL_LEGEND_COLORS = {
    "small": "#999999",   # light gray
    "medium": "#666666",  # medium gray
    "large": "#333333",   # dark gray
}

# Model labels for display
MODEL_LABELS = {"small": "s", "medium": "m", "large": "l"}

# Violin plot resolution
N_VIOLIN_POINTS = 100


# =============================================================================
# Formatting Helpers
# =============================================================================

def format_log_axis_value(x: float, suffix: str = "") -> str:
    """Format a value for log-scale axis labels consistently."""
    if x == 0:
        return f"0{suffix}"
    elif x >= 10:
        return f"{x:.0f}{suffix}"
    elif x >= 1:
        return f"{x:.1f}{suffix}"
    elif x >= 0.1:
        return f"{x:.2f}{suffix}"
    elif x >= 0.01:
        return f"{x:.3f}{suffix}"
    else:
        return f"{x:.4f}{suffix}"


def setup_log_yaxis(ax: Axes, suffix: str = "") -> None:
    """Configure y-axis for log scale with consistent formatting."""
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: format_log_axis_value(x, suffix)
    ))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(np.arange(2, 10) * 0.1).tolist(), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='y', which='minor', length=3)
    ax.tick_params(axis='y', which='major', length=6)


def save_figure(filename: str, tight: bool = True) -> None:
    """Save figure to results directory and close."""
    if tight:
        plt.tight_layout()
    plt.savefig(Path(__file__).parent / filename, dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close()


def draw_half_violin(
    ax: Axes,
    y_data: np.ndarray,
    y_range: np.ndarray,
    x_right: float,
    width: float,
    color,
) -> None:
    """Draw a half-violin (KDE density) extending left from x_right."""
    kde_data = np.log10(y_data)
    kde_range = np.log10(y_range)
    kde = gaussian_kde(kde_data)
    kde.set_bandwidth(kde.factor * 0.15)  # type: ignore[operator]
    density = kde(kde_range)
    density_norm = density / density.max() * width

    ax.fill_betweenx(
        y_range,
        x_right - density_norm,
        x_right,
        color=color,
        alpha=0.4,
        linewidth=0,
        zorder=5,
    )
    ax.plot(
        x_right - density_norm,
        y_range,
        color=color,
        linewidth=1,
        zorder=6,
    )


def create_model_size_legend(ax: Axes, loc: str = "upper left") -> None:
    """Add a legend showing model size markers."""
    legend_elements = [
        Line2D(
            [0], [0],
            marker=MODEL_MARKERS[size],
            markerfacecolor="none",
            markeredgecolor=MODEL_LEGEND_COLORS[size],
            label=size,
            markersize=6,
            linestyle="None",
        )
        for size in MODEL_SIZE_ORDER
    ]
    ax.legend(handles=legend_elements, loc=loc, title="Model size")


# =============================================================================
# Chart: V2 Exp1 Absolute Duration
# =============================================================================

def visualise_v2_exp1_absolute(os_name: str = OS_LINUX) -> None:
    """
    Draw absolute duration chart for V2 Exp1 (async).
    
    Shows total_duration by framework, with model sizes as side-by-side sub-plots
    and relative_env as the small x-axis.
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    # --- Load and prepare data ---
    exp1_df = get_exp1_df()
    exp1_df = exp1_df[exp1_df["config.notes_user"] == os_name].copy()
    df = filter_compiled(exp1_df)
    
    relative_env = get_relative_env_speeds()
    
    df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
    df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # --- Layout parameters ---
    frameworks = ["linen", "nnx", "torch"]
    violin_width = 0.2
    scatter_width = 0.6
    gap = 0.02
    sub_scatter_gap = 0.02
    category_spacing = 1.0
    
    # --- Collect all y values ---
    all_y = np.asarray(df["summary.duration_total"].values)
    y_min = 10  # Fixed minimum for this chart
    
    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_yscale("log")
    
    # --- Y positions for labels (below data area) ---
    y_axis_min = y_min * 0.90
    y_sub_axis = y_axis_min * 0.97
    y_sub_labels = y_axis_min * 0.95
    y_tick_top = y_axis_min * 0.99
    y_tick_top_major = y_axis_min * 1.005
    y_model_label = y_axis_min * 1.02
    y_cat_label = y_axis_min * 0.88
    
    # --- Y range for violin ---
    y_violin_range = np.logspace(np.log10(all_y.min() * 0.95), np.log10(all_y.max() * 1.05), N_VIOLIN_POINTS)
    
    # --- Draw each framework ---
    for fw_idx, framework in enumerate(frameworks):
        cat_center = fw_idx * category_spacing
        scatter_left = cat_center + gap
        violin_right = cat_center
        
        fw_df = df[df["config.framework"] == framework]
        fw_color = FRAMEWORK_COLORS_DARK[framework]
        
        # Collect all y values for this framework's violin
        fw_y = np.asarray(fw_df["summary.duration_total"].values)
        
        # --- Draw half-violin ---
        draw_half_violin(ax, fw_y, y_violin_range, violin_right, violin_width, fw_color)
        
        # --- Draw scatter plots for each model size ---
        n_models = len(MODEL_SIZE_ORDER)
        total_gaps = (n_models - 1) * sub_scatter_gap
        sub_width = (scatter_width - total_gaps) / n_models
        
        for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
            sub_left = scatter_left + model_idx * (sub_width + sub_scatter_gap)
            sub_right = sub_left + sub_width
            
            model_df = fw_df[fw_df["model_size"] == model_size]
            if model_df.empty:
                continue
            
            x_vals = np.asarray(model_df["relative_env"].values)
            y_vals = np.asarray(model_df["summary.duration_total"].values)
            x_min, x_max = x_vals.min(), x_vals.max()
            
            # Draw scatter points
            for x_val, y_val in zip(x_vals, y_vals):
                if x_max > x_min:
                    x_norm = (np.log10(x_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                else:
                    x_norm = 0.5
                x_pos = sub_left + x_norm * sub_width
                
                ax.scatter(
                    [x_pos], [y_val],
                    facecolors="none",
                    edgecolors=FRAMEWORK_COLORS[framework][model_size],
                    marker=MODEL_MARKERS[model_size],
                    s=30,
                    linewidths=1.0,
                    zorder=10,
                )
            
            # Draw sub x-axis
            ax.plot(
                [sub_left, sub_right],
                [y_sub_axis, y_sub_axis],
                color="black",
                linewidth=1,
                zorder=15,
                clip_on=False,
            )
            
            # Add tick marks at specific x values
            tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
            major_ticks = {1, 10, 20}
            if x_max > x_min:
                for tick_val in tick_values:
                    if tick_val >= x_min and tick_val <= x_max:
                        tick_norm = (np.log10(tick_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                        tick_x = sub_left + tick_norm * sub_width
                        tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                        ax.plot(
                            [tick_x, tick_x],
                            [y_sub_axis, tick_top],
                            color="black",
                            linewidth=0.8,
                            zorder=15,
                            clip_on=False,
                        )
            
            # Min and max labels
            ax.text(sub_left, y_sub_labels, f"{x_min:.1f}x", ha="left", va="top", fontsize=6, clip_on=False)
            ax.text(sub_right, y_sub_labels, f"{x_max:.1f}x", ha="right", va="top", fontsize=6, clip_on=False)
            
            # Model size label
            ax.text(
                (sub_left + sub_right) / 2, y_model_label,
                MODEL_LABELS[model_size],
                ha="center", va="bottom",
                fontsize=7,
                clip_on=False,
            )
            
            # Separator line between sub-plots (except after last)
            if model_idx < n_models - 1:
                sep_x = sub_right + sub_scatter_gap / 2
                ax.plot(
                    [sep_x, sep_x],
                    [y_sub_axis, all_y.max() * 1.1],
                    color="#cccccc",
                    linewidth=0.5,
                    zorder=1,
                )
        
        # Framework label
        scatter_right = scatter_left + scatter_width
        ax.text(
            (scatter_left + scatter_right) / 2, y_cat_label,
            framework,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
            color=fw_color,
            clip_on=False,
        )
    
    # --- Configure axes ---
    ax.set_ylabel("Total duration (seconds)")
    ax.set_title(f"V2 Exp1: Total Duration by Framework ({os_label})")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    
    total_width = len(frameworks) * category_spacing
    ax.set_xlim(-violin_width - 0.1, total_width - category_spacing + gap + scatter_width + 0.1)
    ax.set_ylim(y_min, all_y.max() * 1.2)
    
    setup_log_yaxis(ax)
    sns.despine(ax=ax, bottom=True)
    create_model_size_legend(ax)
    
    save_figure(f"v2_exp1_absolute{suffix}.png")
    print(f"Saved v2_exp1_absolute{suffix}.png")


# =============================================================================
# Chart: V2 Exp1 Overhead (Iteration Durations)
# =============================================================================

# Iteration metrics configuration
ITERATION_METRICS = [
    ("summary.duration_iteration_0", "iteration 0"),
    ("summary.duration_iteration_avg_1:7", "iteration 1:7"),
    ("summary.duration_iteration_avg_7:", "iteration 7:100"),
]


def visualise_v2_exp1_overhead(os_name: str = OS_LINUX) -> None:
    """
    Draw iteration duration chart for V2 Exp1.
    
    Shows iteration durations by framework, with iteration type as middle grouping
    and model sizes as inner grouping.
    
    Structure:
    - Outer grouping: linen, nnx, torch (frameworks)
    - Middle grouping: iteration 0, iteration 1:7, iteration 7:100
    - Inner grouping: model sizes (s/m/l) with violin + scatter
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    # --- Load and prepare data ---
    exp1_df = get_exp1_df()
    exp1_df = exp1_df[exp1_df["config.notes_user"] == os_name].copy()
    df = filter_compiled(exp1_df)
    
    relative_env = get_relative_env_speeds()
    
    df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
    df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # --- Layout parameters ---
    frameworks = ["linen", "nnx", "torch"]
    violin_width = 0.12
    scatter_width = 0.35
    gap = 0.02
    sub_scatter_gap = 0.015
    iter_gap = 0.04
    framework_gap = 0.12
    
    # --- Collect all y values ---
    all_y_list = []
    for metric_col, _ in ITERATION_METRICS:
        if metric_col in df.columns:
            all_y_list.extend(df[metric_col].dropna().values)
    all_y = np.array(all_y_list)
    
    if len(all_y) == 0:
        print(f"No data for v2_exp1_overhead{suffix}.png")
        return
    
    y_min = all_y.min()
    y_max = all_y.max()
    
    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_yscale("log")
    
    # --- Y positions for labels ---
    log_range = np.log10(y_max) - np.log10(y_min)
    
    def log_offset(base_y: float, fraction: float) -> float:
        return base_y * (10 ** (-fraction * log_range))
    
    y_axis_min = log_offset(y_min, 0.03)
    y_sub_axis = log_offset(y_axis_min, 0.01)
    y_sub_labels = log_offset(y_axis_min, 0.02)
    y_tick_top = log_offset(y_axis_min, 0.003)
    y_tick_top_major = log_offset(y_axis_min, -0.005)
    y_model_label = log_offset(y_axis_min, -0.008)
    y_iter_label = log_offset(y_axis_min, 0.10)
    y_fw_label = log_offset(y_axis_min, 0.16)
    
    # --- Y range for violin ---
    y_violin_range = np.logspace(np.log10(y_min * 0.95), np.log10(y_max * 1.05), N_VIOLIN_POINTS)
    
    current_x = 0.0
    
    # --- Draw each framework ---
    for fw_idx, framework in enumerate(frameworks):
        fw_start = current_x
        fw_color = FRAMEWORK_COLORS_DARK[framework]
        fw_df = df[df["config.framework"] == framework]
        
        # --- Draw each iteration metric ---
        for iter_idx, (metric_col, iter_label) in enumerate(ITERATION_METRICS):
            iter_start = current_x
            
            if metric_col not in fw_df.columns:
                continue
            
            # Get valid data for this metric
            iter_data = fw_df[fw_df[metric_col].notna()].copy()
            
            if iter_data.empty:
                continue
            
            # Collect all y values for this iteration's violin
            iter_y = np.asarray(iter_data[metric_col].values)
            
            # --- Draw half-violin ---
            violin_right = current_x + violin_width
            draw_half_violin(ax, iter_y, y_violin_range, violin_right, violin_width, fw_color)
            
            # --- Draw scatter plots for each model size ---
            scatter_left = violin_right + gap
            n_models = len(MODEL_SIZE_ORDER)
            total_gaps = (n_models - 1) * sub_scatter_gap
            sub_width = (scatter_width - total_gaps) / n_models
            
            for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
                sub_left = scatter_left + model_idx * (sub_width + sub_scatter_gap)
                sub_right = sub_left + sub_width
                
                model_df = iter_data[iter_data["model_size"] == model_size]
                if model_df.empty:
                    continue
                
                x_vals = np.asarray(model_df["relative_env"].values)
                y_vals = np.asarray(model_df[metric_col].values)
                x_min, x_max = x_vals.min(), x_vals.max()
                
                # Draw scatter points
                for x_val, y_val in zip(x_vals, y_vals):
                    if x_max > x_min:
                        x_norm = (np.log10(x_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                    else:
                        x_norm = 0.5
                    x_pos = sub_left + x_norm * sub_width
                    
                    ax.scatter(
                        [x_pos], [y_val],
                        facecolors="none",
                        edgecolors=FRAMEWORK_COLORS[framework][model_size],
                        marker=MODEL_MARKERS[model_size],
                        s=30,
                        linewidths=1.0,
                        zorder=10,
                    )
                
                # Draw sub x-axis
                ax.plot(
                    [sub_left, sub_right],
                    [y_sub_axis, y_sub_axis],
                    color="black",
                    linewidth=1,
                    zorder=15,
                    clip_on=False,
                )
                
                # Add tick marks at specific x values
                tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
                major_ticks = {1, 10, 20}
                if x_max > x_min:
                    for tick_val in tick_values:
                        if tick_val >= x_min and tick_val <= x_max:
                            tick_norm = (np.log10(tick_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                            tick_x = sub_left + tick_norm * sub_width
                            tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                            ax.plot(
                                [tick_x, tick_x],
                                [y_sub_axis, tick_top],
                                color="black",
                                linewidth=0.8,
                                zorder=15,
                                clip_on=False,
                            )
                
                # Min and max labels (diagonal)
                label_inset = sub_width * 0.12
                ax.text(
                    sub_left + label_inset, y_sub_labels,
                    f"{x_min:.1f}x",
                    ha="center", va="top",
                    fontsize=5,
                    rotation=45,
                    clip_on=False,
                )
                ax.text(
                    sub_right - label_inset, y_sub_labels,
                    f"{x_max:.1f}x",
                    ha="center", va="top",
                    fontsize=5,
                    rotation=45,
                    clip_on=False,
                )
                
                # Model size label
                ax.text(
                    (sub_left + sub_right) / 2, y_model_label,
                    MODEL_LABELS[model_size],
                    ha="center", va="bottom",
                    fontsize=6,
                    clip_on=False,
                )
                
                # Separator line between sub-plots (except after last)
                if model_idx < n_models - 1:
                    sep_x = sub_right + sub_scatter_gap / 2
                    ax.plot(
                        [sep_x, sep_x],
                        [y_sub_axis, y_max * 1.05],
                        color="#cccccc",
                        linewidth=0.5,
                        zorder=1,
                    )
            
            iter_end = scatter_left + scatter_width
            
            # Iteration label (centered over scatter area)
            ax.text(
                scatter_left + scatter_width / 2, y_iter_label,
                iter_label,
                ha="center", va="top",
                fontsize=8,
                color=fw_color,
                clip_on=False,
            )
            
            current_x = iter_end + iter_gap
        
        fw_end = current_x - iter_gap
        
        # Framework label
        ax.text(
            (fw_start + fw_end) / 2, y_fw_label,
            framework.upper(),
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color=fw_color,
            clip_on=False,
        )
        
        # Draw separator between frameworks (except after last)
        if fw_idx < len(frameworks) - 1:
            sep_x = fw_end + framework_gap / 2
            ax.plot(
                [sep_x, sep_x],
                [y_axis_min, y_max * 1.05],
                color="#999999",
                linewidth=1,
                zorder=1,
            )
        
        current_x = fw_end + framework_gap
    
    # --- Configure axes ---
    ax.set_ylabel("Average Iteration Duration (seconds)")
    ax.set_title(f"V2 Exp1 part 2: Overhead ({os_label})")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    
    total_width = current_x - framework_gap
    ax.set_xlim(-0.1, total_width + 0.1)
    ax.set_ylim(log_offset(y_min, 0.22), y_max * 1.2)
    
    setup_log_yaxis(ax)
    sns.despine(ax=ax, bottom=True)
    create_model_size_legend(ax, loc="upper right")
    
    save_figure(f"v2_exp1_overhead{suffix}.png")
    print(f"Saved v2_exp1_overhead{suffix}.png")


# =============================================================================
# Chart: V2 Exp1 Overhead Comparison (Difference)
# =============================================================================

# Comparison pairs: (numerator_col, denominator_col, label)
OVERHEAD_COMPARISONS = [
    ("summary.duration_iteration_0", "summary.duration_iteration_avg_7:", "(iter 0) \u2212 (iter 7:100)"),
    ("summary.duration_iteration_avg_1:7", "summary.duration_iteration_avg_7:", "(iter 1:7) \u2212 (iter 7:100)"),
]


def visualise_v2_exp1_overhead_comparison(os_name: str = OS_LINUX) -> None:
    """
    Draw overhead difference chart for V2 Exp1.
    
    Compares iteration durations to the steady-state (avg 7:100) baseline.
    For each (env, model_size) group, all runs are cross-compared (3x3).
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    title = f"V2 Exp1 part 2: Overhead as difference ({os_label})"
    ylabel = "Average Iteration Duration Difference (seconds)"
    filename = f"v2_exp1_overhead_diff{suffix}.png"
    
    # --- Load and prepare data ---
    exp1_df = get_exp1_df()
    exp1_df = exp1_df[exp1_df["config.notes_user"] == os_name].copy()
    df = filter_compiled(exp1_df)
    
    relative_env = get_relative_env_speeds()
    
    df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
    df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # --- Compute comparison values ---
    # For each (env, model_size, framework), cross-compare all runs
    def compute_comparisons(
        num_col: str, den_col: str, fw_df: pd.DataFrame,
    ) -> dict[str, list[dict]]:
        """Compute comparison values for matching (env, model_size) pairs."""
        results: dict[str, list[dict]] = {size: [] for size in MODEL_SIZE_ORDER}
        
        for (env, model_size), group in fw_df.groupby(["config.env_name", "model_size"]):
            if num_col not in group.columns or den_col not in group.columns:
                continue
            num_vals = group[group[num_col].notna()]
            den_vals = group[group[den_col].notna()]
            
            rel_env = relative_env.get(env, 1.0)
            
            for _, num_row in num_vals.iterrows():
                for _, den_row in den_vals.iterrows():
                    val = num_row[num_col] - den_row[den_col]
                    results[model_size].append({"x": rel_env, "y": val})
        
        return results
    
    # --- Layout parameters ---
    frameworks = ["linen", "nnx", "torch"]
    comp_labels = [c[2] for c in OVERHEAD_COMPARISONS]
    
    violin_width = 0.12
    scatter_width = 0.35
    gap = 0.02
    sub_scatter_gap = 0.015
    comp_gap = 0.04
    framework_gap = 0.12
    
    # --- Pre-compute all comparison data ---
    all_comp_data: dict[str, list[dict[str, list[dict]]]] = {}
    for framework in frameworks:
        fw_df = df[df["config.framework"] == framework]
        fw_comps = []
        for num_col, den_col, _ in OVERHEAD_COMPARISONS:
            fw_comps.append(compute_comparisons(num_col, den_col, fw_df))
        all_comp_data[framework] = fw_comps
    
    # --- Collect all y values ---
    all_y_list = []
    for framework in frameworks:
        for comp_data in all_comp_data[framework]:
            for pts in comp_data.values():
                all_y_list.extend([pt["y"] for pt in pts])
    
    if not all_y_list:
        print(f"No data for {filename}")
        return
    
    all_y = np.array(all_y_list)
    y_min = all_y.min()
    y_max = all_y.max()
    
    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(14, 7))
    linthresh = 1.0
    ax.set_yscale("symlog", linthresh=linthresh)
    
    # --- Symlog-aware position helpers ---
    def _to_symlog(y: float) -> float:
        """Convert data value to symlog visual space."""
        if abs(y) <= linthresh:
            return y
        return np.sign(y) * (linthresh + linthresh * np.log10(abs(y) / linthresh))
    
    def _from_symlog(v: float) -> float:
        """Convert symlog visual coordinate back to data."""
        if abs(v) <= linthresh:
            return v
        return np.sign(v) * linthresh * (10 ** ((abs(v) - linthresh) / linthresh))
    
    vis_min = _to_symlog(y_min)
    vis_max = _to_symlog(y_max)
    vis_range = vis_max - vis_min
    
    def symlog_offset(base_y: float, fraction: float) -> float:
        """Offset base_y downward by fraction of the visual range in symlog space."""
        vis_base = _to_symlog(base_y)
        return _from_symlog(vis_base - fraction * vis_range)
    
    y_axis_min = symlog_offset(y_min, 0.03)
    y_sub_axis = symlog_offset(y_axis_min, 0.01)
    y_sub_labels = symlog_offset(y_axis_min, 0.02)
    y_tick_top = symlog_offset(y_axis_min, 0.003)
    y_tick_top_major = symlog_offset(y_axis_min, -0.005)
    y_model_label = symlog_offset(y_axis_min, -0.008)
    y_comp_label = symlog_offset(y_axis_min, 0.10)
    y_fw_label = symlog_offset(y_axis_min, 0.16)
    
    # --- Y range for violin (evenly spaced in symlog visual space, clamped to axis) ---
    vis_viol_lo = _to_symlog(y_axis_min)
    vis_viol_hi = _to_symlog(y_max + 0.05 * abs(y_max - y_min))
    y_violin_range = np.array([_from_symlog(v) for v in np.linspace(vis_viol_lo, vis_viol_hi, N_VIOLIN_POINTS)])
    
    current_x = 0.0
    
    # --- Draw each framework ---
    for fw_idx, framework in enumerate(frameworks):
        fw_start = current_x
        fw_color = FRAMEWORK_COLORS_DARK[framework]
        
        # --- Draw each comparison ---
        for comp_idx, (comp_data, comp_label) in enumerate(
            zip(all_comp_data[framework], comp_labels)
        ):
            # Collect all y values for this comparison's violin
            comp_y_list = [pt["y"] for pts in comp_data.values() for pt in pts]
            if not comp_y_list:
                continue
            comp_y = np.array(comp_y_list)
            
            # --- Draw half-violin (symlog-aware KDE) ---
            violin_right = current_x + violin_width
            kde_data = np.array([_to_symlog(y) for y in comp_y])
            kde_range = np.array([_to_symlog(y) for y in y_violin_range])
            kde = gaussian_kde(kde_data)
            kde.set_bandwidth(kde.factor * 0.15)  # type: ignore[operator]
            density = kde(kde_range)
            density_norm = density / density.max() * violin_width
            ax.fill_betweenx(
                y_violin_range,
                violin_right - density_norm,
                violin_right,
                color=fw_color,
                alpha=0.4,
                linewidth=0,
                zorder=5,
            )
            ax.plot(
                violin_right - density_norm,
                y_violin_range,
                color=fw_color,
                linewidth=1,
                zorder=6,
            )
            
            # --- Draw scatter plots for each model size ---
            scatter_left = violin_right + gap
            n_models = len(MODEL_SIZE_ORDER)
            total_gaps = (n_models - 1) * sub_scatter_gap
            sub_width = (scatter_width - total_gaps) / n_models
            
            for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
                sub_left = scatter_left + model_idx * (sub_width + sub_scatter_gap)
                sub_right = sub_left + sub_width
                
                data = comp_data.get(model_size, [])
                if not data:
                    continue
                
                x_vals = np.array([pt["x"] for pt in data])
                y_vals = np.array([pt["y"] for pt in data])
                x_min, x_max = x_vals.min(), x_vals.max()
                
                # Draw scatter points
                for x_val, y_val in zip(x_vals, y_vals):
                    if x_max > x_min:
                        x_norm = (np.log10(x_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                    else:
                        x_norm = 0.5
                    x_pos = sub_left + x_norm * sub_width
                    
                    ax.scatter(
                        [x_pos], [y_val],
                        facecolors="none",
                        edgecolors=FRAMEWORK_COLORS[framework][model_size],
                        marker=MODEL_MARKERS[model_size],
                        s=30,
                        linewidths=1.0,
                        zorder=10,
                    )
                
                # Draw sub x-axis
                ax.plot(
                    [sub_left, sub_right],
                    [y_sub_axis, y_sub_axis],
                    color="black",
                    linewidth=1,
                    zorder=15,
                    clip_on=False,
                )
                
                # Add tick marks at specific x values
                tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
                major_ticks = {1, 10, 20}
                if x_max > x_min:
                    for tick_val in tick_values:
                        if tick_val >= x_min and tick_val <= x_max:
                            tick_norm = (np.log10(tick_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                            tick_x = sub_left + tick_norm * sub_width
                            tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                            ax.plot(
                                [tick_x, tick_x],
                                [y_sub_axis, tick_top],
                                color="black",
                                linewidth=0.8,
                                zorder=15,
                                clip_on=False,
                            )
                
                # Min and max labels (diagonal)
                label_inset = sub_width * 0.12
                ax.text(
                    sub_left + label_inset, y_sub_labels,
                    f"{x_min:.1f}x",
                    ha="center", va="top",
                    fontsize=5,
                    rotation=45,
                    clip_on=False,
                )
                ax.text(
                    sub_right - label_inset, y_sub_labels,
                    f"{x_max:.1f}x",
                    ha="center", va="top",
                    fontsize=5,
                    rotation=45,
                    clip_on=False,
                )
                
                # Model size label
                ax.text(
                    (sub_left + sub_right) / 2, y_model_label,
                    MODEL_LABELS[model_size],
                    ha="center", va="bottom",
                    fontsize=6,
                    clip_on=False,
                )
                
                # Separator line between sub-plots (except after last)
                if model_idx < n_models - 1:
                    sep_x = sub_right + sub_scatter_gap / 2
                    ax.plot(
                        [sep_x, sep_x],
                        [y_sub_axis, y_max * 1.05],
                        color="#cccccc",
                        linewidth=0.5,
                        zorder=1,
                    )
            
            iter_end = scatter_left + scatter_width
            
            # Comparison label (centered over scatter area)
            ax.text(
                scatter_left + scatter_width / 2, y_comp_label,
                comp_label,
                ha="center", va="top",
                fontsize=8,
                color=fw_color,
                clip_on=False,
            )
            
            current_x = iter_end + comp_gap
        
        fw_end = current_x - comp_gap
        
        # Framework label
        ax.text(
            (fw_start + fw_end) / 2, y_fw_label,
            framework.upper(),
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color=fw_color,
            clip_on=False,
        )
        
        # Draw separator between frameworks (except after last)
        if fw_idx < len(frameworks) - 1:
            sep_x = fw_end + framework_gap / 2
            ax.plot(
                [sep_x, sep_x],
                [y_axis_min, y_max * 1.05],
                color="#999999",
                linewidth=1,
                zorder=1,
            )
        
        current_x = fw_end + framework_gap
    
    # --- Configure axes ---
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    
    total_width = current_x - framework_gap
    ax.set_xlim(-0.1, total_width + 0.1)
    ax.set_ylim(symlog_offset(y_min, 0.22), y_max * 1.2)
    
    # Add horizontal line at y=0 (no overhead)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, zorder=1)
    
    # Format y-axis for symlog (no units on tick labels)
    def _fmt_symlog_tick(x: float, _pos) -> str:
        if x == 0:
            return "0"
        ax_val = abs(x)
        if ax_val >= 10:
            s = f"{ax_val:.0f}"
        elif ax_val >= 1:
            s = f"{ax_val:.1f}"
        elif ax_val >= 0.1:
            s = f"{ax_val:.2f}"
        else:
            s = f"{ax_val:.3f}"
        return f"\u2212{s}" if x < 0 else s
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_symlog_tick))
    
    # Minor ticks at specific values
    minor_ticks = [-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   2, 3, 4, 5, 6, 7, 8, 9, 20]
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='y', which='minor', length=3)
    ax.tick_params(axis='y', which='major', length=6)
    
    sns.despine(ax=ax, bottom=True)
    create_model_size_legend(ax, loc="upper right")
    
    save_figure(filename)
    print(f"Saved {filename}")


# =============================================================================
# Chart: V2 Exp1 Speedup
# =============================================================================

def visualise_v2_exp1_speedup(
    os_name: str = OS_LINUX,
    metric: str = "total",
) -> None:
    """
    Draw speedup chart for V2 Exp1: nnx→linen, torch→linen, torch→nnx speedups.
    
    Speedup = other_framework_duration / baseline_duration
    So speedup > 1 means baseline is faster.
    
    Args:
        os_name: OS filter (OS_LINUX or OS_WINDOWS).
        metric: "total" for total duration speedup,
                "iteration" for steady-state iteration (avg 7:) speedup.
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    if metric == "iteration":
        metric_col = "summary.duration_iteration_avg_7:"
        chart_title = f"V2 Exp1 part 2: Non-overhead speedup ({os_label})"
        chart_ylabel = "Average Iteration Duration 7:100 speedup (baseline is faster if > 1)"
        chart_filename = f"v2_exp1_speedup_iteration{suffix}.png"
    else:
        metric_col = "summary.duration_total"
        chart_title = f"V2 Exp1: Speedup vs Baseline ({os_label})"
        chart_ylabel = "Speedup (baseline is faster if > 1)"
        chart_filename = f"v2_exp1_speedup{suffix}.png"
    
    # --- Load and prepare data ---
    exp1_df = get_exp1_df()
    exp1_df = exp1_df[exp1_df["config.notes_user"] == os_name].copy()
    df = filter_compiled(exp1_df)
    
    relative_env = get_relative_env_speeds()
    
    df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
    df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # Split by framework
    linen_df = df[df["config.framework"] == "linen"]
    nnx_df = df[df["config.framework"] == "nnx"]
    torch_df = df[df["config.framework"] == "torch"]
    
    # --- Compute speedups ---
    def compute_speedups(numerator_df: pd.DataFrame, denominator_df: pd.DataFrame) -> dict[str, list[dict]]:
        """Compute speedup = numerator / denominator for matching (env, model_size) pairs."""
        speedups_by_model: dict[str, list[dict]] = {size: [] for size in MODEL_SIZE_ORDER}
        for _, num_row in numerator_df.iterrows():
            if pd.isna(num_row.get(metric_col)):
                continue
            matching = denominator_df[
                (denominator_df["config.env_name"] == num_row["config.env_name"]) &
                (denominator_df["model_size"] == num_row["model_size"])
            ]
            model_size = num_row["model_size"]
            for _, den_row in matching.iterrows():
                if pd.isna(den_row.get(metric_col)):
                    continue
                speedup = num_row[metric_col] / den_row[metric_col]
                speedups_by_model[model_size].append({
                    "x": num_row["relative_env"],
                    "y": speedup,
                })
        return speedups_by_model
    
    nnx_linen_speedups = compute_speedups(nnx_df, linen_df)
    torch_linen_speedups = compute_speedups(torch_df, linen_df)
    torch_nnx_speedups = compute_speedups(torch_df, nnx_df)
    
    categories = [
        ("nnx → linen", nnx_linen_speedups, FRAMEWORK_COLORS_DARK["nnx"]),
        ("torch → linen", torch_linen_speedups, FRAMEWORK_COLORS_DARK["torch"]),
        ("torch → nnx", torch_nnx_speedups, FRAMEWORK_COLORS_DARK["torch"]),
    ]
    
    # --- Layout parameters ---
    violin_width = 0.2
    scatter_width = 0.6
    gap = 0.02
    sub_scatter_gap = 0.02
    category_spacing = 1.0
    
    # --- Collect all y values ---
    all_y = np.array([
        pt["y"]
        for _, speedups_by_model, _ in categories
        for pts in speedups_by_model.values()
        for pt in pts
    ])
    y_min = min(all_y.min(), 1.0) * 0.9  # Allow axis to dip below 1x
    
    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_yscale("log")
    
    # --- Y positions for labels ---
    y_axis_min = y_min * 0.90
    y_sub_axis = y_axis_min * 0.97
    y_sub_labels = y_axis_min * 0.95
    y_tick_top = y_axis_min * 0.99
    y_tick_top_major = y_axis_min * 1.005
    y_model_label = y_axis_min * 1.02
    y_cat_label = y_axis_min * 0.88
    
    # --- Y range for violin ---
    y_violin_range = np.logspace(np.log10(all_y.min() * 0.95), np.log10(all_y.max() * 1.05), N_VIOLIN_POINTS)
    
    # --- Draw each category ---
    for cat_idx, (cat_name, speedups_by_model, cat_color) in enumerate(categories):
        cat_center = cat_idx * category_spacing
        scatter_left = cat_center + gap
        violin_right = cat_center
        
        # Collect all y values for this category's violin
        cat_y = np.array([pt["y"] for pts in speedups_by_model.values() for pt in pts])
        
        # --- Draw half-violin ---
        kde_data = np.log10(cat_y)
        kde_range = np.log10(y_violin_range)
        kde = gaussian_kde(kde_data)
        kde.set_bandwidth(kde.factor * 0.15)  # type: ignore[operator]
        density = kde(kde_range)
        density_norm = density / density.max() * violin_width
        
        ax.fill_betweenx(
            y_violin_range,
            violin_right - density_norm,
            violin_right,
            color=cat_color,
            alpha=0.4,
            linewidth=0,
            zorder=5,
        )
        ax.plot(
            violin_right - density_norm,
            y_violin_range,
            color=cat_color,
            linewidth=1,
            zorder=6,
        )
        
        # --- Draw scatter plots for each model size ---
        n_models = len(MODEL_SIZE_ORDER)
        total_gaps = (n_models - 1) * sub_scatter_gap
        sub_width = (scatter_width - total_gaps) / n_models
        
        # Determine which framework's colors to use based on category
        if "nnx → linen" in cat_name:
            color_framework = "nnx"
        else:
            color_framework = "torch"
        
        for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
            sub_left = scatter_left + model_idx * (sub_width + sub_scatter_gap)
            sub_right = sub_left + sub_width
            
            data = speedups_by_model.get(model_size, [])
            if not data:
                continue
            
            x_vals = np.array([pt["x"] for pt in data])
            y_vals = np.array([pt["y"] for pt in data])
            x_min, x_max = x_vals.min(), x_vals.max()
            
            # Draw scatter points
            for x_val, y_val in zip(x_vals, y_vals):
                if x_max > x_min:
                    x_norm = (np.log10(x_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                else:
                    x_norm = 0.5
                x_pos = sub_left + x_norm * sub_width
                
                ax.scatter(
                    [x_pos], [y_val],
                    facecolors="none",
                    edgecolors=FRAMEWORK_COLORS[color_framework][model_size],
                    marker=MODEL_MARKERS[model_size],
                    s=30,
                    linewidths=1.0,
                    zorder=10,
                )
            
            # Draw sub x-axis
            ax.plot(
                [sub_left, sub_right],
                [y_sub_axis, y_sub_axis],
                color="black",
                linewidth=1,
                zorder=15,
                clip_on=False,
            )
            
            # Add tick marks at specific x values
            tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
            major_ticks = {1, 10, 20}
            if x_max > x_min:
                for tick_val in tick_values:
                    if tick_val >= x_min and tick_val <= x_max:
                        tick_norm = (np.log10(tick_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                        tick_x = sub_left + tick_norm * sub_width
                        tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                        ax.plot(
                            [tick_x, tick_x],
                            [y_sub_axis, tick_top],
                            color="black",
                            linewidth=0.8,
                            zorder=15,
                            clip_on=False,
                        )
            
            # Min and max labels
            ax.text(sub_left, y_sub_labels, f"{x_min:.1f}x", ha="left", va="top", fontsize=6, clip_on=False)
            ax.text(sub_right, y_sub_labels, f"{x_max:.1f}x", ha="right", va="top", fontsize=6, clip_on=False)
            
            # Model size label
            ax.text(
                (sub_left + sub_right) / 2, y_model_label,
                MODEL_LABELS[model_size],
                ha="center", va="bottom",
                fontsize=7,
                clip_on=False,
            )
            
            # Separator line between sub-plots (except after last)
            if model_idx < n_models - 1:
                sep_x = sub_right + sub_scatter_gap / 2
                ax.plot(
                    [sep_x, sep_x],
                    [y_sub_axis, all_y.max() * 1.1],
                    color="#cccccc",
                    linewidth=0.5,
                    zorder=1,
                )
        
        # Category label
        scatter_right = scatter_left + scatter_width
        ax.text(
            (scatter_left + scatter_right) / 2, y_cat_label,
            cat_name,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
            color=cat_color,
            clip_on=False,
        )
    
    # --- Configure axes ---
    ax.set_ylabel(chart_ylabel)
    ax.set_title(chart_title)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    
    total_width = len(categories) * category_spacing
    ax.set_xlim(-violin_width - 0.1, total_width - category_spacing + gap + scatter_width + 0.1)
    ax.set_ylim(y_min, all_y.max() * 1.2)
    
    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, zorder=1)
    
    # Y-axis ticks at powers of 2
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_log_axis_value(x, "x")))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    sns.despine(ax=ax, bottom=True)
    create_model_size_legend(ax)
    
    save_figure(chart_filename)
    print(f"Saved {chart_filename}")


# =============================================================================
# Chart: V2 Absolute Sync (Exp2 & Exp3)
# =============================================================================

def visualise_v2_absolute_sync(os_name: str = OS_LINUX) -> None:
    """
    Draw absolute timing chart for sync experiments (exp2 & exp3).
    
    Structure:
    - 2 rows: rollout_avg_7:, update_avg_7:
    - 3 frameworks: Linen | NNX | Torch
    - Per framework: violin + scatter with s/m/l model columns
    - Small model: full x-axis with all envs
    - Medium/Large models: zero-width vertical lines (Acrobot only)
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    # --- Load and prepare data ---
    exp2 = filter_compiled(get_exp2_df())
    exp3 = filter_compiled(get_exp3_df())
    
    exp2 = exp2[exp2["config.notes_user"] == os_name].copy()
    exp3 = exp3[exp3["config.notes_user"] == os_name].copy()
    
    relative_env = get_relative_env_speeds()
    acrobot_relative_env = relative_env.get("Acrobot-v1", 1.0)
    
    for df in [exp2, exp3]:
        df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
        df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # --- Configuration ---
    metrics = [
        ("summary.duration_rollout_avg_7:", "Avg Rollout Duration (s)"),
        ("summary.duration_update_avg_7:", "Avg Update Duration (s)"),
    ]
    frameworks = ["linen", "nnx", "torch"]
    
    # Layout parameters
    violin_width = 0.2
    small_scatter_width = 0.3
    zero_width_spacing = 0.1
    gap = 0.02
    sub_scatter_gap = 0.02
    framework_gap = 0.08
    
    # --- Create figure ---
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    
    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[row_idx]
        ax.set_yscale("log")
        
        # Collect all y values for this metric
        all_y_list = []
        for df in [exp2, exp3]:
            if metric_col in df.columns:
                all_y_list.extend(df[metric_col].dropna().values)
        all_y = np.array(all_y_list)
        
        if len(all_y) == 0:
            continue
        
        y_min = all_y.min()
        y_max = all_y.max()
        
        # Y positions for labels - proportional to log-range
        log_range = np.log10(y_max) - np.log10(y_min)
        
        def log_offset(base_y: float, fraction: float) -> float:
            return base_y * (10 ** (-fraction * log_range))
        
        y_axis_min = log_offset(y_min, 0.03)
        y_sub_axis = log_offset(y_axis_min, 0.01)
        y_sub_labels = log_offset(y_axis_min, 0.02)
        y_tick_top = log_offset(y_axis_min, 0.003)
        y_tick_top_major = log_offset(y_axis_min, -0.005)
        y_model_label = log_offset(y_axis_min, -0.008)
        y_fw_label = log_offset(y_axis_min, 0.12)
        
        # Y range for violin
        y_violin_range = np.logspace(np.log10(y_min * 0.95), np.log10(y_max * 1.05), N_VIOLIN_POINTS)
        
        current_x = 0.0
        
        for framework in frameworks:
            fw_start = current_x
            fw_color = FRAMEWORK_COLORS_DARK[framework]
            
            # Get data for this framework
            exp2_fw = exp2[exp2["config.framework"] == framework]
            exp3_fw = exp3[exp3["config.framework"] == framework]
            
            # Collect data by model size
            # Small: from exp2 (all envs)
            # Medium/Large: from exp3 (Acrobot only)
            data_by_model: dict[str, list[dict]] = {size: [] for size in MODEL_SIZE_ORDER}
            
            for _, row in exp2_fw.iterrows():
                if metric_col in row and pd.notna(row[metric_col]):
                    data_by_model["small"].append({
                        "x": row["relative_env"],
                        "y": row[metric_col],
                    })
            
            for _, row in exp3_fw.iterrows():
                model_size = row["model_size"]
                if model_size in ["medium", "large"] and metric_col in row and pd.notna(row[metric_col]):
                    data_by_model[model_size].append({
                        "x": row["relative_env"],
                        "y": row[metric_col],
                    })
            
            # Collect all y values for this framework's violin
            fw_y = np.array([pt["y"] for pts in data_by_model.values() for pt in pts])
            if len(fw_y) == 0:
                current_x += framework_gap
                continue
            
            # --- Draw half-violin ---
            violin_right = current_x + violin_width
            
            kde_data = np.log10(fw_y)
            kde_range = np.log10(y_violin_range)
            kde = gaussian_kde(kde_data)
            kde.set_bandwidth(kde.factor * 0.15)  # type: ignore[operator]
            density = kde(kde_range)
            density_norm = density / density.max() * violin_width
            
            ax.fill_betweenx(
                y_violin_range,
                violin_right - density_norm,
                violin_right,
                color=fw_color,
                alpha=0.4,
                linewidth=0,
                zorder=5,
            )
            ax.plot(
                violin_right - density_norm,
                y_violin_range,
                color=fw_color,
                linewidth=1,
                zorder=6,
            )
            
            # --- Draw scatter columns ---
            scatter_left = violin_right + gap
            scatter_x = scatter_left
            
            for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
                data = data_by_model.get(model_size, [])
                pt_color = FRAMEWORK_COLORS[framework][model_size]
                
                if model_size == "small":
                    # Full width axis for small model
                    sub_left = scatter_x
                    sub_right = sub_left + small_scatter_width
                    
                    if data:
                        x_vals = np.array([pt["x"] for pt in data])
                        y_vals = np.array([pt["y"] for pt in data])
                        x_min_data, x_max_data = x_vals.min(), x_vals.max()
                        
                        # Draw scatter points
                        for x_val, y_val in zip(x_vals, y_vals):
                            if x_max_data > x_min_data:
                                x_norm = (np.log10(x_val) - np.log10(x_min_data)) / (np.log10(x_max_data) - np.log10(x_min_data))
                            else:
                                x_norm = 0.5
                            x_pos = sub_left + x_norm * small_scatter_width
                            
                            ax.scatter(
                                [x_pos], [y_val],
                                facecolors="none",
                                edgecolors=pt_color,
                                marker=MODEL_MARKERS[model_size],
                                s=30,
                                linewidths=1.0,
                                zorder=10,
                            )
                        
                        # Draw sub x-axis
                        ax.plot(
                            [sub_left, sub_right],
                            [y_sub_axis, y_sub_axis],
                            color="black",
                            linewidth=1,
                            zorder=15,
                            clip_on=False,
                        )
                        
                        # Tick marks at specific x values
                        tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
                        major_ticks = {1, 10, 20}
                        for tick_val in tick_values:
                            if tick_val >= x_min_data and tick_val <= x_max_data:
                                tick_norm = (np.log10(tick_val) - np.log10(x_min_data)) / (np.log10(x_max_data) - np.log10(x_min_data))
                                tick_x = sub_left + tick_norm * small_scatter_width
                                tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                                ax.plot(
                                    [tick_x, tick_x],
                                    [y_sub_axis, tick_top],
                                    color="black",
                                    linewidth=0.8,
                                    zorder=15,
                                    clip_on=False,
                                )
                        
                        # Min/max labels (diagonal)
                        label_inset = small_scatter_width * 0.12
                        ax.text(
                            sub_left + label_inset, y_sub_labels,
                            f"{x_min_data:.1f}x",
                            ha="center", va="top",
                            fontsize=6,
                            rotation=45,
                            clip_on=False,
                        )
                        ax.text(
                            sub_right - label_inset, y_sub_labels,
                            f"{x_max_data:.1f}x",
                            ha="center", va="top",
                            fontsize=6,
                            rotation=45,
                            clip_on=False,
                        )
                    
                    # Model label
                    ax.text(
                        (sub_left + sub_right) / 2, y_model_label,
                        MODEL_LABELS[model_size],
                        ha="center", va="bottom",
                        fontsize=7,
                        clip_on=False,
                    )
                    
                    scatter_x = sub_right + sub_scatter_gap
                    
                else:
                    # Zero-width axis for m/l models
                    line_x = scatter_x + zero_width_spacing / 2
                    
                    # Draw thin vertical line
                    ax.plot(
                        [line_x, line_x],
                        [y_sub_axis, y_max * 1.05],
                        color="#cccccc",
                        linewidth=0.5,
                        zorder=1,
                    )
                    
                    # Draw points
                    if data:
                        for pt in data:
                            ax.scatter(
                                [line_x], [pt["y"]],
                                facecolors="none",
                                edgecolors=pt_color,
                                marker=MODEL_MARKERS[model_size],
                                s=30,
                                linewidths=1.0,
                                zorder=10,
                            )
                    
                    # Model label
                    ax.text(
                        line_x, y_model_label,
                        MODEL_LABELS[model_size],
                        ha="center", va="bottom",
                        fontsize=7,
                        clip_on=False,
                    )
                    
                    # X-value label (diagonal)
                    ax.text(
                        line_x, y_sub_labels,
                        f"{acrobot_relative_env:.1f}x",
                        ha="center", va="top",
                        fontsize=6,
                        rotation=45,
                        clip_on=False,
                    )
                    
                    scatter_x = line_x + zero_width_spacing / 2 + sub_scatter_gap
            
            fw_end = scatter_x - sub_scatter_gap
            
            # Framework label
            ax.text(
                (fw_start + fw_end) / 2, y_fw_label,
                framework.upper(),
                ha="center", va="top",
                fontsize=10, fontweight="bold",
                color=fw_color,
                clip_on=False,
            )
            
            current_x = fw_end + framework_gap
        
        # Configure axes
        ax.set_ylabel(metric_label)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        
        total_width = current_x - framework_gap
        ax.set_xlim(-0.1, total_width + 0.1)
        ax.set_ylim(log_offset(y_min, 0.18), y_max * 1.1)
        
        setup_log_yaxis(ax)
        sns.despine(ax=ax, bottom=True)
        
        if row_idx == 0:
            ax.set_title(f"V2 Sync Timings ({os_label})", fontsize=14, fontweight="bold")
            create_model_size_legend(ax)
    
    plt.tight_layout()
    save_figure(f"v2_absolute_sync{suffix}.png", tight=False)
    print(f"Saved v2_absolute_sync{suffix}.png")


# =============================================================================
# Chart: V2 Speedup Sync (Exp2 & Exp3)
# =============================================================================

def visualise_v2_speedup_sync(os_name: str = OS_LINUX) -> None:
    """
    Draw speedup chart for sync experiments (exp2 & exp3).
    
    Structure:
    - 2 rows: rollout_avg_7:, update_avg_7:
    - 3 speedup categories: nnx→linen, torch→linen, torch→nnx
    - Per category: violin + scatter with s/m/l model columns
    - Small model: full x-axis with all envs
    - Medium/Large models: zero-width vertical lines (Acrobot only)
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    # --- Load and prepare data ---
    exp2 = filter_compiled(get_exp2_df())
    exp3 = filter_compiled(get_exp3_df())
    
    exp2 = exp2[exp2["config.notes_user"] == os_name].copy()
    exp3 = exp3[exp3["config.notes_user"] == os_name].copy()
    
    relative_env = get_relative_env_speeds()
    acrobot_relative_env = relative_env.get("Acrobot-v1", 1.0)
    
    for df in [exp2, exp3]:
        df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
        df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # --- Configuration ---
    metrics = [
        ("summary.duration_rollout_avg_7:", "Rollout Speedup"),
        ("summary.duration_update_avg_7:", "Update Speedup"),
    ]
    
    categories = [
        ("nnx → linen", "nnx", "linen", FRAMEWORK_COLORS_DARK["nnx"]),
        ("torch → linen", "torch", "linen", FRAMEWORK_COLORS_DARK["torch"]),
        ("torch → nnx", "torch", "nnx", FRAMEWORK_COLORS_DARK["torch"]),
    ]
    
    # Layout parameters
    violin_width = 0.2
    small_scatter_width = 0.3
    zero_width_spacing = 0.1
    gap = 0.02
    sub_scatter_gap = 0.02
    category_gap = 0.08
    
    # --- Create figure ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[row_idx]
        ax.set_yscale("log")
        
        # --- Compute speedups for each category ---
        def compute_speedups_by_model(
            num_fw: str, den_fw: str, exp2_df: pd.DataFrame, exp3_df: pd.DataFrame
        ) -> dict[str, list[dict]]:
            """Compute speedup = numerator / denominator for matching (env, model_size) pairs."""
            speedups_by_model: dict[str, list[dict]] = {size: [] for size in MODEL_SIZE_ORDER}
            
            # Small model: from exp2 (all envs)
            num_exp2 = exp2_df[exp2_df["config.framework"] == num_fw]
            den_exp2 = exp2_df[exp2_df["config.framework"] == den_fw]
            
            for _, num_row in num_exp2.iterrows():
                if metric_col not in num_row or pd.isna(num_row[metric_col]):
                    continue
                matching = den_exp2[den_exp2["config.env_name"] == num_row["config.env_name"]]
                for _, den_row in matching.iterrows():
                    if metric_col not in den_row or pd.isna(den_row[metric_col]):
                        continue
                    speedup = num_row[metric_col] / den_row[metric_col]
                    speedups_by_model["small"].append({
                        "x": num_row["relative_env"],
                        "y": speedup,
                    })
            
            # Medium/Large models: from exp3 (Acrobot only)
            num_exp3 = exp3_df[exp3_df["config.framework"] == num_fw]
            den_exp3 = exp3_df[exp3_df["config.framework"] == den_fw]
            
            for _, num_row in num_exp3.iterrows():
                model_size = num_row["model_size"]
                if model_size not in ["medium", "large"]:
                    continue
                if metric_col not in num_row or pd.isna(num_row[metric_col]):
                    continue
                matching = den_exp3[
                    (den_exp3["config.env_name"] == num_row["config.env_name"]) &
                    (den_exp3["model_size"] == model_size)
                ]
                for _, den_row in matching.iterrows():
                    if metric_col not in den_row or pd.isna(den_row[metric_col]):
                        continue
                    speedup = num_row[metric_col] / den_row[metric_col]
                    speedups_by_model[model_size].append({
                        "x": num_row["relative_env"],
                        "y": speedup,
                    })
            
            return speedups_by_model
        
        # Compute speedups for all categories
        category_data = []
        for cat_name, num_fw, den_fw, cat_color in categories:
            speedups = compute_speedups_by_model(num_fw, den_fw, exp2, exp3)
            category_data.append((cat_name, speedups, cat_color, num_fw))
        
        # Collect all y values for this metric
        all_y_list = []
        for _, speedups_by_model, _, _ in category_data:
            for pts in speedups_by_model.values():
                all_y_list.extend([pt["y"] for pt in pts])
        
        if not all_y_list:
            continue
            
        all_y = np.array(all_y_list)
        y_min = min(all_y.min(), 1.0)  # Ensure we show y=1 line
        y_max = all_y.max()
        
        # Y positions for labels - proportional to log-range
        log_range = np.log10(y_max) - np.log10(y_min)
        
        def log_offset(base_y: float, fraction: float) -> float:
            return base_y * (10 ** (-fraction * log_range))
        
        y_axis_min = log_offset(y_min, 0.03)
        y_sub_axis = log_offset(y_axis_min, 0.01)
        y_sub_labels = log_offset(y_axis_min, 0.02)
        y_tick_top = log_offset(y_axis_min, 0.003)
        y_tick_top_major = log_offset(y_axis_min, -0.005)
        y_model_label = log_offset(y_axis_min, -0.008)
        y_cat_label = log_offset(y_axis_min, 0.12)
        
        # Y range for violin
        y_violin_range = np.logspace(np.log10(y_min * 0.95), np.log10(y_max * 1.05), N_VIOLIN_POINTS)
        
        current_x = 0.0
        
        for cat_name, speedups_by_model, cat_color, color_framework in category_data:
            cat_start = current_x
            
            # Collect all y values for this category's violin
            cat_y_list = [pt["y"] for pts in speedups_by_model.values() for pt in pts]
            if not cat_y_list:
                current_x += category_gap
                continue
            cat_y = np.array(cat_y_list)
            
            # --- Draw half-violin ---
            violin_right = current_x + violin_width
            
            kde_data = np.log10(cat_y)
            kde_range = np.log10(y_violin_range)
            kde = gaussian_kde(kde_data)
            kde.set_bandwidth(kde.factor * 0.15)  # type: ignore[operator]
            density = kde(kde_range)
            density_norm = density / density.max() * violin_width
            
            ax.fill_betweenx(
                y_violin_range,
                violin_right - density_norm,
                violin_right,
                color=cat_color,
                alpha=0.4,
                linewidth=0,
                zorder=5,
            )
            ax.plot(
                violin_right - density_norm,
                y_violin_range,
                color=cat_color,
                linewidth=1,
                zorder=6,
            )
            
            # --- Draw scatter columns ---
            scatter_left = violin_right + gap
            scatter_x = scatter_left
            
            for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
                data = speedups_by_model.get(model_size, [])
                pt_color = FRAMEWORK_COLORS[color_framework][model_size]
                
                if model_size == "small":
                    # Full width axis for small model
                    sub_left = scatter_x
                    sub_right = sub_left + small_scatter_width
                    
                    if data:
                        x_vals = np.array([pt["x"] for pt in data])
                        y_vals = np.array([pt["y"] for pt in data])
                        x_min_data, x_max_data = x_vals.min(), x_vals.max()
                        
                        # Draw scatter points
                        for x_val, y_val in zip(x_vals, y_vals):
                            if x_max_data > x_min_data:
                                x_norm = (np.log10(x_val) - np.log10(x_min_data)) / (np.log10(x_max_data) - np.log10(x_min_data))
                            else:
                                x_norm = 0.5
                            x_pos = sub_left + x_norm * small_scatter_width
                            
                            ax.scatter(
                                [x_pos], [y_val],
                                facecolors="none",
                                edgecolors=pt_color,
                                marker=MODEL_MARKERS[model_size],
                                s=30,
                                linewidths=1.0,
                                zorder=10,
                            )
                        
                        # Draw sub x-axis
                        ax.plot(
                            [sub_left, sub_right],
                            [y_sub_axis, y_sub_axis],
                            color="black",
                            linewidth=1,
                            zorder=15,
                            clip_on=False,
                        )
                        
                        # Tick marks at specific x values
                        tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
                        major_ticks = {1, 10, 20}
                        for tick_val in tick_values:
                            if tick_val >= x_min_data and tick_val <= x_max_data:
                                tick_norm = (np.log10(tick_val) - np.log10(x_min_data)) / (np.log10(x_max_data) - np.log10(x_min_data))
                                tick_x = sub_left + tick_norm * small_scatter_width
                                tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                                ax.plot(
                                    [tick_x, tick_x],
                                    [y_sub_axis, tick_top],
                                    color="black",
                                    linewidth=0.8,
                                    zorder=15,
                                    clip_on=False,
                                )
                        
                        # Min/max labels (diagonal)
                        label_inset = small_scatter_width * 0.12
                        ax.text(
                            sub_left + label_inset, y_sub_labels,
                            f"{x_min_data:.1f}x",
                            ha="center", va="top",
                            fontsize=6,
                            rotation=45,
                            clip_on=False,
                        )
                        ax.text(
                            sub_right - label_inset, y_sub_labels,
                            f"{x_max_data:.1f}x",
                            ha="center", va="top",
                            fontsize=6,
                            rotation=45,
                            clip_on=False,
                        )
                    
                    # Model label
                    ax.text(
                        (sub_left + sub_right) / 2, y_model_label,
                        MODEL_LABELS[model_size],
                        ha="center", va="bottom",
                        fontsize=7,
                        clip_on=False,
                    )
                    
                    scatter_x = sub_right + sub_scatter_gap
                    
                else:
                    # Zero-width axis for m/l models
                    line_x = scatter_x + zero_width_spacing / 2
                    
                    # Draw thin vertical line
                    ax.plot(
                        [line_x, line_x],
                        [y_sub_axis, y_max * 1.05],
                        color="#cccccc",
                        linewidth=0.5,
                        zorder=1,
                    )
                    
                    # Draw points
                    if data:
                        for pt in data:
                            ax.scatter(
                                [line_x], [pt["y"]],
                                facecolors="none",
                                edgecolors=pt_color,
                                marker=MODEL_MARKERS[model_size],
                                s=30,
                                linewidths=1.0,
                                zorder=10,
                            )
                    
                    # Model label
                    ax.text(
                        line_x, y_model_label,
                        MODEL_LABELS[model_size],
                        ha="center", va="bottom",
                        fontsize=7,
                        clip_on=False,
                    )
                    
                    # X-value label (diagonal)
                    ax.text(
                        line_x, y_sub_labels,
                        f"{acrobot_relative_env:.1f}x",
                        ha="center", va="top",
                        fontsize=6,
                        rotation=45,
                        clip_on=False,
                    )
                    
                    scatter_x = line_x + zero_width_spacing / 2 + sub_scatter_gap
            
            cat_end = scatter_x - sub_scatter_gap
            
            # Category label
            ax.text(
                (cat_start + cat_end) / 2, y_cat_label,
                cat_name,
                ha="center", va="top",
                fontsize=10, fontweight="bold",
                color=cat_color,
                clip_on=False,
            )
            
            current_x = cat_end + category_gap
        
        # Configure axes
        ax.set_ylabel(metric_label)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        
        total_width = current_x - category_gap
        ax.set_xlim(-0.1, total_width + 0.1)
        ax.set_ylim(log_offset(y_min, 0.18), y_max * 1.1)
        
        # Add horizontal line at y=1 (no speedup)
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, zorder=1)
        
        # Y-axis ticks at powers of 2
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_log_axis_value(x, "x")))
        ax.yaxis.set_minor_formatter(NullFormatter())
        
        sns.despine(ax=ax, bottom=True)
        
        if row_idx == 0:
            ax.set_title(f"V2 Sync Speedup ({os_label})", fontsize=14, fontweight="bold")
            create_model_size_legend(ax)
    
    plt.tight_layout()
    save_figure(f"v2_speedup_sync{suffix}.png", tight=False)
    print(f"Saved v2_speedup_sync{suffix}.png")


# =============================================================================
# Chart: V2 Absolute Compile Options (Exp2 & Exp3)
# =============================================================================

# Compile option categories in order
COMPILE_CATEGORIES = [
    # (framework, compile_value, top_label, middle_label, bottom_label)
    ("linen", "jax.jit", "linen", "jax.jit", ""),
    ("nnx", "nnx.cached_partial", "nnx", "nnx.jit", "+cached_partial"),
    ("torch", "torch.compile", "torch", "torch.compile", "+env jax.jit"),
    ("nnx", "nnx.jit", "nnx", "nnx.jit", ""),
    ("torch", "torch.nocompile/env.jit", "torch", "", "env jax.jit"),
    ("linen", "none", "linen", "None", ""),
    ("nnx", "none", "nnx", "None", ""),
    ("torch", "none", "torch", "None", ""),
]


def visualise_v2_absolute_compile(os_name: str = OS_LINUX) -> None:
    """
    Draw absolute timing chart comparing all compile options (exp2 & exp3).
    
    Structure:
    - 2 rows: rollout_avg_7:, update_avg_7:
    - 8 compile option categories (see COMPILE_CATEGORIES)
    - Per category: violin + scatter with s/m/l model columns
    - Small model: full x-axis with all envs
    - Medium/Large models: zero-width vertical lines (Acrobot only)
    """
    os_label = "Linux" if os_name == OS_LINUX else "Windows"
    suffix = "" if os_name == OS_LINUX else "_windows"
    
    # --- Load and prepare data (no filter_compiled - we want all compile options) ---
    exp2 = get_exp2_df()
    exp3 = get_exp3_df()
    
    exp2 = exp2[exp2["config.notes_user"] == os_name].copy()
    exp3 = exp3[exp3["config.notes_user"] == os_name].copy()
    
    relative_env = get_relative_env_speeds()
    acrobot_relative_env = relative_env.get("Acrobot-v1", 1.0)
    
    for df in [exp2, exp3]:
        df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
        df["relative_env"] = df["config.env_name"].map(relative_env)
    
    # --- Configuration ---
    metrics = [
        ("summary.duration_rollout_avg_7:", "Avg Rollout Duration (s)"),
        ("summary.duration_update_avg_7:", "Avg Update Duration (s)"),
    ]
    
    # Layout parameters
    violin_width = 0.15
    small_scatter_width = 0.2
    zero_width_spacing = 0.08
    gap = 0.02
    sub_scatter_gap = 0.015
    category_gap = 0.06
    
    # --- Create figure ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[row_idx]
        ax.set_yscale("log")
        
        # Collect all y values for this metric
        all_y_list = []
        for df in [exp2, exp3]:
            if metric_col in df.columns:
                all_y_list.extend(df[metric_col].dropna().values)
        all_y = np.array(all_y_list)
        
        if len(all_y) == 0:
            continue
        
        y_min = all_y.min()
        y_max = all_y.max()
        
        # Y positions for labels - proportional to log-range
        log_range = np.log10(y_max) - np.log10(y_min)
        
        def log_offset(base_y: float, fraction: float) -> float:
            return base_y * (10 ** (-fraction * log_range))
        
        y_axis_min = log_offset(y_min, 0.03)
        y_sub_axis = log_offset(y_axis_min, 0.01)
        y_sub_labels = log_offset(y_axis_min, 0.02)
        y_tick_top = log_offset(y_axis_min, 0.003)
        y_tick_top_major = log_offset(y_axis_min, -0.005)
        y_model_label = log_offset(y_axis_min, -0.008)
        y_bottom_label = log_offset(y_axis_min, 0.10)
        y_middle_label = log_offset(y_axis_min, 0.14)
        y_top_label = log_offset(y_axis_min, 0.18)
        
        # Y range for violin
        y_violin_range = np.logspace(np.log10(y_min * 0.95), np.log10(y_max * 1.05), N_VIOLIN_POINTS)
        
        current_x = 0.0
        
        for framework, compile_val, top_lbl, middle_lbl, bottom_lbl in COMPILE_CATEGORIES:
            cat_start = current_x
            fw_color = FRAMEWORK_COLORS_DARK[framework]
            
            # Get data for this framework + compile option
            exp2_cat = exp2[(exp2["config.framework"] == framework) & (exp2["config.compile"] == compile_val)]
            exp3_cat = exp3[(exp3["config.framework"] == framework) & (exp3["config.compile"] == compile_val)]
            
            # Collect data by model size
            # Small: from exp2 (all envs)
            # Medium/Large: from exp3 (Acrobot only)
            data_by_model: dict[str, list[dict]] = {size: [] for size in MODEL_SIZE_ORDER}
            
            for _, row in exp2_cat.iterrows():
                if metric_col in row and pd.notna(row[metric_col]):
                    data_by_model["small"].append({
                        "x": row["relative_env"],
                        "y": row[metric_col],
                    })
            
            for _, row in exp3_cat.iterrows():
                model_size = row["model_size"]
                if model_size in ["medium", "large"] and metric_col in row and pd.notna(row[metric_col]):
                    data_by_model[model_size].append({
                        "x": row["relative_env"],
                        "y": row[metric_col],
                    })
            
            # Collect all y values for this category's violin
            fw_y = np.array([pt["y"] for pts in data_by_model.values() for pt in pts])
            if len(fw_y) == 0:
                current_x += category_gap
                continue
            
            # --- Draw half-violin ---
            violin_right = current_x + violin_width
            draw_half_violin(ax, fw_y, y_violin_range, violin_right, violin_width, fw_color)
            
            # --- Draw scatter columns ---
            scatter_left = violin_right + gap
            scatter_x = scatter_left
            
            for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
                data = data_by_model.get(model_size, [])
                pt_color = FRAMEWORK_COLORS[framework][model_size]
                
                if model_size == "small":
                    # Full width axis for small model
                    sub_left = scatter_x
                    sub_right = sub_left + small_scatter_width
                    
                    if data:
                        x_vals = np.array([pt["x"] for pt in data])
                        y_vals = np.array([pt["y"] for pt in data])
                        x_min_data, x_max_data = x_vals.min(), x_vals.max()
                        
                        # Draw scatter points
                        for x_val, y_val in zip(x_vals, y_vals):
                            if x_max_data > x_min_data:
                                x_norm = (np.log10(x_val) - np.log10(x_min_data)) / (np.log10(x_max_data) - np.log10(x_min_data))
                            else:
                                x_norm = 0.5
                            x_pos = sub_left + x_norm * small_scatter_width
                            
                            ax.scatter(
                                [x_pos], [y_val],
                                facecolors="none",
                                edgecolors=pt_color,
                                marker=MODEL_MARKERS[model_size],
                                s=30,
                                linewidths=1.0,
                                zorder=10,
                            )
                        
                        # Draw sub x-axis
                        ax.plot(
                            [sub_left, sub_right],
                            [y_sub_axis, y_sub_axis],
                            color="black",
                            linewidth=1,
                            zorder=15,
                            clip_on=False,
                        )
                        
                        # Tick marks at specific x values
                        tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
                        major_ticks = {1, 10, 20}
                        for tick_val in tick_values:
                            if tick_val >= x_min_data and tick_val <= x_max_data:
                                tick_norm = (np.log10(tick_val) - np.log10(x_min_data)) / (np.log10(x_max_data) - np.log10(x_min_data))
                                tick_x = sub_left + tick_norm * small_scatter_width
                                tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                                ax.plot(
                                    [tick_x, tick_x],
                                    [y_sub_axis, tick_top],
                                    color="black",
                                    linewidth=0.8,
                                    zorder=15,
                                    clip_on=False,
                                )
                        
                        # Min/max labels (diagonal)
                        label_inset = small_scatter_width * 0.12
                        ax.text(
                            sub_left + label_inset, y_sub_labels,
                            f"{x_min_data:.1f}x",
                            ha="center", va="top",
                            fontsize=5,
                            rotation=45,
                            clip_on=False,
                        )
                        ax.text(
                            sub_right - label_inset, y_sub_labels,
                            f"{x_max_data:.1f}x",
                            ha="center", va="top",
                            fontsize=5,
                            rotation=45,
                            clip_on=False,
                        )
                    
                    # Model label
                    ax.text(
                        (sub_left + sub_right) / 2, y_model_label,
                        MODEL_LABELS[model_size],
                        ha="center", va="bottom",
                        fontsize=6,
                        clip_on=False,
                    )
                    
                    scatter_x = sub_right + sub_scatter_gap
                    
                else:
                    # Zero-width axis for m/l models
                    line_x = scatter_x + zero_width_spacing / 2
                    
                    # Draw thin vertical line
                    ax.plot(
                        [line_x, line_x],
                        [y_sub_axis, y_max * 1.05],
                        color="#cccccc",
                        linewidth=0.5,
                        zorder=1,
                    )
                    
                    # Draw points
                    if data:
                        for pt in data:
                            ax.scatter(
                                [line_x], [pt["y"]],
                                facecolors="none",
                                edgecolors=pt_color,
                                marker=MODEL_MARKERS[model_size],
                                s=30,
                                linewidths=1.0,
                                zorder=10,
                            )
                    
                    # Model label
                    ax.text(
                        line_x, y_model_label,
                        MODEL_LABELS[model_size],
                        ha="center", va="bottom",
                        fontsize=6,
                        clip_on=False,
                    )
                    
                    # X-value label (diagonal)
                    ax.text(
                        line_x, y_sub_labels,
                        f"{acrobot_relative_env:.1f}x",
                        ha="center", va="top",
                        fontsize=5,
                        rotation=45,
                        clip_on=False,
                    )
                    
                    scatter_x = line_x + zero_width_spacing / 2 + sub_scatter_gap
            
            cat_end = scatter_x - sub_scatter_gap
            cat_center = (cat_start + cat_end) / 2
            
            # Category labels (3 lines: top bold, middle, bottom)
            ax.text(
                cat_center, y_top_label,
                top_lbl,
                ha="center", va="top",
                fontsize=9, fontweight="bold",
                color=fw_color,
                clip_on=False,
            )
            if bottom_lbl:
                ax.text(
                    cat_center, y_middle_label,
                    bottom_lbl,
                    ha="center", va="top",
                    fontsize=7,
                    color=fw_color,
                    clip_on=False,
                )
            if middle_lbl:
                ax.text(
                    cat_center, y_bottom_label,
                    middle_lbl,
                    ha="center", va="top",
                    fontsize=7,
                    color=fw_color,
                    clip_on=False,
                )
            
            current_x = cat_end + category_gap
        
        # Configure axes
        ax.set_ylabel(metric_label)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        
        total_width = current_x - category_gap
        ax.set_xlim(-0.1, total_width + 0.1)
        ax.set_ylim(log_offset(y_min, 0.24), y_max * 1.1)
        
        setup_log_yaxis(ax)
        sns.despine(ax=ax, bottom=True)
        
        if row_idx == 0:
            ax.set_title(f"V2 Compile Options: Sync Timings ({os_label})", fontsize=14, fontweight="bold")
            create_model_size_legend(ax)
    
    plt.tight_layout()
    save_figure(f"v2_absolute_compile{suffix}.png", tight=False)
    print(f"Saved v2_absolute_compile{suffix}.png")


# =============================================================================
# Chart: V2 Exp1 Absolute OS Comparison
# =============================================================================

def visualise_v2_exp1_absolute_os() -> None:
    """
    Draw combined absolute duration chart for V2 Exp1, showing Linux and Windows
    side by side on the same y-axis.

    Left group: Linux (linen, nnx, torch)
    Right group: Windows (linen, nnx, torch)
    """
    # --- Load and prepare data for both OS ---
    exp1_df = get_exp1_df()
    relative_env = get_relative_env_speeds()

    frameworks = ["linen", "nnx", "torch"]
    os_list = [(OS_LINUX, "Linux"), (OS_WINDOWS, "Windows")]

    data_by_os: dict[str, pd.DataFrame] = {}
    for os_name, _ in os_list:
        os_df = exp1_df[exp1_df["config.notes_user"] == os_name].copy()
        os_df = filter_compiled(os_df)
        os_df["model_size"] = os_df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
        os_df["relative_env"] = os_df["config.env_name"].map(relative_env)
        data_by_os[os_name] = os_df

    # --- Layout parameters ---
    violin_width = 0.2
    scatter_width = 0.6
    gap = 0.02
    sub_scatter_gap = 0.02
    category_spacing = 1.0
    os_gap = 0.5  # Extra gap between OS groups

    # --- Collect all y values from both OS ---
    all_y_list: list[float] = []
    for os_name, _ in os_list:
        all_y_list.extend(data_by_os[os_name]["summary.duration_total"].values)
    all_y = np.asarray(all_y_list)
    y_min = 10  # Fixed minimum

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_yscale("log")

    # --- Y positions for labels (below data area) ---
    y_axis_min = y_min * 0.90
    y_sub_axis = y_axis_min * 0.97
    y_sub_labels = y_axis_min * 0.95
    y_tick_top = y_axis_min * 0.99
    y_tick_top_major = y_axis_min * 1.005
    y_model_label = y_axis_min * 1.02
    y_cat_label = y_axis_min * 0.88
    y_os_label = y_axis_min * 0.78

    # --- Y range for violin ---
    y_violin_range = np.logspace(
        np.log10(all_y.min() * 0.95), np.log10(all_y.max() * 1.05), N_VIOLIN_POINTS,
    )

    # --- Draw each OS group ---
    for os_idx, (os_name, os_label) in enumerate(os_list):
        df = data_by_os[os_name]
        os_x_offset = os_idx * (len(frameworks) * category_spacing + os_gap)

        for fw_idx, framework in enumerate(frameworks):
            cat_center = os_x_offset + fw_idx * category_spacing
            scatter_left = cat_center + gap
            violin_right = cat_center

            fw_df = df[df["config.framework"] == framework]
            fw_color = FRAMEWORK_COLORS_DARK[framework]

            fw_y = np.asarray(fw_df["summary.duration_total"].values)

            # Draw half-violin (needs >= 2 data points for KDE)
            if len(fw_y) >= 2:
                draw_half_violin(ax, fw_y, y_violin_range, violin_right, violin_width, fw_color)

            # Draw scatter plots for each model size
            n_models = len(MODEL_SIZE_ORDER)
            total_gaps = (n_models - 1) * sub_scatter_gap
            sub_width = (scatter_width - total_gaps) / n_models

            for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
                sub_left = scatter_left + model_idx * (sub_width + sub_scatter_gap)
                sub_right = sub_left + sub_width

                model_df = fw_df[fw_df["model_size"] == model_size]
                if model_df.empty:
                    continue

                x_vals = np.asarray(model_df["relative_env"].values)
                y_vals = np.asarray(model_df["summary.duration_total"].values)
                x_min, x_max = x_vals.min(), x_vals.max()

                # Draw scatter points
                for x_val, y_val in zip(x_vals, y_vals):
                    if x_max > x_min:
                        x_norm = (np.log10(x_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                    else:
                        x_norm = 0.5
                    x_pos = sub_left + x_norm * sub_width

                    ax.scatter(
                        [x_pos], [y_val],
                        facecolors="none",
                        edgecolors=FRAMEWORK_COLORS[framework][model_size],
                        marker=MODEL_MARKERS[model_size],
                        s=30,
                        linewidths=1.0,
                        zorder=10,
                    )

                # Draw sub x-axis
                ax.plot(
                    [sub_left, sub_right],
                    [y_sub_axis, y_sub_axis],
                    color="black",
                    linewidth=1,
                    zorder=15,
                    clip_on=False,
                )

                # Add tick marks at specific x values
                tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
                major_ticks = {1, 10, 20}
                if x_max > x_min:
                    for tick_val in tick_values:
                        if tick_val >= x_min and tick_val <= x_max:
                            tick_norm = (np.log10(tick_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                            tick_x = sub_left + tick_norm * sub_width
                            tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                            ax.plot(
                                [tick_x, tick_x],
                                [y_sub_axis, tick_top],
                                color="black",
                                linewidth=0.8,
                                zorder=15,
                                clip_on=False,
                            )

                # Min and max labels
                ax.text(sub_left, y_sub_labels, f"{x_min:.1f}x", ha="left", va="top", fontsize=6, clip_on=False)
                ax.text(sub_right, y_sub_labels, f"{x_max:.1f}x", ha="right", va="top", fontsize=6, clip_on=False)

                # Model size label
                ax.text(
                    (sub_left + sub_right) / 2, y_model_label,
                    MODEL_LABELS[model_size],
                    ha="center", va="bottom",
                    fontsize=7,
                    clip_on=False,
                )

                # Separator line between sub-plots (except after last)
                if model_idx < n_models - 1:
                    sep_x = sub_right + sub_scatter_gap / 2
                    ax.plot(
                        [sep_x, sep_x],
                        [y_sub_axis, all_y.max() * 1.1],
                        color="#cccccc",
                        linewidth=0.5,
                        zorder=1,
                    )

            # Framework label
            scatter_right_pos = scatter_left + scatter_width
            ax.text(
                (scatter_left + scatter_right_pos) / 2, y_cat_label,
                framework,
                ha="center", va="top",
                fontsize=10, fontweight="bold",
                color=fw_color,
                clip_on=False,
            )

        # OS group label
        os_group_left = os_x_offset - violin_width
        os_group_right = os_x_offset + (len(frameworks) - 1) * category_spacing + gap + scatter_width
        ax.text(
            (os_group_left + os_group_right) / 2, y_os_label,
            os_label,
            ha="center", va="top",
            fontsize=12, fontweight="bold",
            clip_on=False,
        )

        # OS separator (between groups, not after last)
        if os_idx < len(os_list) - 1:
            next_os_offset = (os_idx + 1) * (len(frameworks) * category_spacing + os_gap)
            next_os_violin_left = next_os_offset - violin_width
            sep_x = (os_group_right + next_os_violin_left) / 2
            ax.plot(
                [sep_x, sep_x],
                [y_os_label, all_y.max() * 1.1],
                color="#999999",
                linewidth=1.5,
                linestyle="--",
                zorder=1,
                clip_on=False,
            )

    # --- Configure axes ---
    ax.set_ylabel("Total duration (seconds)")
    ax.set_title("V2 Exp1: Total Duration by Framework (Linux vs Windows)")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)

    last_os_offset = (len(os_list) - 1) * (len(frameworks) * category_spacing + os_gap)
    right_edge = last_os_offset + (len(frameworks) - 1) * category_spacing + gap + scatter_width
    ax.set_xlim(-violin_width - 0.1, right_edge + 0.1)
    ax.set_ylim(y_min, all_y.max() * 1.2)

    setup_log_yaxis(ax)
    sns.despine(ax=ax, bottom=True)
    create_model_size_legend(ax)

    save_figure("v2_exp1_absolute_os.png")
    print("Saved v2_exp1_absolute_os.png")


# =============================================================================
# Chart: V2 Exp1 OS Speedup (Windows \u2192 Linux)
# =============================================================================

def visualise_v2_exp1_speedup_os() -> None:
    """
    Draw OS speedup chart for V2 Exp1: Windows \u2192 Linux speedup per framework.

    Speedup = Windows_duration / Linux_duration
    Speedup > 1 means Linux is faster.

    For each (env_name, model_size, framework) group, there is 1 Windows run
    and 3 Linux runs, yielding 3 comparison points.
    """
    metric_col = "summary.duration_total"

    # --- Load and prepare data ---
    exp1_df = get_exp1_df()
    relative_env = get_relative_env_speeds()

    frameworks = ["linen", "nnx", "torch"]

    linux_df = filter_compiled(exp1_df[exp1_df["config.notes_user"] == OS_LINUX].copy())
    windows_df = filter_compiled(exp1_df[exp1_df["config.notes_user"] == OS_WINDOWS].copy())

    for df in [linux_df, windows_df]:
        df["model_size"] = df["config.hidden_sizes"].apply(str).map(MODEL_SIZES)
        df["relative_env"] = df["config.env_name"].map(relative_env)

    # --- Compute speedups per framework ---
    def compute_os_speedups(framework: str) -> dict[str, list[dict]]:
        """Compute speedup = windows / linux for matching (env, model_size) pairs."""
        win_fw = windows_df[windows_df["config.framework"] == framework]
        lin_fw = linux_df[linux_df["config.framework"] == framework]

        speedups_by_model: dict[str, list[dict]] = {size: [] for size in MODEL_SIZE_ORDER}

        for _, win_row in win_fw.iterrows():
            if pd.isna(win_row.get(metric_col)):
                continue
            matching = lin_fw[
                (lin_fw["config.env_name"] == win_row["config.env_name"]) &
                (lin_fw["model_size"] == win_row["model_size"])
            ]
            model_size = win_row["model_size"]
            for _, lin_row in matching.iterrows():
                if pd.isna(lin_row.get(metric_col)):
                    continue
                speedup = win_row[metric_col] / lin_row[metric_col]
                speedups_by_model[model_size].append({
                    "x": win_row["relative_env"],
                    "y": speedup,
                })

        return speedups_by_model

    categories = [
        (fw, compute_os_speedups(fw), FRAMEWORK_COLORS_DARK[fw])
        for fw in frameworks
    ]

    # --- Layout parameters ---
    violin_width = 0.2
    scatter_width = 0.6
    gap = 0.02
    sub_scatter_gap = 0.02
    category_spacing = 1.0

    # --- Collect all y values ---
    all_y = np.array([
        pt["y"]
        for _, speedups_by_model, _ in categories
        for pts in speedups_by_model.values()
        for pt in pts
    ])

    if len(all_y) == 0:
        print("No data for v2_exp1_speedup_os.png")
        return

    y_min = min(all_y.min(), 1.0) * 0.9

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_yscale("log")

    # --- Y positions for labels ---
    y_axis_min = y_min * 0.90
    y_sub_axis = y_axis_min * 0.97
    y_sub_labels = y_axis_min * 0.95
    y_tick_top = y_axis_min * 0.99
    y_tick_top_major = y_axis_min * 1.005
    y_model_label = y_axis_min * 1.02
    y_cat_label = y_axis_min * 0.88

    # --- Y range for violin ---
    y_violin_range = np.logspace(
        np.log10(all_y.min() * 0.95), np.log10(all_y.max() * 1.05), N_VIOLIN_POINTS,
    )

    # --- Draw each category (framework) ---
    for cat_idx, (cat_name, speedups_by_model, cat_color) in enumerate(categories):
        cat_center = cat_idx * category_spacing
        scatter_left = cat_center + gap
        violin_right = cat_center

        # Collect all y values for this category's violin
        cat_y = np.array([pt["y"] for pts in speedups_by_model.values() for pt in pts])

        if len(cat_y) == 0:
            continue

        # Draw half-violin (needs >= 2 data points)
        if len(cat_y) >= 2:
            draw_half_violin(ax, cat_y, y_violin_range, violin_right, violin_width, cat_color)

        # Draw scatter plots for each model size
        n_models = len(MODEL_SIZE_ORDER)
        total_gaps = (n_models - 1) * sub_scatter_gap
        sub_width = (scatter_width - total_gaps) / n_models

        for model_idx, model_size in enumerate(MODEL_SIZE_ORDER):
            sub_left = scatter_left + model_idx * (sub_width + sub_scatter_gap)
            sub_right = sub_left + sub_width

            data = speedups_by_model.get(model_size, [])
            if not data:
                continue

            x_vals = np.array([pt["x"] for pt in data])
            y_vals = np.array([pt["y"] for pt in data])
            x_min, x_max = x_vals.min(), x_vals.max()

            # Draw scatter points
            for x_val, y_val in zip(x_vals, y_vals):
                if x_max > x_min:
                    x_norm = (np.log10(x_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                else:
                    x_norm = 0.5
                x_pos = sub_left + x_norm * sub_width

                ax.scatter(
                    [x_pos], [y_val],
                    facecolors="none",
                    edgecolors=FRAMEWORK_COLORS[cat_name][model_size],
                    marker=MODEL_MARKERS[model_size],
                    s=30,
                    linewidths=1.0,
                    zorder=10,
                )

            # Draw sub x-axis
            ax.plot(
                [sub_left, sub_right],
                [y_sub_axis, y_sub_axis],
                color="black",
                linewidth=1,
                zorder=15,
                clip_on=False,
            )

            # Add tick marks
            tick_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
            major_ticks = {1, 10, 20}
            if x_max > x_min:
                for tick_val in tick_values:
                    if tick_val >= x_min and tick_val <= x_max:
                        tick_norm = (np.log10(tick_val) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                        tick_x = sub_left + tick_norm * sub_width
                        tick_top = y_tick_top_major if tick_val in major_ticks else y_tick_top
                        ax.plot(
                            [tick_x, tick_x],
                            [y_sub_axis, tick_top],
                            color="black",
                            linewidth=0.8,
                            zorder=15,
                            clip_on=False,
                        )

            # Min and max labels
            ax.text(sub_left, y_sub_labels, f"{x_min:.1f}x", ha="left", va="top", fontsize=6, clip_on=False)
            ax.text(sub_right, y_sub_labels, f"{x_max:.1f}x", ha="right", va="top", fontsize=6, clip_on=False)

            # Model size label
            ax.text(
                (sub_left + sub_right) / 2, y_model_label,
                MODEL_LABELS[model_size],
                ha="center", va="bottom",
                fontsize=7,
                clip_on=False,
            )

            # Separator line between sub-plots (except after last)
            if model_idx < n_models - 1:
                sep_x = sub_right + sub_scatter_gap / 2
                ax.plot(
                    [sep_x, sep_x],
                    [y_sub_axis, all_y.max() * 1.1],
                    color="#cccccc",
                    linewidth=0.5,
                    zorder=1,
                )

        # Category (framework) label
        scatter_right_pos = scatter_left + scatter_width
        ax.text(
            (scatter_left + scatter_right_pos) / 2, y_cat_label,
            cat_name,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
            color=cat_color,
            clip_on=False,
        )

    # --- Configure axes ---
    ax.set_ylabel("Speedup (Linux is faster if > 1)")
    ax.set_title("V2 Exp1: Windows \u2192 Linux Speedup")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)

    total_width = len(categories) * category_spacing
    ax.set_xlim(-violin_width - 0.1, total_width - category_spacing + gap + scatter_width + 0.1)
    ax.set_ylim(y_min, all_y.max() * 1.2)

    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, zorder=1)

    # Y-axis ticks at powers of 2
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_log_axis_value(x, "x")))
    ax.yaxis.set_minor_formatter(NullFormatter())

    sns.despine(ax=ax, bottom=True)
    create_model_size_legend(ax)

    save_figure("v2_exp1_speedup_os.png")
    print("Saved v2_exp1_speedup_os.png")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Generate benchmark charts."""
    # Generate Linux charts
    visualise_v2_exp1_absolute(OS_LINUX)
    visualise_v2_exp1_overhead(OS_LINUX)
    visualise_v2_exp1_overhead_comparison(OS_LINUX)
    visualise_v2_exp1_speedup(OS_LINUX)
    visualise_v2_exp1_speedup(OS_LINUX, metric="iteration")
    visualise_v2_absolute_sync(OS_LINUX)
    visualise_v2_speedup_sync(OS_LINUX)
    visualise_v2_absolute_compile(OS_LINUX)
    
    # Generate Windows charts
    visualise_v2_exp1_absolute(OS_WINDOWS)
    visualise_v2_exp1_overhead(OS_WINDOWS)
    visualise_v2_exp1_overhead_comparison(OS_WINDOWS)
    visualise_v2_exp1_speedup(OS_WINDOWS)
    visualise_v2_exp1_speedup(OS_WINDOWS, metric="iteration")
    visualise_v2_absolute_sync(OS_WINDOWS)
    visualise_v2_speedup_sync(OS_WINDOWS)
    visualise_v2_absolute_compile(OS_WINDOWS)
    
    # Generate OS comparison charts
    visualise_v2_exp1_absolute_os()
    visualise_v2_exp1_speedup_os()
    
    print("Done")


if __name__ == "__main__":
    main()

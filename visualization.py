"""
visualization.py
================
Matplotlib Visualizations for Non-Stationary Multi-Armed Bandit Analysis.

Generates five high-resolution, publication-quality figures:
1. average_reward.png: Average Reward vs Time Steps
2. optimal_actions.png: Percentage of Optimal Actions vs Time Steps
3. cumulative_reward.png: Cumulative Reward Comparison
4. reward_changes.png: Non-Stationary Reward Distribution Walk
5. comparison.png: Performance Comparison Summary Bar Chart
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and scripts
import matplotlib.pyplot as plt


# Palette definitions
COLOR_SAMPLE_AVG = "#D9381E"   # Deep Crimson/Coral
COLOR_CONST_STEP = "#1E88E5"   # Vibrant Royal Blue
COLOR_OPTIMAL_REF = "#2E7D32"  # Forest Green
ARM_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def setup_matplotlib_style() -> None:
    """Configure modern, clean aesthetic defaults for Matplotlib."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAFAFA"
    })


def plot_average_reward(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 1: Average Reward vs Time Steps.
    Compares Sample-Average and Constant Step-Size methods across time steps.
    """
    setup_matplotlib_style()
    steps = results["steps"]
    sa = results["sample_average"]
    cs = results["constant_step"]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Plot raw or smoothed curves
    ax.plot(steps, sa["mean_reward"], label="Sample-Average Method", color=COLOR_SAMPLE_AVG, linewidth=1.8, alpha=0.9)
    ax.plot(steps, cs["mean_reward"], label="Constant Step-Size Method ($\\alpha=0.1$)", color=COLOR_CONST_STEP, linewidth=1.8, alpha=0.95)

    # Shaded error ribbon (1 standard error across runs)
    ax.fill_between(
        steps,
        sa["mean_reward"] - sa["stderr_reward"],
        sa["mean_reward"] + sa["stderr_reward"],
        color=COLOR_SAMPLE_AVG, alpha=0.15
    )
    ax.fill_between(
        steps,
        cs["mean_reward"] - cs["stderr_reward"],
        cs["mean_reward"] + cs["stderr_reward"],
        color=COLOR_CONST_STEP, alpha=0.15
    )

    ax.set_title("Average Reward vs. Time Steps in Non-Stationary Environment", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time Steps", fontsize=12, labelpad=8)
    ax.set_ylabel(f"Average Reward (Averaged over {results['config']['n_runs']} Runs)", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(1, len(steps))

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_optimal_actions(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2: Percentage of Optimal Actions vs Time Steps.
    Demonstrates which method identifies and exploits the optimal arm more frequently.
    """
    setup_matplotlib_style()
    steps = results["steps"]
    sa = results["sample_average"]
    cs = results["constant_step"]
    epsilon = results["config"]["epsilon"]
    n_arms = results["config"]["n_arms"]

    # Theoretical upper bound with epsilon-greedy: (1 - epsilon) + (epsilon / n_arms)
    theoretical_max = ((1.0 - epsilon) + (epsilon / n_arms)) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    ax.plot(steps, sa["pct_optimal"], label="Sample-Average Method", color=COLOR_SAMPLE_AVG, linewidth=1.8, alpha=0.9)
    ax.plot(steps, cs["pct_optimal"], label="Constant Step-Size Method ($\\alpha=0.1$)", color=COLOR_CONST_STEP, linewidth=1.8, alpha=0.95)

    # Reference line for theoretical exploration ceiling
    ax.axhline(
        theoretical_max, color=COLOR_OPTIMAL_REF, linestyle=":", linewidth=1.6,
        label=f"Theoretical Upper Bound (1 - $\\epsilon$ + $\\epsilon/k$ = {theoretical_max:.1f}%)"
    )

    ax.set_title("Percentage of Optimal Actions vs. Time Steps", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time Steps", fontsize=12, labelpad=8)
    ax.set_ylabel("Optimal Action Selections (%)", fontsize=12, labelpad=8)
    ax.set_ylim(0, 100)
    ax.set_xlim(1, len(steps))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower right", fontsize=10.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_cumulative_reward(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3: Cumulative Reward Comparison.
    Illustrates the diverging total rewards earned by both action-value estimation strategies.
    """
    setup_matplotlib_style()
    steps = results["steps"]
    sa_cum = results["sample_average"]["cumulative_reward"]
    cs_cum = results["constant_step"]["cumulative_reward"]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    ax.plot(steps, sa_cum, label="Sample-Average Method", color=COLOR_SAMPLE_AVG, linewidth=2.0)
    ax.plot(steps, cs_cum, label="Constant Step-Size Method ($\\alpha=0.1$)", color=COLOR_CONST_STEP, linewidth=2.0)

    # Shaded difference between curves
    ax.fill_between(steps, sa_cum, cs_cum, color=COLOR_CONST_STEP, alpha=0.12, label="Cumulative Advantage ($\\alpha=0.1$)")

    final_diff = cs_cum[-1] - sa_cum[-1]
    final_pct_gain = (final_diff / sa_cum[-1]) * 100.0 if sa_cum[-1] != 0 else 0.0

    # Annotation of final difference
    ax.annotate(
        f"Advantage: +{final_diff:,.0f} (+{final_pct_gain:.1f}%)",
        xy=(steps[-1], cs_cum[-1]),
        xytext=(steps[-1] * 0.70, cs_cum[-1] * 0.88),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#FBC02D", lw=1.2)
    )

    ax.set_title("Cumulative Reward Comparison over Time", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time Steps", fontsize=12, labelpad=8)
    ax.set_ylabel("Total Cumulative Reward", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_xlim(1, len(steps))

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_reward_changes(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 4: Reward Distribution Changes over Time.
    Visualizes how the true expected rewards q_*(a) of all arms drift via Gaussian random walk.
    """
    setup_matplotlib_style()
    steps = results["steps"]
    q_walk = results["q_walk_run0"]  # Shape: (n_steps, n_arms)
    n_arms = q_walk.shape[1]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    for arm_idx in range(n_arms):
        color = ARM_PALETTE[arm_idx % len(ARM_PALETTE)]
        ax.plot(
            steps,
            q_walk[:, arm_idx],
            label=f"Arm {arm_idx + 1}",
            color=color,
            linewidth=1.4,
            alpha=0.85
        )

    ax.set_title("Reward Distribution Non-Stationarity: Gaussian Random Walk of True Arm Values $q_*(a)$ (Run 1)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time Steps", fontsize=12, labelpad=8)
    ax.set_ylabel("True Expected Reward $q_*(a)$", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, ncol=1, frameon=True)
    ax.set_xlim(1, len(steps))

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_performance_comparison(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 5: Performance Comparison Summary Bar Chart.
    Two side-by-side subplots comparing overall and final phase metrics:
    - Left: Average Reward (Overall vs Final Phase)
    - Right: % Optimal Action (Overall vs Final Phase)
    """
    setup_matplotlib_style()
    sa_sum = results["summary"]["sample_average"]
    cs_sum = results["summary"]["constant_step"]

    fig, (ax_r, ax_opt) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)

    bar_width = 0.32
    x_indices = np.arange(2)  # 0: Overall, 1: Final Phase

    # ------------------
    # Subplot 1: Average Reward
    # ------------------
    sa_rewards = [sa_sum["overall_avg_reward"], sa_sum["final_phase_avg_reward"]]
    cs_rewards = [cs_sum["overall_avg_reward"], cs_sum["final_phase_avg_reward"]]

    rects1 = ax_r.bar(x_indices - bar_width / 2, sa_rewards, bar_width, label="Sample-Average",
                      color=COLOR_SAMPLE_AVG, edgecolor="#333333", linewidth=0.8, alpha=0.9)
    rects2 = ax_r.bar(x_indices + bar_width / 2, cs_rewards, bar_width, label="Constant Step-Size ($\\alpha=0.1$)",
                      color=COLOR_CONST_STEP, edgecolor="#333333", linewidth=0.8, alpha=0.9)

    ax_r.set_title("Average Reward Comparison", fontsize=12.5, fontweight="bold", pad=10)
    ax_r.set_xticks(x_indices)
    ax_r.set_xticklabels(["Overall Simulation", "Final Phase (Last 10%)"], fontsize=10.5)
    ax_r.set_ylabel("Average Reward", fontsize=11)
    ax_r.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax_r.legend(loc="upper left", fontsize=9.5)

    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax_r.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                      xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax_r.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                      xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # ------------------
    # Subplot 2: % Optimal Action
    # ------------------
    sa_opts = [sa_sum["overall_optimal_pct"], sa_sum["final_phase_optimal_pct"]]
    cs_opts = [cs_sum["overall_optimal_pct"], cs_sum["final_phase_optimal_pct"]]

    rects3 = ax_opt.bar(x_indices - bar_width / 2, sa_opts, bar_width, label="Sample-Average",
                        color=COLOR_SAMPLE_AVG, edgecolor="#333333", linewidth=0.8, alpha=0.9)
    rects4 = ax_opt.bar(x_indices + bar_width / 2, cs_opts, bar_width, label="Constant Step-Size ($\\alpha=0.1$)",
                        color=COLOR_CONST_STEP, edgecolor="#333333", linewidth=0.8, alpha=0.9)

    ax_opt.set_title("Optimal Action Selection (%) Comparison", fontsize=12.5, fontweight="bold", pad=10)
    ax_opt.set_xticks(x_indices)
    ax_opt.set_xticklabels(["Overall Simulation", "Final Phase (Last 10%)"], fontsize=10.5)
    ax_opt.set_ylabel("Optimal Action (%)", fontsize=11)
    ax_opt.set_ylim(0, 100)
    ax_opt.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax_opt.legend(loc="upper left", fontsize=9.5)

    # Value labels on bars
    for rect in rects3:
        h = rect.get_height()
        ax_opt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax_opt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    fig.suptitle("Performance Comparison Bar Chart: Non-Stationary Multi-Armed Bandit",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def generate_all_plots(results: Dict[str, Any], output_dir: str) -> Dict[str, plt.Figure]:
    """
    Generate and save all 5 required figures to the specified directory.

    Args:
        results: Dictionary returned by run_experiment.
        output_dir: Folder to store the generated PNG files.

    Returns:
        dict: Mapping from graph identifier to matplotlib Figure objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    p1 = os.path.join(output_dir, "average_reward.png")
    p2 = os.path.join(output_dir, "optimal_actions.png")
    p3 = os.path.join(output_dir, "cumulative_reward.png")
    p4 = os.path.join(output_dir, "reward_changes.png")
    p5 = os.path.join(output_dir, "comparison.png")

    fig1 = plot_average_reward(results, p1)
    fig2 = plot_optimal_actions(results, p2)
    fig3 = plot_cumulative_reward(results, p3)
    fig4 = plot_reward_changes(results, p4)
    fig5 = plot_performance_comparison(results, p5)

    return {
        "average_reward": fig1,
        "optimal_actions": fig2,
        "cumulative_reward": fig3,
        "reward_changes": fig4,
        "comparison": fig5
    }


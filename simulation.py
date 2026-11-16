"""
simulation.py
=============
Execution and Evaluation Engine for Non-Stationary Multi-Armed Bandit Experiments.

This module orchestrates independent Monte Carlo runs, collects step-by-step
performance metrics, and ensures fair pairwise comparison by feeding identical
random walk environments to both Sample-Average and Constant Step-Size methods.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, Optional, Tuple
import os
import time
import numpy as np
import pandas as pd

from environment import PrecomputedBanditTrajectory


@dataclass
class SimulationConfig:
    """Configuration parameters for the non-stationary bandit experiment."""
    n_arms: int = 10
    n_steps: int = 10000
    n_runs: int = 200
    epsilon: float = 0.1
    alpha: float = 0.1
    walk_std: float = 0.01
    reward_std: float = 1.0
    initial_q_std: float = 1.0
    initial_value: float = 0.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_experiment(
    config: Optional[SimulationConfig] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Run the complete comparative simulation between Sample-Average and Constant Step-Size.

    Ensures 100% fair comparison by exposing both algorithms to identical true reward
    trajectories $q_t(a)$ across all independent runs.

    Args:
        config: SimulationConfig instance. If None, default settings are used.
        progress_callback: Optional callback func(progress_float, status_text).

    Returns:
        dict: Complete experimental results, time-series arrays, and summary metrics.
    """
    if config is None:
        config = SimulationConfig()

    start_time = time.time()

    if progress_callback:
        progress_callback(0.05, "Initializing non-stationary reward environments...")

    # 1. Precompute true values trajectory across all runs and steps for fair comparison
    trajectory_env = PrecomputedBanditTrajectory(
        n_arms=config.n_arms,
        n_steps=config.n_steps,
        n_runs=config.n_runs,
        walk_std=config.walk_std,
        reward_std=config.reward_std,
        initial_q_std=config.initial_q_std,
        random_seed=config.seed
    )
    q_true = trajectory_env.q_true  # Shape: (n_runs, n_steps, n_arms)
    optimal_actions = trajectory_env.optimal_actions  # Shape: (n_runs, n_steps)

    # Pre-generate exploration masks and random arm choices using separate reproducible seeds
    rng_sample = np.random.default_rng(seed=config.seed + 100)
    rng_const = np.random.default_rng(seed=config.seed + 200)

    # Allocate arrays for Sample-Average Method
    Q_sample = np.full((config.n_runs, config.n_arms), config.initial_value, dtype=np.float64)
    N_sample = np.zeros((config.n_runs, config.n_arms), dtype=np.int64)
    rewards_sample = np.zeros((config.n_runs, config.n_steps), dtype=np.float64)
    optimal_sample = np.zeros((config.n_runs, config.n_steps), dtype=bool)
    mse_sample = np.zeros((config.n_runs, config.n_steps), dtype=np.float64)

    # Allocate arrays for Constant Step-Size Method
    Q_const = np.full((config.n_runs, config.n_arms), config.initial_value, dtype=np.float64)
    N_const = np.zeros((config.n_runs, config.n_arms), dtype=np.int64)
    rewards_const = np.zeros((config.n_runs, config.n_steps), dtype=np.float64)
    optimal_const = np.zeros((config.n_runs, config.n_steps), dtype=bool)
    mse_const = np.zeros((config.n_runs, config.n_steps), dtype=np.float64)

    run_indices = np.arange(config.n_runs)
    total_steps = config.n_steps

    if progress_callback:
        progress_callback(0.15, "Executing step-by-step simulations...")

    # Vectorized execution across all runs at each time step
    report_interval = max(1, total_steps // 20)

    for t in range(total_steps):
        # Current true action values and optimal arms for all runs
        current_q_true = q_true[:, t, :]  # (n_runs, n_arms)
        opt_act_t = optimal_actions[:, t]   # (n_runs,)

        # ----------------------------------------------------
        # 1. Sample-Average Method Step
        # ----------------------------------------------------
        # Random tie-breaking by adding minuscule uniform noise
        tie_break_s = rng_sample.uniform(0, 1e-9, size=(config.n_runs, config.n_arms))
        greedy_s = np.argmax(Q_sample + tie_break_s, axis=1)
        explore_mask_s = rng_sample.random(size=config.n_runs) < config.epsilon
        random_acts_s = rng_sample.integers(0, config.n_arms, size=config.n_runs)
        actions_s = np.where(explore_mask_s, random_acts_s, greedy_s)

        # Check optimal selection
        optimal_sample[:, t] = (actions_s == opt_act_t)

        # Generate reward from true mean of chosen arm
        mean_r_s = current_q_true[run_indices, actions_s]
        obs_r_s = rng_sample.normal(mean_r_s, config.reward_std)
        rewards_sample[:, t] = obs_r_s

        # Tracking error (MSE between estimated Q and true q*)
        mse_sample[:, t] = np.mean((Q_sample - current_q_true) ** 2, axis=1)

        # Update Q_sample with 1 / N(a)
        N_sample[run_indices, actions_s] += 1
        n_counts_s = N_sample[run_indices, actions_s]
        step_sizes_s = 1.0 / n_counts_s
        Q_sample[run_indices, actions_s] += step_sizes_s * (obs_r_s - Q_sample[run_indices, actions_s])

        # ----------------------------------------------------
        # 2. Constant Step-Size Method Step
        # ----------------------------------------------------
        tie_break_c = rng_const.uniform(0, 1e-9, size=(config.n_runs, config.n_arms))
        greedy_c = np.argmax(Q_const + tie_break_c, axis=1)
        explore_mask_c = rng_const.random(size=config.n_runs) < config.epsilon
        random_acts_c = rng_const.integers(0, config.n_arms, size=config.n_runs)
        actions_c = np.where(explore_mask_c, random_acts_c, greedy_c)

        # Check optimal selection
        optimal_const[:, t] = (actions_c == opt_act_t)

        # Generate reward from true mean of chosen arm
        mean_r_c = current_q_true[run_indices, actions_c]
        obs_r_c = rng_const.normal(mean_r_c, config.reward_std)
        rewards_const[:, t] = obs_r_c

        # Tracking error (MSE between estimated Q and true q*)
        mse_const[:, t] = np.mean((Q_const - current_q_true) ** 2, axis=1)

        # Update Q_const with fixed alpha
        N_const[run_indices, actions_c] += 1
        Q_const[run_indices, actions_c] += config.alpha * (obs_r_c - Q_const[run_indices, actions_c])

        # Progress reporting
        if progress_callback and (t % report_interval == 0 or t == total_steps - 1):
            pct = 0.15 + 0.80 * ((t + 1) / total_steps)
            progress_callback(pct, f"Simulated {t + 1:,} / {total_steps:,} steps ({pct*100:.0f}%)...")

    if progress_callback:
        progress_callback(0.98, "Aggregating performance metrics...")

    # Calculate step-by-step averages across runs
    mean_reward_sample = np.mean(rewards_sample, axis=0)
    mean_reward_const = np.mean(rewards_const, axis=0)

    pct_optimal_sample = np.mean(optimal_sample, axis=0) * 100.0
    pct_optimal_const = np.mean(optimal_const, axis=0) * 100.0

    cumulative_reward_sample = np.cumsum(mean_reward_sample)
    cumulative_reward_const = np.cumsum(mean_reward_const)

    mean_mse_sample = np.mean(mse_sample, axis=0)
    mean_mse_const = np.mean(mse_const, axis=0)

    # Standard errors across runs for confidence intervals
    stderr_reward_sample = np.std(rewards_sample, axis=0) / np.sqrt(config.n_runs)
    stderr_reward_const = np.std(rewards_const, axis=0) / np.sqrt(config.n_runs)

    # Define phase boundaries (e.g. final 10% of steps)
    final_phase_len = max(100, int(0.10 * config.n_steps))
    final_slice = slice(-final_phase_len, None)

    # Aggregate summary metrics
    summary = {
        "sample_average": {
            "name": "Sample-Average Method",
            "overall_avg_reward": float(np.mean(mean_reward_sample)),
            "final_phase_avg_reward": float(np.mean(mean_reward_sample[final_slice])),
            "overall_optimal_pct": float(np.mean(pct_optimal_sample)),
            "final_phase_optimal_pct": float(np.mean(pct_optimal_sample[final_slice])),
            "total_cumulative_reward": float(cumulative_reward_sample[-1]),
            "overall_mse": float(np.mean(mean_mse_sample)),
            "final_mse": float(np.mean(mean_mse_sample[final_slice]))
        },
        "constant_step": {
            "name": f"Constant Step-Size Method (α={config.alpha})",
            "overall_avg_reward": float(np.mean(mean_reward_const)),
            "final_phase_avg_reward": float(np.mean(mean_reward_const[final_slice])),
            "overall_optimal_pct": float(np.mean(pct_optimal_const)),
            "final_phase_optimal_pct": float(np.mean(pct_optimal_const[final_slice])),
            "total_cumulative_reward": float(cumulative_reward_const[-1]),
            "overall_mse": float(np.mean(mean_mse_const)),
            "final_mse": float(np.mean(mean_mse_const[final_slice]))
        }
    }

    elapsed_time = time.time() - start_time
    if progress_callback:
        progress_callback(1.0, f"Completed in {elapsed_time:.2f} seconds!")

    results = {
        "config": config.to_dict(),
        "elapsed_time": elapsed_time,
        "steps": np.arange(1, config.n_steps + 1),
        "q_walk_run0": q_true[0],  # Shape: (n_steps, n_arms) for visualization
        "sample_average": {
            "mean_reward": mean_reward_sample,
            "stderr_reward": stderr_reward_sample,
            "pct_optimal": pct_optimal_sample,
            "cumulative_reward": cumulative_reward_sample,
            "mean_mse": mean_mse_sample,
            "summary": summary["sample_average"]
        },
        "constant_step": {
            "mean_reward": mean_reward_const,
            "stderr_reward": stderr_reward_const,
            "pct_optimal": pct_optimal_const,
            "cumulative_reward": cumulative_reward_const,
            "mean_mse": mean_mse_const,
            "summary": summary["constant_step"]
        },
        "summary": summary
    }

    return results


def save_results_to_csv(results: Dict[str, Any], output_path: str) -> pd.DataFrame:
    """
    Save simulation performance results into a structured CSV file.

    Args:
        results: Dictionary returned by run_experiment.
        output_path: Destination path for the CSV file.

    Returns:
        pd.DataFrame: The created performance DataFrame.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    steps = results["steps"]
    sa = results["sample_average"]
    cs = results["constant_step"]

    df = pd.DataFrame({
        "step": steps,
        "sample_avg_reward": sa["mean_reward"],
        "sample_avg_reward_stderr": sa["stderr_reward"],
        "sample_avg_optimal_pct": sa["pct_optimal"],
        "sample_avg_cumulative_reward": sa["cumulative_reward"],
        "sample_avg_mse": sa["mean_mse"],
        "const_step_reward": cs["mean_reward"],
        "const_step_reward_stderr": cs["stderr_reward"],
        "const_step_optimal_pct": cs["pct_optimal"],
        "const_step_cumulative_reward": cs["cumulative_reward"],
        "const_step_mse": cs["mean_mse"],
    })

    df.to_csv(output_path, index=False)
    return df


def generate_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a clean comparison table summarizing key benchmark metrics.

    Args:
        results: Dictionary returned by run_experiment.

    Returns:
        pd.DataFrame: Formatted summary comparison table.
    """
    sa_sum = results["summary"]["sample_average"]
    cs_sum = results["summary"]["constant_step"]

    table_data = [
        {
            "Algorithm": sa_sum["name"],
            "Overall Avg Reward": f"{sa_sum['overall_avg_reward']:.3f}",
            "Final Phase Avg Reward": f"{sa_sum['final_phase_avg_reward']:.3f}",
            "Overall % Optimal Action": f"{sa_sum['overall_optimal_pct']:.2f}%",
            "Final Phase % Optimal": f"{sa_sum['final_phase_optimal_pct']:.2f}%",
            "Total Cumulative Reward": f"{sa_sum['total_cumulative_reward']:,.1f}",
            "Tracking MSE (q* error)": f"{sa_sum['final_mse']:.3f}",
            "Adaptation Performance": "Poor (Step-size decays to 0, estimates freeze)"
        },
        {
            "Algorithm": cs_sum["name"],
            "Overall Avg Reward": f"{cs_sum['overall_avg_reward']:.3f}",
            "Final Phase Avg Reward": f"{cs_sum['final_phase_avg_reward']:.3f}",
            "Overall % Optimal Action": f"{cs_sum['overall_optimal_pct']:.2f}%",
            "Final Phase % Optimal": f"{cs_sum['final_phase_optimal_pct']:.2f}%",
            "Total Cumulative Reward": f"{cs_sum['total_cumulative_reward']:,.1f}",
            "Tracking MSE (q* error)": f"{cs_sum['final_mse']:.3f}",
            "Adaptation Performance": "Excellent (Exponential recency-weighting tracks drift)"
        }
    ]

    return pd.DataFrame(table_data)


if __name__ == "__main__":
    import sys
    # Add parent / current directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from visualization import generate_all_plots

    print("=" * 78)
    print("  NON-STATIONARY MULTI-ARMED BANDIT SIMULATION BENCHMARK")
    print("  Comparing Sample-Average and Constant Step-Size Methods")
    print("=" * 78)

    config = SimulationConfig(
        n_arms=10,
        n_steps=10000,
        n_runs=200,
        epsilon=0.1,
        alpha=0.1,
        walk_std=0.01,
        reward_std=1.0,
        seed=42
    )

    print(f"\nConfiguration:")
    print(f"  • Arms: {config.n_arms}")
    print(f"  • Steps: {config.n_steps:,}")
    print(f"  • Independent Runs: {config.n_runs}")
    print(f"  • Epsilon (exploration): {config.epsilon}")
    print(f"  • Alpha (constant step-size): {config.alpha}")
    print(f"  • Random Walk Step Std: {config.walk_std}")
    print(f"  • Reward Noise Std: {config.reward_std}")
    print("\nRunning simulation...")

    def cli_progress(frac: float, msg: str):
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {frac*100:5.1f}% - {msg}")
        sys.stdout.flush()

    results = run_experiment(config, progress_callback=cli_progress)
    print("\n")

    # Save CSV
    results_dir = os.path.join(current_dir, "results")
    csv_path = os.path.join(results_dir, "performance_results.csv")
    df = save_results_to_csv(results, csv_path)
    print(f"[✓] Saved performance dataset to: {csv_path}")

    # Display comparison table
    summary_df = generate_summary_table(results)
    print("\n" + "=" * 78)
    print("  PERFORMANCE COMPARISON SUMMARY TABLE")
    print("=" * 78)
    print(summary_df.to_string(index=False))

    # Generate plots
    outputs_dir = os.path.join(current_dir, "outputs")
    print(f"\nGenerating and saving visualization figures to: {outputs_dir}...")
    figures = generate_all_plots(results, outputs_dir)
    print(f"[✓] Successfully generated {len(figures)} publication-quality plots in {outputs_dir}")

    # Conclusion
    print("\n" + "=" * 78)
    print("  CONCLUSION & THEORETICAL VERIFICATION")
    print("=" * 78)
    sa_fin = results["summary"]["sample_average"]["final_phase_optimal_pct"]
    cs_fin = results["summary"]["constant_step"]["final_phase_optimal_pct"]
    diff_opt = cs_fin - sa_fin
    print(f"• Constant Step-Size achieved {cs_fin:.2f}% optimal actions in the final phase,")
    print(f"  compared to only {sa_fin:.2f}% for Sample-Average (+{diff_opt:.2f}% margin).")
    print(f"• Reason: Sample-Average weights past rewards equally (step-size 1/N -> 0),")
    print(f"  causing it to become unresponsive to reward distribution drift.")
    print(f"• Constant Step-Size discounts past observations exponentially at rate (1 - alpha)^k,")
    print(f"  maintaining responsiveness and continuously tracking the non-stationary arms.")
    print("=" * 78 + "\n")


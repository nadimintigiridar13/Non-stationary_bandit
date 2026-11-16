"""
app.py
======
Interactive Streamlit Dashboard for Non-Stationary Multi-Armed Bandit Analysis.
Comparing Sample-Average and Constant Step-Size Action-Value Estimation Methods.
"""

import sys
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local imports resolve
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from simulation import (
    SimulationConfig,
    run_experiment,
    save_results_to_csv,
    generate_summary_table
)
from visualization import (
    plot_average_reward,
    plot_optimal_actions,
    plot_cumulative_reward,
    plot_reward_changes,
    plot_performance_comparison
)


# Page configuration
st.set_page_config(
    page_title="Non-Stationary Multi-Armed Bandit Benchmark",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-sample {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-const {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_default_benchmark_results():
    """Run or load default benchmark (10 arms, 10,000 steps, 200 runs)."""
    default_cfg = SimulationConfig(
        n_arms=10,
        n_steps=10000,
        n_runs=200,
        epsilon=0.1,
        alpha=0.1,
        walk_std=0.01,
        reward_std=1.0,
        seed=42
    )
    return run_experiment(default_cfg)


# Initialize Session State
if "results" not in st.session_state:
    with st.spinner("Initializing benchmark dataset (10,000 steps, 200 runs)..."):
        st.session_state.results = load_default_benchmark_results()


# ==========================================
# SIDEBAR: Simulation Settings
# ==========================================
with st.sidebar:
    st.header("⚙️ Simulation Settings")

    preset = st.selectbox(
        "Configuration Preset",
        ["Full Benchmark (10,000 steps, 200 runs)",
         "Fast Interactive (2,500 steps, 50 runs)",
         "Quick Demo (1,000 steps, 25 runs)",
         "Custom"],
        index=0
    )

    if preset == "Full Benchmark (10,000 steps, 200 runs)":
        def_arms, def_steps, def_runs = 10, 10000, 200
    elif preset == "Fast Interactive (2,500 steps, 50 runs)":
        def_arms, def_steps, def_runs = 10, 2500, 50
    elif preset == "Quick Demo (1,000 steps, 25 runs)":
        def_arms, def_steps, def_runs = 10, 1000, 25
    else:
        def_arms, def_steps, def_runs = 10, 5000, 100

    n_arms = st.slider("Number of Arms (k)", min_value=2, max_value=20, value=def_arms, step=1)
    n_steps = st.slider("Simulation Steps (T)", min_value=500, max_value=15000, value=def_steps, step=500)
    n_runs = st.slider("Independent Runs (N)", min_value=10, max_value=300, value=def_runs, step=10)

    st.markdown("---")
    st.subheader("Algorithm Hyperparameters")
    epsilon = st.slider("Exploration Rate (ε)", min_value=0.0, max_value=0.5, value=0.10, step=0.01,
                        help="Probability of selecting a random arm (epsilon-greedy).")
    alpha = st.slider("Constant Step-Size (α)", min_value=0.01, max_value=0.50, value=0.10, step=0.01,
                      help="Fixed learning rate used by Constant Step-Size Method.")

    st.markdown("---")
    st.subheader("Environment Dynamics")
    walk_std = st.slider("Random Walk Std Dev (σ_walk)", min_value=0.001, max_value=0.050, value=0.010, step=0.001,
                         format="%.3f", help="Gaussian increment std added to true arm values at each step.")
    reward_std = st.slider("Reward Noise Std Dev (σ_R)", min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                           help="Standard deviation of observed Gaussian reward noise.")
    seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

    st.markdown("---")
    run_btn = st.button("🚀 Run Custom Simulation", use_container_width=True, type="primary")

    if run_btn:
        cfg = SimulationConfig(
            n_arms=n_arms,
            n_steps=n_steps,
            n_runs=n_runs,
            epsilon=epsilon,
            alpha=alpha,
            walk_std=walk_std,
            reward_std=reward_std,
            seed=int(seed)
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(min(1.0, max(0.0, pct)))
            status_text.text(msg)

        res = run_experiment(cfg, progress_callback=update_progress)
        st.session_state.results = res
        status_text.text("Simulation complete!")
        progress_bar.empty()
        st.success(f"Simulation completed in {res['elapsed_time']:.2f}s!")


# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
st.markdown('<div class="main-title">🎰 Non-Stationary Multi-Armed Bandit Benchmark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Rigorous Empirical Comparison: <b>Sample-Average Method</b> vs. <b>Constant Step-Size Method</b> under Gaussian Reward Drift</div>',
    unsafe_allow_html=True
)

tab_home, tab_simulation, tab_comparison, tab_conclusion = st.tabs([
    "🏠 Home & Theoretical Overview",
    "📈 Run Simulation & Visualizations",
    "📊 Detailed Metrics Comparison",
    "💡 Mathematical Conclusion & Analysis"
])

res = st.session_state.results
cfg = res["config"]
sa_sum = res["summary"]["sample_average"]
cs_sum = res["summary"]["constant_step"]


# ==========================================
# TAB 1: HOME & THEORETICAL OVERVIEW
# ==========================================
with tab_home:
    st.header("Project Overview & Problem Formulation")

    col_desc, col_math = st.columns([1.1, 0.9])

    with col_desc:
        st.markdown("""
        ### The Non-Stationary Multi-Armed Bandit Problem
        In standard **stationary** bandit problems, each arm's expected reward $q_*(a) = \\mathbb{E}[R_t \\mid A_t = a]$ remains fixed over time.
        In contrast, real-world reinforcement learning systems (such as financial markets, recommendation systems, and robotics) face **non-stationary** environments where reward distributions continuously evolve.

        Here, we evaluate a **10-armed bandit** where true values wander via independent Gaussian random walks:
        $$q_{t+1}(a) = q_t(a) + \\Delta q_t(a), \\quad \\Delta q_t(a) \\sim \\mathcal{N}(0, \\sigma_{\\text{walk}}^2)$$

        When an arm is pulled, the agent observes:
        $$R_t \\sim \\mathcal{N}(q_t(A_t), \\sigma_{\\text{reward}}^2)$$

        ### Core Research Question
        > **"How do Sample-Average and Constant Step-Size methods differ in their ability to adapt to changing reward distributions in a non-stationary multi-armed bandit environment?"**
        """)

    with col_math:
        st.info("💡 **Key Theoretical Distinction: Weighting of Past Rewards**")
        st.markdown(r"""
        #### 1. Sample-Average Method ($1/n$)
        $$Q_{n+1} = Q_n + \frac{1}{n}\left[R_n - Q_n\right] = \frac{1}{n}\sum_{i=1}^n R_i$$
        - Every observation receives equal weight $\frac{1}{n}$.
        - As $n \to \infty$, step-size $\frac{1}{n} \to 0$.
        - **Consequence:** The agent freezes its estimates and ignores recent environmental drift.

        #### 2. Constant Step-Size Method ($\alpha$)
        $$Q_{n+1} = Q_n + \alpha \left[R_n - Q_n\right]$$
        Expanding recursively:
        $$Q_{n+1} = (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha (1 - \alpha)^{n-i} R_i$$
        - Prioritizes recent observations with geometrically decaying weights $\alpha(1-\alpha)^{n-i}$.
        - **Consequence:** Continuous tracking and active adaptation to non-stationary shifts.
        """)

    st.markdown("---")
    st.subheader("Exponential Recency-Weighting vs. Equal Weighting Decay Curve")

    col_w_plot, col_w_desc = st.columns([1.2, 0.8])
    with col_w_plot:
        # Interactive plot comparing weights of past 40 observations
        k_obs = np.arange(1, 41)
        # Weight given to an observation made 'lag' steps ago
        lags = np.arange(0, 40)
        alpha_val = cfg["alpha"]
        geom_weights = alpha_val * ((1 - alpha_val) ** lags)
        sample_weights_n10 = np.full(40, 1.0 / 10.0)
        sample_weights_n100 = np.full(40, 1.0 / 100.0)
        sample_weights_n1000 = np.full(40, 1.0 / 1000.0)

        fig_w, ax_w = plt.subplots(figsize=(8, 4), dpi=200)
        ax_w.plot(lags, geom_weights, label=f"Constant Step-Size (α={alpha_val}): α(1-α)^lag", color="#1E88E5", linewidth=2.4)
        ax_w.plot(lags, sample_weights_n10, label="Sample-Average (at n=10): 1/10 = 0.10", color="#E65100", linestyle="--", linewidth=1.8)
        ax_w.plot(lags, sample_weights_n100, label="Sample-Average (at n=100): 1/100 = 0.01", color="#D9381E", linestyle="-.", linewidth=1.8)
        ax_w.plot(lags, sample_weights_n1000, label="Sample-Average (at n=1000): 1/1000 = 0.001", color="#7F1D1D", linestyle=":", linewidth=2.0)
        ax_w.set_title("Memory Weight Assigned to an Observation as Time Passes", fontsize=11, fontweight="bold")
        ax_w.set_xlabel("Steps Ago Observation Occurred (Lag)", fontsize=10)
        ax_w.set_ylabel("Weight Assigned to Observation", fontsize=10)
        ax_w.grid(True, linestyle="--", alpha=0.6)
        ax_w.legend(fontsize=8.5)
        fig_w.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    with col_w_desc:
        st.markdown(f"""
        **What this chart shows:**
        - At **lag 0** (most recent observation), the Constant Step-Size method immediately incorporates the latest feedback with weight **$\\alpha = {alpha_val}$**.
        - Older rewards decay smoothly at rate $(1 - \\alpha)^{{\\text{{lag}}}}$. After ~30 steps, their influence has decayed to near zero, freeing the agent from obsolete history.
        - In contrast, the Sample-Average method weights an observation made **5,000 steps ago** with the exact same weight as an observation received **one second ago** ($1/n$). Once $n > 500$, new rewards have virtually zero impact!
        """)


# ==========================================
# TAB 2: RUN SIMULATION & VISUALIZATIONS
# ==========================================
with tab_simulation:
    st.header("Simulation Results & Visualizations")

    # High-level metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Overall Avg Reward",
            value=f"{cs_sum['overall_avg_reward']:.3f}",
            delta=f"+{cs_sum['overall_avg_reward'] - sa_sum['overall_avg_reward']:.3f} vs Sample-Avg"
        )
    with m2:
        st.metric(
            label="Final Phase % Optimal Action",
            value=f"{cs_sum['final_phase_optimal_pct']:.1f}%",
            delta=f"+{cs_sum['final_phase_optimal_pct'] - sa_sum['final_phase_optimal_pct']:.1f}% vs Sample-Avg"
        )
    with m3:
        st.metric(
            label="Total Cumulative Reward",
            value=f"{cs_sum['total_cumulative_reward']:,.0f}",
            delta=f"+{cs_sum['total_cumulative_reward'] - sa_sum['total_cumulative_reward']:,.0f}"
        )
    with m4:
        st.metric(
            label="Tracking Error (MSE to q*)",
            value=f"{cs_sum['final_mse']:.3f}",
            delta=f"{cs_sum['final_mse'] - sa_sum['final_mse']:.3f} (Lower is better)",
            delta_color="inverse"
        )

    st.markdown("---")

    graph_choice = st.selectbox(
        "Select Graph to Inspect:",
        [
            "1. Average Reward vs Time Steps",
            "2. Percentage of Optimal Actions vs Time Steps",
            "3. Cumulative Reward Comparison",
            "4. Non-Stationary Reward Distribution (Random Walk)",
            "5. Performance Comparison Summary Bar Chart",
            "Show All 5 Graphs Together"
        ],
        index=0
    )

    if graph_choice == "1. Average Reward vs Time Steps" or graph_choice == "Show All 5 Graphs Together":
        st.subheader("1. Average Reward vs Time Steps")
        fig1 = plot_average_reward(res)
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption(
            "**Interpretation:** Constant Step-Size maintains a persistently higher average reward. "
            "Because arm values drift upward and downward, Sample-Average gets stuck selecting arms that "
            "were good in the past but have since decayed."
        )

    if graph_choice == "2. Percentage of Optimal Actions vs Time Steps" or graph_choice == "Show All 5 Graphs Together":
        st.subheader("2. Percentage of Optimal Actions vs Time Steps")
        fig2 = plot_optimal_actions(res)
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption(
            "**Interpretation:** Constant Step-Size hovers close to the theoretical exploration ceiling "
            f"($\\approx 1 - \\epsilon + \\epsilon/k = {((1 - cfg['epsilon']) + cfg['epsilon']/cfg['n_arms'])*100:.1f}\\%$). "
            "Sample-Average exhibits a steep continuous degradation as $n \\to \\infty$, dropping from early peaks "
            "down toward random performance."
        )

    if graph_choice == "3. Cumulative Reward Comparison" or graph_choice == "Show All 5 Graphs Together":
        st.subheader("3. Cumulative Reward Comparison")
        fig3 = plot_cumulative_reward(res)
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption(
            "**Interpretation:** The shaded gap highlights the compounding advantage of the Constant Step-Size method. "
            f"Over {cfg['n_steps']:,} steps, Constant Step-Size accumulates significantly more total reward."
        )

    if graph_choice == "4. Non-Stationary Reward Distribution (Random Walk)" or graph_choice == "Show All 5 Graphs Together":
        st.subheader("4. Non-Stationary Reward Distribution (Random Walk of True Arm Values)")
        fig4 = plot_reward_changes(res)
        st.pyplot(fig4)
        plt.close(fig4)
        st.caption(
            "**Interpretation:** Trajectory of the true expected reward $q_*(a)$ for each of the 10 arms during a representative run. "
            "Notice how the lines cross repeatedly: the arm that is best at step 1,000 may be the worst at step 4,000. "
            "Algorithms that cannot unlearn past performance inevitably fail in this environment."
        )

    if graph_choice == "5. Performance Comparison Summary Bar Chart" or graph_choice == "Show All 5 Graphs Together":
        st.subheader("5. Performance Comparison Summary Bar Chart")
        fig5 = plot_performance_comparison(res)
        st.pyplot(fig5)
        plt.close(fig5)
        st.caption(
            "**Interpretation:** Direct side-by-side comparison of overall simulation metrics versus final phase performance. "
            "Notice how the performance gap widens dramatically in the final phase as Sample-Average loses all adaptability."
        )


# ==========================================
# TAB 3: DETAILED METRICS COMPARISON
# ==========================================
with tab_comparison:
    st.header("Detailed Performance Comparison")

    summary_df = generate_summary_table(res)
    st.table(summary_df)

    st.subheader("Phase-by-Phase Breakdown")
    n_total = len(res["steps"])
    phase_slices = {
        "Early Phase (Steps 1 - 1,000)": slice(0, min(1000, n_total)),
        "Mid Phase (Steps 1,001 - 5,000)": slice(min(1000, n_total), min(5000, n_total)),
        "Late Phase (Final 1,000 Steps)": slice(-min(1000, n_total), None)
    }

    phase_rows = []
    for pname, pslice in phase_slices.items():
        sa_r = float(np.mean(res["sample_average"]["mean_reward"][pslice]))
        cs_r = float(np.mean(res["constant_step"]["mean_reward"][pslice]))
        sa_o = float(np.mean(res["sample_average"]["pct_optimal"][pslice]))
        cs_o = float(np.mean(res["constant_step"]["pct_optimal"][pslice]))
        phase_rows.append({
            "Simulation Phase": pname,
            "Sample-Avg Reward": f"{sa_r:.3f}",
            "Const-Step Reward": f"{cs_r:.3f}",
            "Reward Gain": f"{cs_r - sa_r:+.3f}",
            "Sample-Avg % Optimal": f"{sa_o:.1f}%",
            "Const-Step % Optimal": f"{cs_o:.1f}%",
            "% Optimal Gain": f"{cs_o - sa_o:+.1f}%"
        })

    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Download Simulation Dataset")
    csv_bytes = pd.DataFrame({
        "step": res["steps"],
        "sample_avg_reward": res["sample_average"]["mean_reward"],
        "const_step_reward": res["constant_step"]["mean_reward"],
        "sample_avg_optimal_pct": res["sample_average"]["pct_optimal"],
        "const_step_optimal_pct": res["constant_step"]["pct_optimal"],
        "sample_avg_cumulative": res["sample_average"]["cumulative_reward"],
        "const_step_cumulative": res["constant_step"]["cumulative_reward"]
    }).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Performance Results (CSV)",
        data=csv_bytes,
        file_name="performance_results.csv",
        mime="text/csv"
    )


# ==========================================
# TAB 4: MATHEMATICAL CONCLUSION & ANALYSIS
# ==========================================
with tab_conclusion:
    st.header("💡 Experimental Conclusion & Mathematical Analysis")

    opt_diff = cs_sum["final_phase_optimal_pct"] - sa_sum["final_phase_optimal_pct"]
    rew_diff = cs_sum["final_phase_avg_reward"] - sa_sum["final_phase_avg_reward"]

    st.markdown(f"""
    ### 🏆 Winner: Constant Step-Size Method ($\\alpha={cfg['alpha']}$)

    In our controlled {cfg['n_steps']:,}-step benchmark over {cfg['n_runs']} independent runs:
    - **Final Phase Optimal Action Selection:** Constant Step-Size achieved **{cs_sum['final_phase_optimal_pct']:.2f}%**, outperforming Sample-Average (**{sa_sum['final_phase_optimal_pct']:.2f}%**) by **+{opt_diff:.2f}%**.
    - **Final Phase Average Reward:** Constant Step-Size achieved **{cs_sum['final_phase_avg_reward']:.3f}**, surpassing Sample-Average (**{sa_sum['final_phase_avg_reward']:.3f}**) by **+{rew_diff:.3f}**.
    - **Tracking MSE:** Constant Step-Size kept an estimate error of **{cs_sum['final_mse']:.3f}**, compared to **{sa_sum['final_mse']:.3f}** for Sample-Average.
    """)

    st.markdown("---")

    st.markdown(r"""
    ### 🔍 Why Does Sample-Average Fail in Non-Stationary Environments?
    The sample-average method is designed to satisfy the Robbins-Monro convergence conditions:
    1. $\sum_{n=1}^\infty \alpha_n = \infty$ (step sizes are large enough to overcome initial conditions)
    2. $\sum_{n=1}^\infty \alpha_n^2 < \infty$ (step sizes become small enough to assure convergence to a constant value)

    For sample-average, $\alpha_n = \frac{1}{n}$. Because $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6} < \infty$, it is guaranteed to converge to a **fixed constant** mean.

    However, in a **non-stationary** environment, **the target is NOT constant**!
    As $n$ grows:
    $$\alpha_n = \frac{1}{n} \to 0$$
    The learning rate decays to zero, effectively **freezing** the agent's estimates. When an arm that was previously optimal degrades or a previously poor arm drifts upward, the sample-average agent requires an impractically large number of steps to adjust its estimate.

    ---

    ### 🚀 Why Does Constant Step-Size Excel?
    Constant step-size sets $\alpha_n = \alpha$, deliberately violating the second Robbins-Monro condition:
    $$\sum_{n=1}^\infty \alpha^2 = \infty$$
    Because the step-size never shrinks to zero, the estimates never settle down to a fixed value. Instead, they continually fluctuate in response to the most recently received rewards.

    From the recursive expansion:
    $$Q_{n+1} = (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha (1 - \alpha)^{n-i} R_i$$

    Because $(1 - \alpha) < 1$, the weight assigned to reward $R_i$ decays exponentially as it recedes into the past:
    $$w_i = \alpha (1 - \alpha)^{n-i}$$
    This produces an **exponentially recency-weighted average** that effortlessly tracks non-stationary reward distributions.

    ---

    ### 📌 Key Takeaways for Reinforcement Learning Practitioners
    1. **Never use sample-average in dynamic or changing environments:** When the environment is non-stationary (e.g., continuous online learning, shifting user preferences, adversarial bandits), decaying step sizes ($1/n$) degrade performance over time.
    2. **Use constant step sizes for tracking:** A constant step size $\alpha \in (0, 1]$ prioritizes recent rewards and enables persistent adaptation.
    3. **Trade-off between adaptation speed and variance:** A larger $\alpha$ adapts faster to sudden shifts but maintains higher steady-state variance around the true mean. A value of $\alpha \approx 0.1$ provides an effective balance between rapid adaptation and low estimation variance.
    """)

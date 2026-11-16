# Non-Stationary Multi-Armed Bandit: Comparing Sample-Average and Constant Step-Size Methods

An empirical Reinforcement Learning benchmark comparing **Sample-Average** and **Constant Step-Size** action-value estimation methods in a non-stationary 10-armed bandit environment subject to continuous Gaussian random walk reward drift.

---

## 📌 Project Overview

In standard **stationary** multi-armed bandit problems, the true expected reward of each arm $q_*(a)$ is invariant over time. Under stationarity, the Sample-Average method guarantees convergence to true action values by the Law of Large Numbers.

However, real-world reinforcement learning scenarios—such as financial market modeling, dynamic recommender systems, network routing, and continuous control—are inherently **non-stationary**. When true action values drift over time, an agent must prioritize recent observations over distant historical data.

This project investigates the central research question:

> **"How do sample-average and constant step-size methods differ in their ability to adapt to changing reward distributions in a non-stationary multi-armed bandit environment?"**

---

## 📐 Mathematical Formulation

### 1. Environment Dynamics
- **Number of Arms ($k$):** $10$
- **Initial True Values:** $q_1(a) \sim \mathcal{N}(0, 1)$ for $a \in \{0, 1, \dots, 9\}$
- **Non-Stationary Random Walk:** At each time step $t$, every arm's true expected reward evolves according to an independent Gaussian increment:
  $$q_{t+1}(a) = q_t(a) + \Delta q_t(a), \quad \Delta q_t(a) \sim \mathcal{N}(0, \sigma_{\text{walk}}^2)$$
  where $\sigma_{\text{walk}} = 0.01$.
- **Observed Reward Generation:** When arm $A_t$ is selected:
  $$R_t \sim \mathcal{N}(q_t(A_t), \sigma_{\text{reward}}^2), \quad \sigma_{\text{reward}} = 1.0$$

### 2. Action Selection ($\epsilon$-Greedy Policy)
Both methods use an $\epsilon$-greedy policy ($\epsilon = 0.1$) with random tie-breaking:
$$A_t = \begin{cases} \text{random arm from } \{0, \dots, k-1\} & \text{with probability } \epsilon \\ \arg\max_a Q_t(a) \text{ (random tie-break)} & \text{with probability } 1 - \epsilon \end{cases}$$

### 3. Sample-Average Action-Value Update
Estimates action values by computing the arithmetic mean of all rewards observed for arm $a$:
$$Q_{n+1}(a) = Q_n(a) + \frac{1}{N_n(a)} \left[ R_n - Q_n(a) \right] = \frac{1}{N_n(a)} \sum_{i=1}^{N_n(a)} R_i$$
- **Step-Size:** $\alpha_n = \frac{1}{n} \to 0$ as $n \to \infty$.
- **Consequence:** Past observations are weighted equally to recent observations. As action count $N(a)$ increases, step-size shrinks toward zero, effectively **freezing** the agent's estimates and preventing adaptation to environmental changes.

### 4. Constant Step-Size Action-Value Update
Updates action values using a fixed step-size parameter $\alpha \in (0, 1]$ (here $\alpha = 0.1$):
$$Q_{n+1}(a) = Q_n(a) + \alpha \left[ R_n - Q_n(a) \right]$$

Expanding this update rule recursively reveals an **exponential recency-weighted average**:
$$Q_{n+1}(a) = (1 - \alpha)^n Q_1(a) + \sum_{i=1}^n \alpha (1 - \alpha)^{n-i} R_i$$
- The weight assigned to reward $R_i$ received $n - i$ steps ago is $w_i = \alpha (1 - \alpha)^{n-i}$.
- Because $(1 - \alpha) < 1$, the influence of past rewards decays geometrically.
- **Consequence:** The agent continuously forgets obsolete historical rewards and rapidly tracks the drifting action values.

---

## 📂 Project Structure

```
non_stationary_bandit/
│
├── non_stationary_bandit.ipynb # Complete Jupyter Notebook with inline visualizations
├── app.py                      # Interactive Streamlit Web Dashboard
├── environment.py              # NonStationaryBandit & PrecomputedBanditTrajectory
├── algorithms.py               # BanditAlgorithm, SampleAverageAgent, ConstantStepSizeAgent
├── simulation.py               # Monte Carlo simulation engine & CLI runner
├── visualization.py            # High-resolution Matplotlib plotting suite
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation and analysis
│
├── outputs/                    # Generated benchmark figures (300 DPI)
│   ├── average_reward.png      # 1. Average Reward vs Time Steps
│   ├── optimal_actions.png     # 2. % Optimal Action Selection vs Time Steps
│   ├── cumulative_reward.png   # 3. Cumulative Reward Comparison
│   ├── reward_changes.png      # 4. Evolution of True Arm Values q*(a)
│   └── comparison.png          # 5. Summary Performance Bar Chart
│
└── results/                    # Exported experimental datasets
    └── performance_results.csv # Step-by-step metrics across 10,000 steps
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+ (tested on Python 3.12)
- pip package manager

### Installation Steps

1. Clone or navigate to the project directory:
   ```bash
   cd non_stationary_bandit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Project

### 1. Execute Simulation from Command Line (CLI)
To run the full 200-run, 10,000-step benchmark, export `performance_results.csv`, and generate all five publication plots:

```bash
python simulation.py
```

#### CLI Output Preview:
```text
==============================================================================
  NON-STATIONARY MULTI-ARMED BANDIT SIMULATION BENCHMARK
  Comparing Sample-Average and Constant Step-Size Methods
==============================================================================

Configuration:
  • Arms: 10
  • Steps: 10,000
  • Independent Runs: 200
  • Epsilon (exploration): 0.1
  • Alpha (constant step-size): 0.1
  • Random Walk Step Std: 0.01
  • Reward Noise Std: 1.0

Running simulation...
  [██████████████████████████████] 100.0% - Completed in 2.19 seconds!

[✓] Saved performance dataset to: .../results/performance_results.csv

==============================================================================
  PERFORMANCE COMPARISON SUMMARY TABLE
==============================================================================
                        Algorithm Overall Avg Reward Final Phase Avg Reward Overall % Optimal Action Final Phase % Optimal Total Cumulative Reward Tracking MSE (q* error)                                 Adaptation Performance
            Sample-Average Method              1.469                  1.610                   63.52%                58.29%                14,694.9                   0.351         Poor (Step-size decays to 0, estimates freeze)
Constant Step-Size Method (α=0.1)              1.567                  1.809                   77.88%                80.86%                15,665.3                   0.101 Excellent (Exponential recency-weighting tracks drift)
```

---

### 2. Launch Interactive Streamlit Dashboard
Launch the web interface for real-time parameter tuning, interactive plots, and mathematical breakdowns:

```bash
streamlit run app.py
```

The application opens in your default browser at `http://localhost:8501`.

#### Streamlit Dashboard Features:
- **Home & Theoretical Foundations:** Mathematical overview, Robbins-Monro convergence conditions, and interactive memory decay plots.
- **Simulation Settings:** Adjust number of arms, simulation steps, runs, $\epsilon$, $\alpha$, and random walk drift variance with live feedback.
- **Run Simulation:** Real-time progress bar, dynamic metric cards, and plot viewers.
- **Detailed Metrics Comparison:** Phase-by-phase breakdown (Early, Mid, Late) and CSV dataset export.
- **Mathematical Conclusion:** Automatically generated analysis and core insights.

---

## 📊 Experimental Results & Figure Analysis

All experiments were conducted over **200 independent runs** of **10,000 steps** each, using identical random seeds per run to guarantee fair environmental exposure.

### Summary Comparison Table

| Metric | Sample-Average Method | Constant Step-Size ($\alpha=0.1$) | Advantage ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Average Reward** | $1.469$ | **$1.567$** | **$+0.098$** |
| **Final Phase (Last 1k Steps) Reward** | $1.610$ | **$1.809$** | **$+0.199$** |
| **Overall Optimal Action %** | $63.52\%$ | **$77.88\%$** | **$+14.36\%$** |
| **Final Phase Optimal Action %** | $58.29\%$ | **$80.86\%$** | **$+22.57\%$** |
| **Total Cumulative Reward** | $14,694.9$ | **$15,665.3$** | **$+970.4$** |
| **Tracking Error (MSE to $q_*$)** | $0.351$ | **$0.101$** | **$-0.250$ (71% lower error)** |

---

### In-Depth Analysis of Generated Figures

#### 1. Average Reward vs. Time Steps (`outputs/average_reward.png`)
- Early in training ($t < 500$), both algorithms perform similarly while discovering initial reward distributions.
- Over time, as true values wander, the Constant Step-Size method sustains a persistently higher average reward.
- Sample-Average suffers from sluggish adjustments, remaining committed to arms that were profitable hundreds of steps earlier.

#### 2. Percentage of Optimal Actions vs. Time Steps (`outputs/optimal_actions.png`)
- Constant Step-Size rapidly ascends to approximately **$80-82\%$ optimal action selection**, stabilizing near the theoretical exploration bound ($1 - \epsilon + \frac{\epsilon}{k} = 91\%$).
- Sample-Average peaks early (~$68\%$) and then systematically degrades over time down to **$58\%$**, because its step-size $\frac{1}{n}$ approaches zero, preventing it from tracking newly emerging optimal arms.

#### 3. Cumulative Reward Comparison (`outputs/cumulative_reward.png`)
- The cumulative reward curve of Constant Step-Size diverges consistently from Sample-Average.
- Over 10,000 steps, Constant Step-Size nets an extra **+$970.4$ reward units per run**, reflecting significant cumulative efficiency.

#### 4. Reward Distribution Changes (`outputs/reward_changes.png`)
- Tracks the true value trajectories $q_*(a)$ of all 10 arms for a representative run.
- The frequent rank crossovers visually confirm the necessity of continual forgetting and adaptability.

#### 5. Performance Comparison Bar Chart (`outputs/comparison.png`)
- Side-by-side grouped bars clearly distinguish overall simulation averages from final-phase metrics.
- Highlights that the performance gap widens dramatically in later phases (+22.57% optimal action margin in the final 1,000 steps).

---

## 🔬 Mathematical Explanation: Why Constant Step-Size Adapts Better

The mathematical basis lies in the **Robbins-Monro conditions** for stochastic approximation:
$$\sum_{n=1}^\infty \alpha_n = \infty \quad \text{and} \quad \sum_{n=1}^\infty \alpha_n^2 < \infty$$

1. **Why Sample-Average Fails in Non-Stationary Settings:**
   The sample-average step size $\alpha_n = \frac{1}{n}$ satisfies both Robbins-Monro conditions:
   $$\sum_{n=1}^\infty \frac{1}{n} = \infty, \quad \sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6} < \infty$$
   This guarantees convergence to a **stationary** expectation. But in non-stationary problems, the target $q_*(a)$ never settles. Because $\frac{1}{n} \to 0$, newly observed rewards have negligible effect, and the agent becomes permanently biased toward historical values.

2. **Why Constant Step-Size Succeeds:**
   Constant step-size sets $\alpha_n = \alpha = 0.1$, which satisfies the first condition ($\sum \alpha = \infty$) but **deliberately violates** the second ($\sum \alpha^2 = \infty$).
   Because the step size never approaches zero, estimates never freeze. Instead, past rewards decay exponentially:
   $$w_i = \alpha (1 - \alpha)^{n-i}$$
   Observations made 30 steps ago retain less than $5\%$ of their original weight, enabling agile tracking of dynamic reward signals.

---

## 🛠️ Verification & Testing

Unit tests and verification checks confirm:
1. **Identical Environments:** Both agents receive identical random walk trajectories $q_t(a)$ in each run.
2. **Proper Tie-Breaking:** Epsilon-greedy implementation uses uniform random tie-breaking among maximal action values.
3. **Reproducibility:** Seeded random number generation produces consistent, deterministic results.
4. **Clean Codebase:** Zero warnings, vectorized NumPy computation (<3 seconds for 2M steps), and clean modular architecture.

---

## 📜 License
MIT License. Built for Reinforcement Learning educational and research demonstration.


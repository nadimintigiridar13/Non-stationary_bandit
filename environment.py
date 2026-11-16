"""
environment.py
==============
Multi-Armed Bandit Environment with Non-Stationary Reward Distributions.

This module implements a 10-armed bandit environment where true expected rewards
undergo a Gaussian random walk at every time step, modeling a non-stationary
reinforcement learning problem (Sutton & Barto, Reinforcement Learning: An Introduction,
Section 2.5).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class NonStationaryBandit:
    """
    A non-stationary Multi-Armed Bandit environment.
    
    In this environment:
    - There are `n_arms` action choices.
    - Each arm $a$ has a true expected reward $q_*(a)$.
    - True expected rewards are initialized randomly from a normal distribution.
    - At each time step, all true expected rewards shift independently according
      to a Gaussian random walk:
          $q_{t+1}(a) = q_t(a) + \\Delta q_t(a)$, where $\\Delta q_t(a) \\sim \\mathcal{N}(0, \\sigma_{\\text{walk}}^2)$
    - When arm $a$ is pulled, the observed reward is sampled from:
          $R_t \\sim \\mathcal{N}(q_t(a), \\sigma_{\\text{reward}}^2)$
    """

    def __init__(
        self,
        n_arms: int = 10,
        walk_std: float = 0.01,
        reward_std: float = 1.0,
        initial_q_std: float = 1.0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize the non-stationary bandit environment.

        Args:
            n_arms: Number of arms/actions available (default: 10).
            walk_std: Standard deviation of Gaussian random walk step (default: 0.01).
            reward_std: Standard deviation of observed reward noise (default: 1.0).
            initial_q_std: Standard deviation for initial arm values (default: 1.0).
            random_seed: Optional integer seed for reproducibility.
        """
        self.n_arms = int(n_arms)
        self.walk_std = float(walk_std)
        self.reward_std = float(reward_std)
        self.initial_q_std = float(initial_q_std)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

        self.true_values: np.ndarray = np.zeros(self.n_arms, dtype=np.float64)
        self.time_step: int = 0
        self.history_q: list = []

        self.reset()

    def reset(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to its initial state with newly sampled true values.

        Args:
            random_seed: Optional new seed for random number generator.

        Returns:
            np.ndarray: Initial true expected reward values of all arms.
        """
        if random_seed is not None:
            self.random_seed = random_seed
            self.rng = np.random.default_rng(seed=random_seed)

        # Sample initial true action values from N(0, initial_q_std^2)
        self.true_values = self.rng.normal(0.0, self.initial_q_std, size=self.n_arms)
        self.time_step = 0
        self.history_q = [self.true_values.copy()]
        return self.true_values.copy()

    def step(self, action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Take an action by pulling arm `action`.

        1. Sample reward from Gaussian distribution around true expected reward.
        2. Identify whether the action was optimal at this time step.
        3. Evolve the true expected rewards of all arms via Gaussian random walk.
        4. Advance time step.

        Args:
            action: Index of the selected arm (0 <= action < n_arms).

        Returns:
            reward (float): Observed scalar reward.
            is_optimal (bool): True if selected arm had highest expected reward.
            info (dict): Auxiliary diagnostics (e.g. current optimal arm, true values).
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Action {action} is invalid for bandit with {self.n_arms} arms.")

        # Determine optimal arm(s) at current time step
        max_q = np.max(self.true_values)
        optimal_arms = np.where(self.true_values == max_q)[0]
        is_optimal = bool(action in optimal_arms)

        # Sample observed reward: R_t ~ N(q_t(a), reward_std^2)
        mean_reward = self.true_values[action]
        reward = float(self.rng.normal(mean_reward, self.reward_std))

        # Store metadata prior to random walk update
        info = {
            "time_step": self.time_step,
            "selected_action": action,
            "true_mean": mean_reward,
            "optimal_action": int(self.rng.choice(optimal_arms)),
            "true_values": self.true_values.copy()
        }

        # Apply Gaussian random walk to update all arm values: q_{t+1}(a) = q_t(a) + N(0, walk_std^2)
        walk_increment = self.rng.normal(0.0, self.walk_std, size=self.n_arms)
        self.true_values = self.true_values + walk_increment
        self.time_step += 1
        self.history_q.append(self.true_values.copy())

        return reward, is_optimal, info

    def get_optimal_action(self) -> int:
        """
        Get the index of the arm currently having the highest true value.
        Ties are broken uniformly at random.
        """
        max_q = np.max(self.true_values)
        optimal_arms = np.where(self.true_values == max_q)[0]
        return int(self.rng.choice(optimal_arms))

    def get_true_values(self) -> np.ndarray:
        """Return a copy of the current true expected rewards."""
        return self.true_values.copy()

    def get_history(self) -> np.ndarray:
        """Return history of true action values over time as array of shape (steps+1, n_arms)."""
        return np.array(self.history_q)


class PrecomputedBanditTrajectory:
    """
    Generates and stores the complete trajectory of true action values $q_t(a)$
    for multiple runs and time steps simultaneously.

    This ensures that:
    1. Both algorithms experience 100% IDENTICAL environments in every run.
    2. Vectorized multi-run simulations run in seconds rather than minutes.
    """

    def __init__(
        self,
        n_arms: int = 10,
        n_steps: int = 10000,
        n_runs: int = 200,
        walk_std: float = 0.01,
        reward_std: float = 1.0,
        initial_q_std: float = 1.0,
        random_seed: int = 42
    ) -> None:
        self.n_arms = int(n_arms)
        self.n_steps = int(n_steps)
        self.n_runs = int(n_runs)
        self.walk_std = float(walk_std)
        self.reward_std = float(reward_std)
        self.initial_q_std = float(initial_q_std)
        self.random_seed = int(random_seed)

        # Generate complete true values trajectory: shape (n_runs, n_steps, n_arms)
        rng = np.random.default_rng(seed=self.random_seed)

        # Initial values: shape (n_runs, 1, n_arms)
        initial_q = rng.normal(0.0, self.initial_q_std, size=(self.n_runs, 1, self.n_arms))

        # Random walk increments for remaining steps: shape (n_runs, n_steps - 1, n_arms)
        increments = rng.normal(0.0, self.walk_std, size=(self.n_runs, self.n_steps - 1, self.n_arms))

        # Cumulative walk: shape (n_runs, n_steps - 1, n_arms)
        walk = initial_q + np.cumsum(increments, axis=1)

        # Full trajectory q_true: shape (n_runs, n_steps, n_arms)
        self.q_true = np.concatenate([initial_q, walk], axis=1)

        # Optimal action for every run at every step: shape (n_runs, n_steps)
        self.optimal_actions = np.argmax(self.q_true, axis=2)

    def get_run_trajectory(self, run_idx: int) -> np.ndarray:
        """Get trajectory of true values for a specific run: shape (n_steps, n_arms)."""
        return self.q_true[run_idx]


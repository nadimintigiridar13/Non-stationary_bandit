"""
algorithms.py
=============
Action-Value Estimation Methods for Multi-Armed Bandits.

This module implements two foundational action-value estimation methods:
1. Sample-Average Method: step-size = 1 / N(a)
2. Constant Step-Size Method: step-size = alpha (constant)

Both algorithms use epsilon-greedy action selection with random tie-breaking.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BanditAlgorithm(ABC):
    """
    Abstract Base Class for Multi-Armed Bandit Action-Value Estimation Algorithms.
    
    Attributes:
        n_arms (int): Number of arms in the bandit.
        epsilon (float): Exploration probability (epsilon-greedy).
        initial_value (float): Initial estimate for all action values Q(a).
        name (str): Readable name for plotting and reporting.
        q_values (np.ndarray): Current estimates of action values Q(a).
        action_counts (np.ndarray): Number of times each action was selected N(a).
        total_steps (int): Total number of time steps taken.
    """

    def __init__(
        self,
        n_arms: int = 10,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        name: str = "BaseBanditAlgorithm"
    ) -> None:
        self.n_arms = int(n_arms)
        self.epsilon = float(epsilon)
        self.initial_value = float(initial_value)
        self.name = name

        self.q_values = np.full(self.n_arms, self.initial_value, dtype=np.float64)
        self.action_counts = np.zeros(self.n_arms, dtype=np.int64)
        self.total_steps = 0

    def select_action(self, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an action using epsilon-greedy policy with random tie-breaking.

        - With probability epsilon: select an arm uniformly at random (exploration).
        - With probability 1 - epsilon: select the greedy arm argmax_a Q(a) (exploitation).
          If multiple arms achieve the maximal Q-value, break ties uniformly at random.

        Args:
            rng: Optional numpy Generator for reproducibility.

        Returns:
            int: The index of the selected action.
        """
        generator = rng if rng is not None else np.random.default_rng()

        # Exploration: choose random arm
        if generator.random() < self.epsilon:
            return int(generator.integers(0, self.n_arms))

        # Exploitation: choose greedy arm with uniform random tie-breaking
        max_q = np.max(self.q_values)
        best_arms = np.where(self.q_values == max_q)[0]
        return int(generator.choice(best_arms))

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the action-value estimate for the selected action given observed reward.

        Args:
            action: Index of the action that was selected.
            reward: Scalar reward received from the environment.
        """
        pass

    def reset(self) -> None:
        """Reset estimated values and action counts to their initial states."""
        self.q_values = np.full(self.n_arms, self.initial_value, dtype=np.float64)
        self.action_counts = np.zeros(self.n_arms, dtype=np.int64)
        self.total_steps = 0

    def get_action_values(self) -> np.ndarray:
        """Return a copy of current action value estimates Q(a)."""
        return self.q_values.copy()

    def get_action_counts(self) -> np.ndarray:
        """Return a copy of action selection counts N(a)."""
        return self.action_counts.copy()


class SampleAverageAgent(BanditAlgorithm):
    """
    Sample-Average Action-Value Estimation Method.

    In this method, action values are estimated by averaging all rewards received
    for that action:
        $Q_n(a) = \\frac{1}{N(a)} \\sum_{i=1}^{N(a)} R_i$

    In incremental form:
        $Q(a) \\leftarrow Q(a) + \\frac{1}{N(a)} [R - Q(a)]$

    Mathematical Property:
    - Step-size parameter $\\alpha_k = \\frac{1}{k}$ decays to zero as $k \\to \\infty$.
    - Guarantees convergence to true values by the Law of Large Numbers in
      STATIONARY environments.
    - IN NON-STATIONARY environments: because $\\frac{1}{k} \\to 0$, recent rewards
      receive negligible weight compared to historical rewards. As a result, the agent
      cannot adapt when true action values drift over time.
    """

    def __init__(
        self,
        n_arms: int = 10,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        name: str = "Sample-Average Method"
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            epsilon=epsilon,
            initial_value=initial_value,
            name=name
        )

    def update(self, action: int, reward: float) -> None:
        """
        Increment count and update Q(action) using sample-average step size 1 / N(a).
        """
        self.action_counts[action] += 1
        self.total_steps += 1
        n = self.action_counts[action]

        # Step size = 1 / N(a)
        step_size = 1.0 / n
        self.q_values[action] += step_size * (reward - self.q_values[action])


class ConstantStepSizeAgent(BanditAlgorithm):
    """
    Constant Step-Size Action-Value Estimation Method.

    In this method, action values are updated using a fixed step-size parameter $\\alpha \\in (0, 1]$:
        $Q(a) \\leftarrow Q(a) + \\alpha [R - Q(a)]$

    Mathematical Property (Exponential Recency-Weighted Average):
    Expanding the update rule recursively:
        $Q_{k+1} = \\alpha R_k + (1 - \\alpha) Q_k$
                 $= (1 - \\alpha)^k Q_1 + \\sum_{i=1}^k \\alpha (1 - \\alpha)^{k-i} R_i$

    - The weight assigned to reward $R_i$ observed $k - i$ steps ago is:
          $w_i = \\alpha (1 - \\alpha)^{k-i}$
    - Because $(1 - \\alpha) < 1$, older rewards decay exponentially.
    - Recent rewards are prioritized, allowing the agent to continuously adapt
      to changing reward distributions in NON-STATIONARY environments.
    """

    def __init__(
        self,
        n_arms: int = 10,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        initial_value: float = 0.0,
        name: str = "Constant Step-Size Method ($\\alpha=0.1$)"
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            epsilon=epsilon,
            initial_value=initial_value,
            name=name
        )
        self.alpha = float(alpha)

    def update(self, action: int, reward: float) -> None:
        """
        Increment count and update Q(action) using constant step size alpha.
        """
        self.action_counts[action] += 1
        self.total_steps += 1

        # Constant step size = alpha
        self.q_values[action] += self.alpha * (reward - self.q_values[action])


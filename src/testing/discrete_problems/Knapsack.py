import numpy as np


class KnapsackProblem:
    """
    0/1 Knapsack Problem instance.

    Parameters
    ----------
    values   : array-like, shape (n_items,)  Profit/value of each item.
    weights  : array-like, shape (n_items,)  Weight of each item.
    capacity : float                          Maximum knapsack capacity.
    """

    def __init__(self, values, weights, capacity):
        self.values   = np.asarray(values,  dtype=float)
        self.weights  = np.asarray(weights, dtype=float)
        self.capacity = float(capacity)
        self.n_items  = len(self.values)

    def fitness(self, x):
        """
        Evaluate a binary solution vector.

        Returns total value if the weight constraint is satisfied, else 0.
        """
        x            = np.asarray(x, dtype=int)
        total_weight = float(x @ self.weights)
        total_value  = float(x @ self.values)
        return total_value if total_weight <= self.capacity else 0.0

    def is_feasible(self, x):
        """Return True iff total weight does not exceed capacity."""
        return float(np.asarray(x, dtype=int) @ self.weights) <= self.capacity

    @classmethod
    def generate(cls, n_items=20, capacity_ratio=0.5, seed=None):
        """
        Create a random knapsack instance.

        Parameters
        ----------
        n_items        : int    Number of items.
        capacity_ratio : float  Capacity as a fraction of total weight.
        seed           : int    Random seed.
        """
        rng     = np.random.default_rng(seed)
        values  = rng.integers(1, 101, size=n_items).astype(float)
        weights = rng.integers(1, 51,  size=n_items).astype(float)
        return cls(values, weights, capacity_ratio * weights.sum())

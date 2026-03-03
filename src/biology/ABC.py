import numpy as np
from matplotlib import pyplot
import os
import sys


class ABC:
    """
    Artificial Bee Colony algorithm for binary (discrete) optimisation.

    Parameters
    ----------
    fitness_func : callable
        Function that maps a binary ndarray of length ``n_dims`` to a
        scalar score.  Higher is better (maximisation).
    n_dims       : int
        Dimensionality of the binary search space (number of bits).
    n_bees       : int
        Number of employed bees (= number of food sources = number of
        onlooker bees).
    max_iter     : int
        Maximum number of foraging cycles.
    limit        : int or None
        Number of unsuccessful trials before a source is abandoned.
    """

    def __init__(self, fitness_func, n_dims, n_bees=30, max_iter=300, limit=None):
        self.fitness_func = fitness_func
        self.n_dims       = n_dims
        self.n_bees       = n_bees
        self.max_iter     = max_iter
        self.limit        = limit if limit is not None else n_bees * n_dims

        self.best_solution = None
        self.best_score    = -np.inf
        self.history       = []

    def _random_source(self):
        """Generate a random binary solution vector (food source)."""
        return np.random.randint(0, 2, size=self.n_dims)

    def _neighbour(self, source):
        """
        Flip a single random bit of the source.

        This is the standard employed / onlooker bee exploitation step.
        """
        neighbour      = source.copy()
        idx            = np.random.randint(self.n_dims)
        neighbour[idx] = 1 - neighbour[idx]
        return neighbour

    def _selection_probabilities(self, fitnesses):
        """Roulette-wheel probability proportional to fitness (non-negative)."""
        shifted = fitnesses - fitnesses.min()
        total   = shifted.sum()
        if total == 0:
            return np.ones(len(fitnesses)) / len(fitnesses)
        return shifted / total

    def run(self, verbose=True):
        """
        Execute the ABC algorithm.

        Parameters
        ----------
        verbose : bool
            If True, print a line whenever the global best improves.

        Returns
        -------
        best_solution : ndarray  Best binary solution vector found.
        best_score    : float    Corresponding fitness value.
        history       : list     Global best score per iteration.
        """
        sources = np.array([self._random_source() for _ in range(self.n_bees)])
        fits    = np.array([self.fitness_func(s) for s in sources])
        trials  = np.zeros(self.n_bees, dtype=int)

        self.best_solution = sources[np.argmax(fits)].copy()
        self.best_score    = float(fits.max())
        self.history       = []

        for iteration in range(1, self.max_iter + 1):
            for i in range(self.n_bees):
                candidate = self._neighbour(sources[i])
                cand_fit  = self.fitness_func(candidate)
                if cand_fit >= fits[i]:
                    sources[i] = candidate
                    fits[i]    = cand_fit
                    trials[i]  = 0
                else:
                    trials[i] += 1

            probs = self._selection_probabilities(fits)
            for _ in range(self.n_bees):
                i         = np.random.choice(self.n_bees, p=probs)
                candidate = self._neighbour(sources[i])
                cand_fit  = self.fitness_func(candidate)
                if cand_fit >= fits[i]:
                    sources[i] = candidate
                    fits[i]    = cand_fit
                    trials[i]  = 0
                else:
                    trials[i] += 1

            exhausted = np.where(trials >= self.limit)[0]
            for i in exhausted:
                sources[i] = self._random_source()
                fits[i]    = self.fitness_func(sources[i])
                trials[i]  = 0

            iter_best_idx = int(np.argmax(fits))
            iter_best     = float(fits[iter_best_idx])

            if iter_best > self.best_score:
                self.best_score    = iter_best
                self.best_solution = sources[iter_best_idx].copy()
                if verbose:
                    print(
                        "Iteration: %d  Best score: %.4f"
                        % (iteration, self.best_score)
                    )

            self.history.append(self.best_score)

        return self.best_solution, self.best_score, self.history

    def plot(self):
        """Plot the convergence curve (best score per iteration)."""
        if not self.history:
            print("No history to plot. Call run() first.")
            return
        pyplot.plot(self.history, ".-", markersize=3)
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Best Score")
        pyplot.title("Artificial Bee Colony – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    N_ITEMS  = 30
    values   = rng.integers(1, 101, size=N_ITEMS).astype(float)
    weights  = rng.integers(1, 51,  size=N_ITEMS).astype(float)
    capacity = 0.5 * weights.sum()

    def fitness_func(x):
        """0/1 Knapsack fitness: total value if feasible, else 0."""
        x = np.asarray(x, dtype=int)
        return float(x @ values) if float(x @ weights) <= capacity else 0.0

    N_BEES   = 40
    MAX_ITER = 500
    LIMIT    = None

    print("=" * 56)
    print("  ABC – Artificial Bee Colony")
    print("  Problem  : 0/1 Knapsack (%d items)" % N_ITEMS)
    print("  Capacity : %.1f  (50%% of total weight)" % capacity)
    print("=" * 56)

    abc = ABC(
        fitness_func=fitness_func,
        n_dims=N_ITEMS,
        n_bees=N_BEES,
        max_iter=MAX_ITER,
        limit=LIMIT,
    )

    best_sol, best_score, history = abc.run(verbose=True)

    selected_items = np.where(best_sol == 1)[0].tolist()
    total_weight   = float(best_sol @ weights)
    total_value    = float(best_sol @ values)

    print("\nBest solution (selected items) :", selected_items)
    print("Best score (total value)       : %.2f" % best_score)
    print("Total weight                   : %.2f / %.2f" % (total_weight, capacity))
    print("Feasible                       :", total_weight <= capacity)

    abc.plot()

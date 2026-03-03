import numpy as np
from matplotlib import pyplot
import os
import sys


class HillClimbing:
    """
    Hill Climbing Algorithm (Steepest Ascent for minimisation).

    At each iteration the algorithm generates `n_neighbours` random perturbations
    of the current solution and immediately moves to the best one that improves
    the objective.  The search stops when no neighbour is better (local optimum)
    or `max_iter` is reached.

    Parameters
    ----------
    obj_func   : callable
        Objective function to *minimise*.  Signature: f(x: np.ndarray) -> float.
    bounds     : np.ndarray, shape (dim, 2)
        Lower and upper bounds per dimension [[lo, hi], ...].
    dim        : int
        Number of dimensions.
    max_iter   : int
        Maximum number of improvement iterations.
    step_size  : float
        Neighbourhood radius as a fraction of each dimension's range.
    n_neighbours : int
        Number of candidate neighbours evaluated per iteration.
    """

    def __init__(self, obj_func, bounds, dim=2, max_iter=1000, step_size=0.05, n_neighbours=8):
        self.obj_func    = obj_func
        self.bounds      = np.asarray(bounds, dtype=float)
        self.dim         = dim
        self.max_iter    = max_iter
        self.step_size   = step_size
        self.n_neighbours = n_neighbours

        self.best_solution = None
        self.best_cost     = np.inf
        self.history       = []

    def _initialise(self):
        """Draw a random starting point uniformly inside the bounds."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.random.rand(self.dim) * (hi - lo)

    def _neighbours(self, x):
        """Generate `n_neighbours` random perturbations of x, clipped to bounds."""
        lo, hi  = self.bounds[:, 0], self.bounds[:, 1]
        sigma   = self.step_size * (hi - lo)
        deltas  = np.random.randn(self.n_neighbours, self.dim) * sigma
        return np.clip(x + deltas, lo, hi)

    def run(self, verbose=True):

        current_sol  = self._initialise()
        current_cost = self.obj_func(current_sol)

        self.best_solution = current_sol.copy()
        self.best_cost     = current_cost
        self.history       = [current_cost]

        for iteration in range(1, self.max_iter + 1):
            candidates      = self._neighbours(current_sol)
            candidate_costs = np.array([self.obj_func(c) for c in candidates])

            best_idx  = int(np.argmin(candidate_costs))
            best_cost = candidate_costs[best_idx]

            if best_cost >= current_cost:
                # local optimum reached
                break

            current_sol  = candidates[best_idx]
            current_cost = best_cost

            if current_cost < self.best_cost:
                self.best_solution = current_sol.copy()
                self.best_cost     = current_cost
                if verbose:
                    print(
                        "Iteration: %d  f([%s]) = %.6f"
                        % (iteration, np.around(self.best_solution, 5), self.best_cost)
                    )

            self.history.append(self.best_cost)

        return self.best_solution, self.best_cost, self.history

    def plot(self):
        """Plot the convergence curve of the best cost over iterations."""
        if not self.history:
            print("No history to plot. Call run() first.")
            return
        pyplot.plot(self.history, ".-", markersize=2)
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Best Cost")
        pyplot.title("Hill Climbing – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from testing.continous_problems import sphere, rastrigin, ackley

    DIMS   = 10
    BOUNDS = np.array([(-5.12, 5.12)] * DIMS)

    TEST_FUNCS = [
        ("Sphere",    sphere,    np.array([(-5.0,    5.0)]    * DIMS)),
        ("Rastrigin", rastrigin, np.array([(-5.12,   5.12)]   * DIMS)),
        ("Ackley",    ackley,    np.array([(-32.768, 32.768)] * DIMS)),
    ]

    for name, func, bounds in TEST_FUNCS:
        print("\n" + "=" * 56)
        print("  Hill Climbing – Steepest Ascent (Minimisation)")
        print("  Function : %s  (%d-D)" % (name, DIMS))
        print("=" * 56)

        hc = HillClimbing(
            obj_func=func,
            bounds=bounds,
            dim=DIMS,
            max_iter=1000,
            step_size=0.05,
            n_neighbours=8,
        )

        best_sol, best_cost, history = hc.run(verbose=True)

        print("\nBest solution : f([%s])" % np.around(best_sol, decimals=5))
        print("Best cost     : %.6f" % best_cost)
        hc.plot()

import numpy as np
from matplotlib import pyplot
import os
import sys


class FA:
    """
    Firefly Algorithm for continuous minimisation problems.

    Parameters
    ----------
    obj_func     : callable
        Objective function to *minimise*.  Signature: f(x: ndarray) -> float.
    bounds       : ndarray, shape (n_dims, 2)
        Lower and upper search bounds [[lo, hi], ...].
    n_fireflies  : int
        Population size (number of fireflies).
    max_iter     : int
        Maximum number of iterations.
    alpha        : float
        Randomness scale (step size of the random walk).
    beta0        : float
        Maximum attractiveness at zero distance.
    gamma        : float
        Light absorption coefficient (controls how fast β decreases).
    alpha_decay  : float
        Multiplicative decay of alpha per iteration (set to 1.0 = no decay).
    """

    def __init__(self, obj_func, bounds, n_fireflies=25, max_iter=500, alpha=0.5, beta0=1.0, gamma=1.0, alpha_decay=0.97):
        self.obj_func    = obj_func
        self.bounds      = np.asarray(bounds, dtype=float)
        self.n_dims      = len(self.bounds)
        self.n_fireflies = n_fireflies
        self.max_iter    = max_iter
        self.alpha       = alpha
        self.beta0       = beta0
        self.gamma       = gamma
        self.alpha_decay = alpha_decay

        self.best_position = None
        self.best_score    = np.inf
        self.history       = []

    def _initialise(self):
        """Uniformly sample initial firefly positions inside the bounds."""
        lo, hi    = self.bounds[:, 0], self.bounds[:, 1]
        positions = lo + np.random.rand(self.n_fireflies, self.n_dims) * (hi - lo)
        scores    = np.array([self.obj_func(p) for p in positions])
        return positions, scores

    def _attractiveness(self, r_squared):
        """Gaussian attractiveness β(r) = β₀ * exp(-γ * r²)."""
        return self.beta0 * np.exp(-self.gamma * r_squared)

    def run(self, verbose=True):
        """
        Execute the Firefly Algorithm.

        Parameters
        ----------
        verbose : bool
            If True, print a progress line whenever a new best is found.

        Returns
        -------
        best_position : ndarray  Best solution vector found.
        best_score    : float    Corresponding objective value.
        history       : list     Global best score per iteration.
        """
        lo, hi            = self.bounds[:, 0], self.bounds[:, 1]
        positions, scores = self._initialise()

        self.best_score    = float(scores.min())
        self.best_position = positions[np.argmin(scores)].copy()
        self.history       = []
        alpha              = self.alpha

        for iteration in range(1, self.max_iter + 1):
            new_positions = positions.copy()

            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if scores[j] < scores[i]:
                        diff      = positions[j] - positions[i]
                        r_sq      = float(np.dot(diff, diff))
                        beta      = self._attractiveness(r_sq)
                        epsilon   = np.random.rand(self.n_dims) - 0.5
                        step      = beta * diff + alpha * epsilon
                        candidate = np.clip(new_positions[i] + step, lo, hi)
                        new_positions[i] = candidate

            positions = new_positions
            scores    = np.array([self.obj_func(p) for p in positions])

            alpha *= self.alpha_decay

            iter_best_idx = int(np.argmin(scores))
            iter_best     = float(scores[iter_best_idx])

            if iter_best < self.best_score:
                self.best_score    = iter_best
                self.best_position = positions[iter_best_idx].copy()
                if verbose:
                    print(
                        "Iteration: %d  f([%s]) = %.5f"
                        % (iteration, np.around(self.best_position, 5), self.best_score)
                    )

            self.history.append(self.best_score)

        return self.best_position, self.best_score, self.history

    def plot(self):
        """Plot the convergence curve (global best score per iteration)."""
        if not self.history:
            print("No history to plot. Call run() first.")
            return
        pyplot.plot(self.history, ".-", markersize=2)
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Best Score")
        pyplot.title("Firefly Algorithm – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from testing.continous_problems import sphere, rastrigin, ackley

    DIMS        = 10
    N_FIREFLIES = 30
    MAX_ITER    = 300
    ALPHA       = 0.5
    BETA0       = 1.0
    GAMMA       = 1.0
    ALPHA_DECAY = 0.97

    TEST_FUNCS = [
        ("Sphere",    sphere,    np.array([(-5.0,   5.0)]    * DIMS)),
        ("Rastrigin", rastrigin, np.array([(-5.12,  5.12)]   * DIMS)),
        ("Ackley",    ackley,    np.array([(-32.768, 32.768)] * DIMS)),
    ]

    for name, func, bounds in TEST_FUNCS:
        print("\n" + "=" * 56)
        print("  FA – Firefly Algorithm")
        print("  Function : %s  (%d-D)" % (name, DIMS))
        print("=" * 56)

        fa = FA(
            obj_func=func,
            bounds=bounds,
            n_fireflies=N_FIREFLIES,
            max_iter=MAX_ITER,
            alpha=ALPHA,
            beta0=BETA0,
            gamma=GAMMA,
            alpha_decay=ALPHA_DECAY,
        )

        best_pos, best_score, history = fa.run(verbose=True)

        print("\nBest solution : f([%s])" % np.around(best_pos, decimals=5))
        print("Best score    : %.5f" % best_score)
        fa.plot()

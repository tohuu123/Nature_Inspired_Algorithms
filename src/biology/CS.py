import numpy as np
from matplotlib import pyplot
import os
import sys


class CS:
    """
    Cuckoo Search for continuous minimisation problems.

    Parameters
    ----------
    n_nests    : int
        Number of host nests (population size).
    max_iter   : int
        Maximum number of iterations.
    pa         : float
        Fraction of worst nests abandoned per iteration (alien egg discovery rate).
    alpha      : float
        Step size scale for the Lévy flight.
    beta_levy  : float
        Lévy exponent (1 < β ≤ 2; β=1.5 is typical).
    
    Notes
    -----
    The objective function and bounds are passed to the run() method.
    """

    def __init__(self, n_nests=25, max_iter=500, pa=0.25, alpha=0.01, beta_levy=1.5):
        self.n_nests   = n_nests
        self.max_iter  = max_iter
        self.pa        = pa
        self.alpha     = alpha
        self.beta_levy = beta_levy

        from math import gamma as gamma_func, pi
        num = gamma_func(1 + beta_levy) * np.sin(pi * beta_levy / 2)
        den = gamma_func((1 + beta_levy) / 2) * beta_levy * (2 ** ((beta_levy - 1) / 2))
        self._sigma_u = (num / den) ** (1.0 / beta_levy)

        self.obj_func      = None
        self.bounds        = None
        self.n_dims        = None
        self.best_position = None
        self.best_score    = np.inf
        self.history       = []

    def _initialise(self):
        """Uniformly initialise nest positions inside the bounds."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        nests  = lo + np.random.rand(self.n_nests, self.n_dims) * (hi - lo)
        scores = np.array([self.obj_func(n) for n in nests])
        return nests, scores

    def _levy_step(self):
        """
        Generate a Lévy flight step vector via Mantegna's algorithm.

        Returns an ndarray of shape (n_dims,).
        """
        u    = np.random.randn(self.n_dims) * self._sigma_u
        v    = np.random.randn(self.n_dims)
        step = u / (np.abs(v) ** (1.0 / self.beta_levy))
        return step

    def _abandon_worst(self, nests, scores):
        """
        Replace the worst pa fraction of nests with new random positions.

        Parameters
        ----------
        nests  : ndarray  Current nest positions (modified in place copy).
        scores : ndarray  Corresponding objective values.

        Returns
        -------
        nests  : ndarray  Updated nest positions.
        scores : ndarray  Updated objective values.
        """
        lo, hi    = self.bounds[:, 0], self.bounds[:, 1]
        n_abandon = max(1, int(self.pa * self.n_nests))
        worst_idx = np.argsort(scores)[-n_abandon:]

        for i in worst_idx:
            nests[i]  = lo + np.random.rand(self.n_dims) * (hi - lo)
            scores[i] = self.obj_func(nests[i])

        return nests, scores

    def run(self, obj_func, bounds, verbose=True):
        """
        Execute the Cuckoo Search algorithm.

        Parameters
        ----------
        obj_func : callable
            Objective function to minimise. Signature: f(x: ndarray) -> float.
        bounds : ndarray, shape (n_dims, 2)
            Lower and upper bounds for each dimension [[lo, hi], ...].
        verbose : bool
            If True, print a progress line whenever a new best is found.

        Returns
        -------
        best_position : ndarray  Best solution vector found.
        best_score    : float    Corresponding objective value.
        history       : list     Global best score per iteration.
        """
        self.obj_func = obj_func
        self.bounds   = np.asarray(bounds, dtype=float)
        self.n_dims   = len(self.bounds)
        
        lo, hi        = self.bounds[:, 0], self.bounds[:, 1]
        nests, scores = self._initialise()

        best_idx           = int(np.argmin(scores))
        self.best_position = nests[best_idx].copy()
        self.best_score    = float(scores[best_idx])
        self.history       = []

        for iteration in range(1, self.max_iter + 1):
            for i in range(self.n_nests):
                step      = self.alpha * self._levy_step()
                candidate = np.clip(nests[i] + step, lo, hi)
                cand_score = self.obj_func(candidate)

                j = np.random.randint(self.n_nests)
                if cand_score < scores[j]:
                    nests[j]  = candidate
                    scores[j] = cand_score

            nests, scores = self._abandon_worst(nests, scores)

            iter_best_idx = int(np.argmin(scores))
            iter_best     = float(scores[iter_best_idx])

            if iter_best < self.best_score:
                self.best_score    = iter_best
                self.best_position = nests[iter_best_idx].copy()
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
        pyplot.title("Cuckoo Search – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from testing.continous_problems import sphere, rastrigin, ackley

    DIMS      = 10
    N_NESTS   = 25
    MAX_ITER  = 500
    PA        = 0.25
    ALPHA     = 0.01
    BETA_LEVY = 1.5

    TEST_FUNCS = [
        ("Sphere",    sphere,    np.array([(-5.0,    5.0)]    * DIMS)),
        ("Rastrigin", rastrigin, np.array([(-5.12,   5.12)]   * DIMS)),
        ("Ackley",    ackley,    np.array([(-32.768,  32.768)] * DIMS)),
    ]

    for name, func, bounds in TEST_FUNCS:
        print("\n" + "=" * 56)
        print("  CS – Cuckoo Search")
        print("  Function : %s  (%d-D)" % (name, DIMS))
        print("=" * 56)

        cs = CS(
            n_nests=N_NESTS,
            max_iter=MAX_ITER,
            pa=PA,
            alpha=ALPHA,
            beta_levy=BETA_LEVY,
        )

        best_pos, best_score, history = cs.run(obj_func=func, bounds=bounds, verbose=True)

        print("\nBest solution : f([%s])" % np.around(best_pos, decimals=5))
        print("Best score    : %.5f" % best_score)
        cs.plot()

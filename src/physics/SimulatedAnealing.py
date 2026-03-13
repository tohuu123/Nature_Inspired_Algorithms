import numpy as np
from matplotlib import pyplot
import os
import sys

class SA:
    """
    Simulated Annealing for continuous minimisation problems.

    The neighbourhood is explored by perturbing the current solution with
    Gaussian noise scaled by the current temperature, then clipping to the
    search bounds.

    Parameters
    ----------
    T0         : float
        Initial (starting) temperature.
    T_min      : float
        Stopping temperature; the algorithm halts when T <= T_min.
    max_iter   : int
        Maximum number of iterations regardless of temperature.
    alpha      : float
        Exponential cooling rate (0 < alpha; larger → faster cooling).
    step_scale : float
        Scales the Gaussian perturbation. A larger value explores wider.
    
    Notes
    -----
    The objective function and bounds are passed to the run() method.
    """

    def __init__(self, T0=1000.0, T_min=1e-3, max_iter=10_000, alpha=0.005, step_scale=0.1):
        self.T0         = T0
        self.T_min      = T_min
        self.max_iter   = max_iter
        self.alpha      = alpha
        self.step_scale = step_scale

        self.obj_func      = None
        self.bounds        = None
        self.best_solution = None
        self.best_cost     = np.inf
        self.history       = []

    def _cooling_schedule(self, iteration):
        """Exponential cooling: T(k) = T0 * exp(-alpha * k)."""
        return self.T0 * np.exp(-self.alpha * iteration)

    def _initialise(self):
        """Draw a random starting point uniformly inside the bounds."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.random.rand(len(self.bounds)) * (hi - lo)

    def _neighbour(self, x, T):
        """Generate a neighbouring solution using Gaussian perturbation."""
        lo, hi  = self.bounds[:, 0], self.bounds[:, 1]
        sigma   = self.step_scale * (hi - lo) * (T / self.T0)
        perturb = np.random.randn(len(x)) * sigma
        return np.clip(x + perturb, lo, hi)

    @staticmethod
    def _acceptance_probability(delta_cost, T):
        """Metropolis acceptance probability for a worse solution."""
        # delta_cost > 0 means the neighbour is worse (we are minimising)
        return np.exp(-delta_cost / T)

    def run(self, obj_func, bounds, verbose=True):
        """
        Execute the Simulated Annealing optimisation.

        Parameters
        ----------
        obj_func : callable
            Objective function to minimise. Signature: f(x: np.ndarray) -> float.
        bounds : np.ndarray, shape (n_dims, 2)
            Lower and upper bounds for each dimension [[lo, hi], ...].
        verbose : bool
            If True, print a progress line whenever a new best is found.

        Returns
        -------
        best_solution : np.ndarray  The best solution vector found.
        best_cost     : float       The corresponding objective value.
        history       : list[float] Best cost recorded at every iteration.
        """
        self.obj_func = obj_func
        self.bounds   = np.asarray(bounds, dtype=float)
        
        current_sol  = self._initialise()
        current_cost = self.obj_func(current_sol)

        self.best_solution = current_sol.copy()
        self.best_cost     = current_cost
        self.history       = []

        for iteration in range(1, self.max_iter + 1):
            T = self._cooling_schedule(iteration)
            if T <= self.T_min:
                break

            candidate_sol  = self._neighbour(current_sol, T)
            candidate_cost = self.obj_func(candidate_sol)
            delta          = candidate_cost - current_cost

            if delta < 0 or np.random.rand() < self._acceptance_probability(delta, T):
                current_sol  = candidate_sol
                current_cost = candidate_cost

            if current_cost < self.best_cost:
                self.best_solution = current_sol.copy()
                self.best_cost     = current_cost
                if verbose:
                    print(
                        "Iteration: %d  T: %.5f  f([%s]) = %.5f"
                        % (iteration, T, np.around(self.best_solution, 5), self.best_cost)
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
        pyplot.title("Simulated Annealing – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from testing.continous import sphere, rastrigin, rosenbrock, griewank, ackley

    DIMS     = 2
    BOUNDS   = np.array([(-5.0, 5.0)] * DIMS)
    T0       = 1000.0
    T_MIN    = 1e-4
    MAX_ITER = 20_000
    ALPHA    = 0.005
    STEP     = 0.3

    sa = SA(
        T0=T0,
        T_min=T_MIN,
        max_iter=MAX_ITER,
        alpha=ALPHA,
        step_scale=STEP,
    )

    best_sol, best_cost, history = sa.run(obj_func=rastrigin, bounds=BOUNDS, verbose=True)
    print("\nSolution: f([%s]) = %.5f" % (np.around(best_sol, decimals=5), best_cost))
    sa.plot()

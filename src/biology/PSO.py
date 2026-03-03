import numpy as np
from matplotlib import pyplot
import os
import sys


class PSO:
    """
    Particle Swarm Optimization for continuous minimisation problems.

    Parameters
    ----------
    obj_func    : callable
        Objective function to *minimise*.  Signature: f(x: ndarray) -> float.
    bounds      : ndarray, shape (n_dims, 2)
        Lower and upper bounds for each dimension [[lo, hi], ...].
    n_particles : int
        Number of particles in the swarm.
    max_iter    : int
        Maximum number of iterations.
    w           : float
        Inertia weight (controls momentum of each particle).
    c1          : float
        Cognitive coefficient (attraction toward personal best).
    c2          : float
        Social coefficient (attraction toward global best).
    v_max_ratio : float
        Velocity clipping as a fraction of the search range per dimension.
    """

    def __init__(self, obj_func, bounds, n_particles=30, max_iter=500, w=0.7, c1=1.5, c2=1.5, v_max_ratio=0.2):
        self.obj_func    = obj_func
        self.bounds      = np.asarray(bounds, dtype=float)
        self.n_dims      = len(self.bounds)
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.w           = w
        self.c1          = c1
        self.c2          = c2

        lo, hi     = self.bounds[:, 0], self.bounds[:, 1]
        self.v_max = v_max_ratio * (hi - lo)
        self.v_min = -self.v_max

        self.best_position = None
        self.best_score    = np.inf
        self.history       = []

    def _initialise(self):
        """Uniformly initialise particle positions and zero velocities."""
        lo, hi     = self.bounds[:, 0], self.bounds[:, 1]
        positions  = lo + np.random.rand(self.n_particles, self.n_dims) * (hi - lo)
        velocities = np.zeros((self.n_particles, self.n_dims))
        return positions, velocities

    def _evaluate(self, positions):
        """Evaluate the objective for every particle."""
        return np.array([self.obj_func(p) for p in positions])

    def run(self, verbose=True):
        """
        Execute the PSO algorithm.

        Parameters
        ----------
        verbose : bool
            If True, print a line whenever the global best improves.

        Returns
        -------
        best_position : ndarray  Best solution vector found.
        best_score    : float    Corresponding objective value.
        history       : list     Global best score at every iteration.
        """
        lo, hi                 = self.bounds[:, 0], self.bounds[:, 1]
        positions, velocities  = self._initialise()
        scores                 = self._evaluate(positions)

        p_best_pos   = positions.copy()
        p_best_score = scores.copy()

        g_best_idx         = int(np.argmin(scores))
        self.best_position = positions[g_best_idx].copy()
        self.best_score    = float(scores[g_best_idx])
        self.history       = []

        for iteration in range(1, self.max_iter + 1):
            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            cognitive  = self.c1 * r1 * (p_best_pos - positions)
            social     = self.c2 * r2 * (self.best_position - positions)
            velocities = self.w * velocities + cognitive + social
            velocities = np.clip(velocities, self.v_min, self.v_max)

            positions = positions + velocities
            positions = np.clip(positions, lo, hi)

            scores = self._evaluate(positions)

            improved_mask               = scores < p_best_score
            p_best_pos[improved_mask]   = positions[improved_mask].copy()
            p_best_score[improved_mask] = scores[improved_mask]

            iter_best_idx = int(np.argmin(p_best_score))
            iter_best     = p_best_score[iter_best_idx]

            if iter_best < self.best_score:
                self.best_score    = float(iter_best)
                self.best_position = p_best_pos[iter_best_idx].copy()
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
        pyplot.title("Particle Swarm Optimization – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from testing.continous_problems import sphere, rastrigin, ackley

    DIMS        = 10
    N_PARTICLES = 40
    MAX_ITER    = 500
    W           = 0.7
    C1          = 1.5
    C2          = 1.5

    TEST_FUNCS = [
        ("Sphere",    sphere,    np.array([(-5.0,    5.0)]    * DIMS)),
        ("Rastrigin", rastrigin, np.array([(-5.12,   5.12)]   * DIMS)),
        ("Ackley",    ackley,    np.array([(-32.768, 32.768)] * DIMS)),
    ]

    for name, func, bounds in TEST_FUNCS:
        print("\n" + "=" * 56)
        print("  PSO – Particle Swarm Optimization")
        print("  Function : %s  (%d-D)" % (name, DIMS))
        print("=" * 56)

        pso = PSO(
            obj_func=func,
            bounds=bounds,
            n_particles=N_PARTICLES,
            max_iter=MAX_ITER,
            w=W,
            c1=C1,
            c2=C2,
        )

        best_pos, best_score, history = pso.run(verbose=True)

        print("\nBest solution : f([%s])" % np.around(best_pos, decimals=5))
        print("Best score    : %.5f" % best_score)
        pso.plot()

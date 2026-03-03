import os
import sys
import numpy as np
from matplotlib import pyplot


class DE:
    """
    Differential Evolution for continuous minimisation problems.

    Parameters
    ----------
    bounds   : ndarray (D, 2)  Lower and upper bounds per dimension.
    obj      : callable        Objective function to minimise.
    pop_size : int             Population size (default 10).
    max_iter : int             Maximum number of generations (default 100).
    F        : float           Mutation scaling factor (default 0.5).
    CR       : float           Crossover probability (default 0.7).
    """

    def __init__(self, bounds, obj, pop_size=10, max_iter=100, F=0.5, CR=0.7):
        self.bounds   = np.asarray(bounds, dtype=float)
        self.obj      = obj
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F        = F
        self.CR       = CR

    def run(self, verbose=True):
        """
        Run the DE optimisation loop.

        Returns
        -------
        best_vector : ndarray  Best solution found.
        best_obj    : float    Objective value of the best solution.
        history     : list     Best objective value at each improvement.
        """
        dims   = len(self.bounds)
        bounds = self.bounds

        def _mutation(a, b, c):
            return a + self.F * (b - c)

        def _crossover(mutated, target):
            p = np.random.rand(dims)
            return np.array([mutated[i] if p[i] < self.CR else target[i] for i in range(dims)])

        def _clip(mutated):
            return np.clip(mutated, bounds[:, 0], bounds[:, 1])

        pop     = bounds[:, 0] + np.random.rand(self.pop_size, dims) * (bounds[:, 1] - bounds[:, 0])
        obj_all = [self.obj(ind) for ind in pop]

        best_vector = pop[np.argmin(obj_all)].copy()
        best_obj    = float(np.min(obj_all))
        prev_obj    = best_obj
        history     = []

        for i in range(self.max_iter):
            for j in range(self.pop_size):
                candidates = [c for c in range(self.pop_size) if c != j]
                a, b, c    = pop[np.random.choice(candidates, 3, replace=False)]
                mutated    = _clip(_mutation(a, b, c))
                trial      = _crossover(mutated, pop[j])
                obj_trial  = self.obj(trial)
                if obj_trial < obj_all[j]:
                    pop[j]     = trial
                    obj_all[j] = obj_trial

            gen_best_obj = float(np.min(obj_all))
            if gen_best_obj < prev_obj:
                best_vector = pop[np.argmin(obj_all)].copy()
                prev_obj    = gen_best_obj
                best_obj    = gen_best_obj
                history.append(best_obj)
                if verbose:
                    print("Iteration: %d f([%s]) = %.5f" % (i, np.around(best_vector, decimals=5), best_obj))

        return best_vector, best_obj, history


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from testing.continous_problems.test_functions import ackley

    bounds = np.array([(-5.0, 5.0), (-5.0, 5.0)])

    de = DE(bounds=bounds, obj=ackley, pop_size=10, max_iter=100, F=0.5, CR=0.7)
    best_vector, best_obj, history = de.run(verbose=True)

    print("\nSolution: f([%s]) = %.5f" % (np.around(best_vector, decimals=5), best_obj))
    pyplot.plot(history, ".-")
    pyplot.xlabel("Improvement Number")
    pyplot.ylabel("Objective Value")
    pyplot.title("Differential Evolution – Convergence")
    pyplot.show()

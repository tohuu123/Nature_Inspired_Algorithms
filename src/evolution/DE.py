import os
import sys
import numpy as np
from matplotlib import pyplot


class DE:
    """
    Differential Evolution for continuous minimisation problems.
    """

    def __init__(self, pop_size=10, max_iter=100, F=0.5, CR=0.7):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.bounds = None
        self.func = None

    def run(self, bounds, obj_func, verbose=True):

        self.bounds = np.asarray(bounds, dtype=float)
        self.func = obj_func

        dims = len(self.bounds)
        bounds = self.bounds

        # mutation
        def _mutation(a, b, c):
            return a + self.F * (b - c)

        # crossover
        def _crossover(mutated, target):
            p = np.random.rand(dims)
            return np.array(
                [mutated[i] if p[i] < self.CR else target[i] for i in range(dims)]
            )

        # clip to bounds
        def _clip(v):
            return np.clip(v, bounds[:, 0], bounds[:, 1])

        # initialize population
        pop = bounds[:, 0] + np.random.rand(self.pop_size, dims) * (
            bounds[:, 1] - bounds[:, 0]
        )

        obj_all = [self.func(ind) for ind in pop]

        best_vector = pop[np.argmin(obj_all)].copy()
        best_obj = float(np.min(obj_all))
        prev_obj = best_obj

        history = [best_obj]

        for i in range(self.max_iter):

            for j in range(self.pop_size):

                candidates = [c for c in range(self.pop_size) if c != j]
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]

                mutated = _clip(_mutation(a, b, c))
                trial = _crossover(mutated, pop[j])

                obj_trial = self.func(trial)

                if obj_trial < obj_all[j]:
                    pop[j] = trial
                    obj_all[j] = obj_trial

            gen_best_idx = np.argmin(obj_all)
            gen_best_obj = float(obj_all[gen_best_idx])

            if gen_best_obj < prev_obj:
                best_vector = pop[gen_best_idx].copy()
                best_obj = gen_best_obj
                prev_obj = gen_best_obj
                history.append(best_obj)

                if verbose:
                    print(
                        f"Iteration {i}: f({np.round(best_vector,5)}) = {best_obj:.5f}"
                    )

        return best_vector, best_obj, history



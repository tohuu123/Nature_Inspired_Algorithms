"""
Teaching-Learning-Based Optimization (TLBO)

A population-based metaheuristic inspired by the teaching-learning process
in a classroom. It consists of two phases:
  • Teacher phase  – learners improve by learning from the teacher (best solution)
  • Learner phase  – learners improve by interacting with a random peer

Reference
---------
R. V. Rao, V. J. Savsani, D. P. Vakharia,
"Teaching–learning-based optimization: A novel method for constrained
mechanical design optimization problems", 2011.
"""

import numpy as np
import matplotlib.pyplot as plt


class TLBO:
    """
    Teaching-Learning-Based Optimization for continuous minimisation.

    Parameters
    ----------
    obj_func : callable
        Objective function  f(x: ndarray) -> float  (to minimise).
    bounds   : ndarray, shape (D, 2)
        Lower and upper bounds per dimension.
    pop_size : int
        Number of learners (population size).
    max_iter : int
        Maximum number of iterations (generations).
    """

    def __init__(self, obj_func, bounds, pop_size=30, max_iter=500):
        self.obj_func = obj_func
        self.bounds   = np.asarray(bounds, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim      = len(self.bounds)
        self.history  = []

    # ------------------------------------------------------------------
    def run(self, verbose=True):
        """
        Execute TLBO.

        Returns
        -------
        best_solution : ndarray   Best learner found.
        best_cost     : float     Objective value of the best learner.
        history       : list      Best cost at every iteration.
        """
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        rng = np.random.default_rng()

        # initialise population uniformly within bounds
        pop = lb + rng.random((self.pop_size, self.dim)) * (ub - lb)
        fitness = np.array([self.obj_func(ind) for ind in pop])

        best_idx  = np.argmin(fitness)
        best_sol  = pop[best_idx].copy()
        best_cost = fitness[best_idx]
        self.history = []

        for it in range(1, self.max_iter + 1):
            # ---- Teacher Phase ----
            teacher = pop[np.argmin(fitness)].copy()
            mean_pop = pop.mean(axis=0)
            TF = rng.integers(1, 3)              # teaching factor ∈ {1, 2}
            r  = rng.random(self.dim)

            for i in range(self.pop_size):
                new = pop[i] + r * (teacher - TF * mean_pop)
                new = np.clip(new, lb, ub)
                new_fit = self.obj_func(new)
                if new_fit < fitness[i]:
                    pop[i]     = new
                    fitness[i] = new_fit

            # ---- Learner Phase ----
            for i in range(self.pop_size):
                j = i
                while j == i:
                    j = rng.integers(0, self.pop_size)
                r = rng.random(self.dim)

                if fitness[i] < fitness[j]:
                    new = pop[i] + r * (pop[i] - pop[j])
                else:
                    new = pop[i] + r * (pop[j] - pop[i])

                new = np.clip(new, lb, ub)
                new_fit = self.obj_func(new)
                if new_fit < fitness[i]:
                    pop[i]     = new
                    fitness[i] = new_fit

            # update global best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_cost:
                best_cost = fitness[gen_best_idx]
                best_sol  = pop[gen_best_idx].copy()

            self.history.append(best_cost)
            if verbose and it % max(1, self.max_iter // 10) == 0:
                print(f"TLBO | Iter {it:>5d}/{self.max_iter}  best = {best_cost:.6e}")

        return best_sol, best_cost, self.history

    # ------------------------------------------------------------------
    def plot(self):
        """Plot the convergence curve."""
        if not self.history:
            print("No history — run the algorithm first.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.history, linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Best Cost")
        plt.title("TLBO Convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

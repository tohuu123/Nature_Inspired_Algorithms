import numpy as np
from matplotlib import pyplot


class GA:
    """
    Binary Genetic Algorithm for discrete maximisation problems.

    Parameters
    ----------
    fitness_func    : callable  Function to maximise; receives a binary chromosome.
    chrom_len       : int       Chromosome length (number of genes).
    pop_size        : int       Population size (default 200).
    max_iter        : int       Number of generations (default 100).
    CR              : float     Two-point crossover probability (default 0.9).
    mutation_rate   : float     Per-bit flip probability (default 0.01).
    tournament_size : int       k-way tournament size (default 3).
    elitism_rate    : float     Fraction of elite individuals kept each generation (default 0.1).
    """

    def __init__(self, fitness_func, chrom_len, pop_size=200, max_iter=100, CR=0.9, mutation_rate=0.01, tournament_size=3, elitism_rate=0.1):
        self.fitness_func    = fitness_func
        self.chrom_len       = chrom_len
        self.pop_size        = pop_size
        self.max_iter        = max_iter
        self.CR              = CR
        self.mutation_rate   = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_rate    = elitism_rate

    def run(self, verbose=True):
        """
        Run the GA optimisation loop.

        Returns
        -------
        best_sol : ndarray  Best binary chromosome found.
        best_fit : float    Fitness of the best solution.
        history  : list     Best fitness value recorded each generation.
        """
        pop_size        = self.pop_size
        chrom_len       = self.chrom_len
        elitism_count   = max(1, int(pop_size * self.elitism_rate))
        offspring_size  = pop_size - elitism_count

        def _evaluate(pop):
            return np.array([self.fitness_func(ind) for ind in pop])

        def _tournament_select(pop, fitness):
            parents = []
            for _ in range(offspring_size):
                idx = np.random.choice(pop_size, self.tournament_size, replace=False)
                parents.append(pop[np.argmax(fitness[idx])])
            return np.array(parents)

        def _two_point_crossover(parents):
            offspring = []
            for i in range(0, offspring_size, 2):
                p1, p2 = parents[i], parents[min(i + 1, offspring_size - 1)]
                if np.random.rand() < self.CR:
                    cut1, cut2 = sorted(np.random.choice(chrom_len, 2, replace=False))
                    offspring.extend([
                        np.concatenate([p1[:cut1], p2[cut1:cut2], p1[cut2:]]),
                        np.concatenate([p2[:cut1], p1[cut1:cut2], p2[cut2:]])
                    ])
                else:
                    offspring.extend([p1.copy(), p2.copy()])
            return np.array(offspring[:offspring_size])

        def _bit_flip_mutation(pop):
            mask = np.random.random((offspring_size, chrom_len)) < self.mutation_rate
            return np.where(mask, 1 - pop, pop)

        def _apply_elitism(pop, fitness):
            elite_idx = np.argsort(fitness)[-elitism_count:]
            return pop[elite_idx].copy(), fitness[elite_idx].copy()

        pop      = np.random.randint(0, 2, size=(pop_size, chrom_len))
        best_sol = None
        best_fit = -np.inf
        history  = []

        for i in range(self.max_iter):
            fitness          = _evaluate(pop)
            idx              = np.argmax(fitness)
            if fitness[idx] > best_fit:
                best_fit = fitness[idx]
                best_sol = pop[idx].copy()
            history.append(best_fit)

            elite_pop, _     = _apply_elitism(pop, fitness)
            parents          = _tournament_select(pop, fitness)
            offspring        = _two_point_crossover(parents)
            offspring        = _bit_flip_mutation(offspring)
            pop              = np.vstack([elite_pop, offspring])

            if verbose:
                print("Iteration: %d f([%s]) = %.5f" % (i, best_sol, np.around(best_fit, decimals=5)))

        return best_sol, best_fit, history


if __name__ == "__main__":
    def fitness_func(x):
        return float(np.sum(x))

    ga = GA(
        fitness_func=fitness_func,
        chrom_len=100,
        pop_size=150,
        max_iter=150,
        CR=0.8,
        mutation_rate=0.01,
        tournament_size=3,
        elitism_rate=0.2,
    )
    best_sol, best_fit, history = ga.run(verbose=True)

    print("\nSolution: f([%s]) = %.5f" % (best_sol, np.around(best_fit, decimals=5)))
    pyplot.plot(history, ".-")
    pyplot.xlabel("Generation")
    pyplot.ylabel("Best Fitness")
    pyplot.title("Genetic Algorithm – Convergence")
    pyplot.show()

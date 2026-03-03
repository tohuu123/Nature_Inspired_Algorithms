import numpy as np
from matplotlib import pyplot

class GraphColoring:
    """
    Graph Coloring Problem

    Parameters:
    -----------
    n_vertices : int
        Number of vertices
    edges : list of tuples, optional
        List of edges (i, j)
        If None, random graph will be generated
    edge_probability : float, optional
        Probability of edge between vertices (for random graph)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_vertices, edges=None, edge_probability=0.5, seed=None):
        self.n_vertices = n_vertices
        self.rng = np.random.default_rng(seed)

        if edges is None:
            edges = self._generate_random_edges(edge_probability)

        self.edges = list(edges)
        self.adj_matrix = self._build_adjacency_matrix()

    def _generate_random_edges(self, edge_probability):
        """Generate a random set of edges using the given edge probability."""
        edges = []
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if self.rng.random() < edge_probability:
                    edges.append((i, j))
        return edges

    def _build_adjacency_matrix(self):
        """Build a symmetric adjacency matrix from the edge list."""
        adj = np.zeros((self.n_vertices, self.n_vertices), dtype=int)
        for i, j in self.edges:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    def n_conflicts(self, coloring):
        """Count the number of edges where both endpoints share the same color."""
        coloring = np.asarray(coloring, dtype=int)
        return sum(1 for i, j in self.edges if coloring[i] == coloring[j])

    def is_valid(self, coloring):
        """Return True iff no two adjacent vertices share the same color."""
        return self.n_conflicts(coloring) == 0

    def n_colors_used(self, coloring):
        """Return the number of distinct colors used in the coloring."""
        return int(np.unique(coloring).size)

    def random_coloring(self, n_colors):
        """Return a random coloring using at most n_colors distinct colors."""
        return self.rng.integers(0, n_colors, size=self.n_vertices)

    @classmethod
    def generate(cls, n_vertices=20, edge_probability=0.4, seed=None):
        """
        Generate a random graph coloring instance.

        Parameters
        ----------
        n_vertices       : int   Number of vertices in the graph.
        edge_probability : float Probability that an edge exists between any pair of vertices.
        seed             : int   Random seed for reproducibility.
        """
        return cls(n_vertices=n_vertices, edges=None, edge_probability=edge_probability, seed=seed)


class GraphColoringSolver:
    """
    Solver wrapper for Graph Coloring instances.

    Provides two optimisation strategies as methods:
        solve_sa  –  Simulated Annealing
        solve_ga  –  Genetic Algorithm

    Parameters
    ----------
    problem : GraphColoring   The problem instance to solve.
    n_colors : int            Number of colors to use.
    beta : float              Penalty multiplier for conflict violations (default 1.0).
    """

    def __init__(self, problem, n_colors, beta=1.0):
        self.problem = problem
        self.n_colors = n_colors
        self.beta = beta

    def _fitness(self, coloring):
        """Penalised fitness: number of colors used + beta * number of conflicts."""
        return self.problem.n_colors_used(coloring) + self.beta * self.problem.n_conflicts(coloring)

    def solve_sa(self, T0=100.0, T_min=1e-3, max_iter=10_000, alpha=0.003, verbose=True):
        """
        Solve graph coloring with Simulated Annealing.

        Neighbourhood: reassign a random vertex to a random color.
        Cooling      : exponential  T(k) = T0 * exp(-alpha * k).

        Parameters
        ----------
        T0       : float  Initial temperature.
        T_min    : float  Stopping temperature.
        max_iter : int    Maximum number of iterations.
        alpha    : float  Exponential cooling rate.
        verbose  : bool   Print progress when a new best is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Best fitness value at every iteration.
        """
        def _cooling_schedule(iteration):
            """Exponential cooling: T(k) = T0 * exp(-alpha * k)."""
            return T0 * np.exp(-alpha * iteration)

        def _neighbour(coloring):
            """Reassign a random vertex to a random color."""
            new_coloring = coloring.copy()
            vertex = np.random.randint(self.problem.n_vertices)
            new_coloring[vertex] = np.random.randint(self.n_colors)
            return new_coloring

        current = self.problem.random_coloring(self.n_colors)
        current_fit = self._fitness(current)

        best_coloring = current.copy()
        best_fit = current_fit
        history = []

        for iteration in range(1, max_iter + 1):
            T = _cooling_schedule(iteration)
            if T <= T_min:
                break

            candidate = _neighbour(current)
            candidate_fit = self._fitness(candidate)
            delta = candidate_fit - current_fit

            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current = candidate
                current_fit = candidate_fit

            if current_fit < best_fit:
                best_coloring = current.copy()
                best_fit = current_fit
                if verbose:
                    print(
                        "[SA] Iter: %d  T: %.4f  colors: %d  conflicts: %d  valid: %s"
                        % (iteration, T, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )

            history.append(best_fit)

        return best_coloring, self.problem.n_colors_used(best_coloring), history

    def solve_ga(self, pop_size=100, max_iter=500, CR=0.8, mutation_rate=0.05, tournament_size=3, elitism_rate=0.1, verbose=True):
        """
        Solve graph coloring with a Genetic Algorithm.

        Operators:
            Selection  – tournament selection (minimise fitness)
            Crossover  – uniform crossover
            Mutation   – random gene reassignment
            Elitism    – top `elitism_rate` individuals survive each generation

        Parameters
        ----------
        pop_size        : int    Population size.
        max_iter        : int    Number of generations.
        CR              : float  Crossover probability per gene.
        mutation_rate   : float  Per-individual mutation probability.
        tournament_size : int    Number of candidates in each tournament.
        elitism_rate    : float  Fraction of best individuals kept each generation.
        verbose         : bool   Print progress when a new best is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Best fitness value per generation.
        """
        def _uniform_crossover(p1, p2):
            """Uniform crossover: each gene independently taken from p1 or p2."""
            mask = np.random.rand(len(p1)) < CR
            child = np.where(mask, p1, p2)
            return child

        def _mutate(coloring):
            """Reassign a random vertex to a random color."""
            new_coloring = coloring.copy()
            vertex = np.random.randint(self.problem.n_vertices)
            new_coloring[vertex] = np.random.randint(self.n_colors)
            return new_coloring

        def _tournament_select(population, fitnesses):
            """k-way tournament selection. Returns one parent coloring."""
            idx = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = idx[np.argmin(fitnesses[idx])]
            return population[best_idx].copy()

        population = np.array([self.problem.random_coloring(self.n_colors) for _ in range(pop_size)])
        fitnesses = np.array([self._fitness(c) for c in population])

        best_idx = np.argmin(fitnesses)
        best_coloring = population[best_idx].copy()
        best_fit = fitnesses[best_idx]
        history = []

        elitism_count = max(1, int(pop_size * elitism_rate))
        offspring_size = pop_size - elitism_count

        for generation in range(1, max_iter + 1):
            elite_idx = np.argsort(fitnesses)[:elitism_count]
            elite_pop = population[elite_idx].copy()

            offspring = []
            while len(offspring) < offspring_size:
                p1 = _tournament_select(population, fitnesses)
                p2 = _tournament_select(population, fitnesses)
                child = _uniform_crossover(p1, p2)
                if np.random.rand() < mutation_rate:
                    child = _mutate(child)
                offspring.append(child)

            offspring = np.array(offspring[:offspring_size])
            population = np.vstack([elite_pop, offspring])
            fitnesses = np.array([self._fitness(c) for c in population])

            gen_best_idx = np.argmin(fitnesses)
            gen_best_fit = fitnesses[gen_best_idx]

            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_coloring = population[gen_best_idx].copy()
                if verbose:
                    print(
                        "[GA] Gen: %d  colors: %d  conflicts: %d  valid: %s"
                        % (generation, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )

            history.append(best_fit)

        return best_coloring, self.problem.n_colors_used(best_coloring), history


if __name__ == "__main__":
    N_VERTICES = 20
    N_COLORS = 5
    SEED = 42

    problem = GraphColoring.generate(
        n_vertices=N_VERTICES,
        edge_probability=0.4,
        seed=SEED,
    )

    print("=" * 56)
    print("  Graph Coloring Demo  –  %d vertices" % N_VERTICES)
    print("  Edges       : %d" % len(problem.edges))
    print("  Colors       : %d" % N_COLORS)
    print("=" * 56)

    solver = GraphColoringSolver(problem, n_colors=N_COLORS, beta=1.0)

    print("\n>>> Simulated Annealing")
    sa_coloring, sa_colors, sa_history = solver.solve_sa(
        T0=100.0,
        T_min=1e-3,
        max_iter=15_000,
        alpha=0.003,
        verbose=True,
    )
    print("\n[SA] Best coloring  :", sa_coloring)
    print("[SA] Colors used    : %d" % sa_colors)
    print("[SA] Conflicts      : %d" % problem.n_conflicts(sa_coloring))
    print("[SA] Valid coloring : %s" % problem.is_valid(sa_coloring))

    print("\n>>> Genetic Algorithm")
    ga_coloring, ga_colors, ga_history = solver.solve_ga(
        pop_size=120,
        max_iter=600,
        CR=0.8,
        mutation_rate=0.05,
        tournament_size=3,
        elitism_rate=0.1,
        verbose=True,
    )
    print("\n[GA] Best coloring  :", ga_coloring)
    print("[GA] Colors used    : %d" % ga_colors)
    print("[GA] Conflicts      : %d" % problem.n_conflicts(ga_coloring))
    print("[GA] Valid coloring : %s" % problem.is_valid(ga_coloring))

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sa_history, ".-", markersize=2)
    axes[0].set_title("SA – Convergence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best Fitness")

    axes[1].plot(ga_history, ".-", markersize=2, color="tab:orange")
    axes[1].set_title("GA – Convergence")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Best Fitness")

    pyplot.suptitle("Graph Coloring  –  %d vertices, %d colors" % (N_VERTICES, N_COLORS))
    pyplot.tight_layout()
    pyplot.show()

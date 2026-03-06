import numpy as np
from matplotlib import pyplot


class TSP:
    """
    Travelling Salesman Problem instance with optional time and cost constraints.

    Parameters
    ----------
    dist_matrix  : ndarray (n, n)  Distance between every pair of cities.
    time_matrix  : ndarray (n, n)  Travel time between cities (optional).
    cost_matrix  : ndarray (n, n)  Travel cost per edge        (optional).
    time_limit   : float           Maximum allowed total tour time (optional).
    cost_limit   : float           Maximum allowed total tour cost (optional).
    """

    def __init__(self, dist_matrix, time_matrix=None, cost_matrix=None, time_limit=None, cost_limit=None):
        self.dist_matrix = np.asarray(dist_matrix, dtype=float)
        self.time_matrix = np.asarray(time_matrix, dtype=float) if time_matrix is not None else None
        self.cost_matrix = np.asarray(cost_matrix, dtype=float) if cost_matrix is not None else None
        self.time_limit  = time_limit
        self.cost_limit  = cost_limit
        self.n_cities    = self.dist_matrix.shape[0]

    def total_distance(self, tour):
        """Sum of travel distances along the closed tour."""
        tour = np.asarray(tour, dtype=int)
        return float(
            sum(self.dist_matrix[tour[i], tour[(i + 1) % self.n_cities]]
                for i in range(self.n_cities))
        )

    def total_time(self, tour):
        """Sum of travel times along the closed tour (requires time_matrix)."""
        if self.time_matrix is None:
            return 0.0
        tour = np.asarray(tour, dtype=int)
        return float(
            sum(self.time_matrix[tour[i], tour[(i + 1) % self.n_cities]]
                for i in range(self.n_cities))
        )

    def total_cost(self, tour):
        """Sum of edge costs along the closed tour (requires cost_matrix)."""
        if self.cost_matrix is None:
            return 0.0
        tour = np.asarray(tour, dtype=int)
        return float(
            sum(self.cost_matrix[tour[i], tour[(i + 1) % self.n_cities]]
                for i in range(self.n_cities))
        )

    def is_feasible(self, tour):
        """Return True iff the tour satisfies all active constraints."""
        if self.time_limit is not None and self.total_time(tour) > self.time_limit:
            return False
        if self.cost_limit is not None and self.total_cost(tour) > self.cost_limit:
            return False
        return True

    def constraint_violation(self, tour):
        """Total amount by which the tour exceeds active constraints (≥ 0)."""
        v = 0.0
        if self.time_limit is not None:
            v += max(0.0, self.total_time(tour) - self.time_limit)
        if self.cost_limit is not None:
            v += max(0.0, self.total_cost(tour) - self.cost_limit)
        return v

    def random_tour(self):
        """Return a random permutation of city indices."""
        return np.random.permutation(self.n_cities)

    @classmethod
    def generate(cls, n_cities=15, seed=None, time_limit=None, cost_limit=None):
        """
        Generate a random TSP instance by placing cities on a 2-D plane.

        Distances  : Euclidean distances between random (x, y) coordinates
                     in [0, 100]².
        Time matrix: dist_matrix * U(0.8, 1.2)  (correlated, slight noise).
        Cost matrix: dist_matrix * U(0.5, 1.5)  (correlated, more variance).

        Parameters
        ----------
        n_cities   : int    Number of cities.
        seed       : int    Random seed for reproducibility.
        time_limit : float  If given, passed directly to the TSP instance.
        cost_limit : float  If given, passed directly to the TSP instance.
        """
        rng    = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n_cities, 2))

        diff        = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

        # Correlated time and cost matrices with random noise
        time_matrix = dist_matrix * rng.uniform(0.8, 1.2, size=(n_cities, n_cities))
        cost_matrix = dist_matrix * rng.uniform(0.5, 1.5, size=(n_cities, n_cities))

        time_matrix = (time_matrix + time_matrix.T) / 2
        cost_matrix = (cost_matrix + cost_matrix.T) / 2
        np.fill_diagonal(time_matrix, 0)
        np.fill_diagonal(cost_matrix, 0)

        return cls(
            dist_matrix=dist_matrix,
            time_matrix=time_matrix,
            cost_matrix=cost_matrix,
            time_limit=time_limit,
            cost_limit=cost_limit,
        )


class TSPSolver:
    """
    Solver wrapper for TSP instances.

    Provides two optimisation strategies as methods:
        solve_sa  –  Simulated Annealing with 2-opt neighbourhood
        solve_ga  –  Genetic Algorithm with OX crossover and swap mutation

    Parameters
    ----------
    tsp  : TSP   The problem instance to solve.
    beta : float Penalty multiplier for constraint violations (default 2.0).
    """

    def __init__(self, tsp, beta=2.0):
        self.tsp  = tsp
        self.beta = beta

    def _penalised_distance(self, tour):
        """Distance + beta * constraint_violation (shared by both solvers)."""
        return self.tsp.total_distance(tour) + self.beta * self.tsp.constraint_violation(tour)

    def solve_sa(self, T0=500.0, T_min=1e-3, max_iter=10_000, alpha=0.003, verbose=True):
        """
        Solve the TSP with Simulated Annealing.

        Neighbourhood: 2-opt (reverse a random sub-segment).
        Cooling      : exponential  T(k) = T0 * exp(-alpha * k).

        Parameters
        ----------
        T0        : float  Initial temperature.
        T_min     : float  Stopping temperature.
        max_iter  : int    Maximum number of iterations.
        alpha     : float  Exponential cooling rate.
        verbose   : bool   Print progress when a new best is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Best (penalised) distance at every iteration.
        """
        def _cooling_schedule(iteration):
            """Exponential cooling: T(k) = T0 * exp(-alpha * k)."""
            return T0 * np.exp(-alpha * iteration)

        def _two_opt(tour):
            """2-opt move: reverse a random sub-segment of the tour."""
            n    = len(tour)
            i, j = sorted(np.random.choice(n, 2, replace=False))
            new_tour        = tour.copy()
            new_tour[i:j+1] = tour[i:j+1][::-1]
            return new_tour

        def _acceptance_probability(delta, T):
            """Metropolis criterion for accepting a worse solution."""
            return np.exp(-delta / T)

        current_tour = self.tsp.random_tour()
        current_dist = self._penalised_distance(current_tour)

        best_tour = current_tour.copy()
        best_dist = current_dist
        history   = []

        for iteration in range(1, max_iter + 1):
            T = _cooling_schedule(iteration)
            if T <= T_min:
                break

            candidate      = _two_opt(current_tour)
            candidate_dist = self._penalised_distance(candidate)
            delta          = candidate_dist - current_dist

            if delta < 0 or np.random.rand() < _acceptance_probability(delta, T):
                current_tour = candidate
                current_dist = candidate_dist

            if current_dist < best_dist:
                best_tour = current_tour.copy()
                best_dist = current_dist
                if verbose:
                    print(
                        "[SA] Iter: %d  T: %.4f  dist: %.4f  feasible: %s"
                        % (iteration, T, self.tsp.total_distance(best_tour),
                           self.tsp.is_feasible(best_tour))
                    )

            history.append(best_dist)

        return best_tour, self.tsp.total_distance(best_tour), history

    def solve_ga(self, pop_size=100, max_iter=500, CR=0.9, mutation_rate=0.02, tournament_size=3, elitism_rate=0.1, beta=2.0, verbose=True):
        """
        Solve the TSP with a Genetic Algorithm.

        Operators:
            Selection  – binary tournament (minimise penalised distance)
            Crossover  – Order Crossover (OX)
            Mutation   – swap mutation
            Elitism    – top `elitism_rate` individuals survive each generation

        Parameters
        ----------
        pop_size        : int    Population size.
        max_iter        : int    Number of generations.
        CR              : float  Crossover probability.
        mutation_rate   : float  Per-individual mutation probability.
        tournament_size : int    Number of candidates in each tournament.
        elitism_rate    : float  Fraction of best individuals kept each gen.
        beta            : float  Overrides the instance-level penalty factor.
        verbose         : bool   Print progress when a new best is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Best (penalised) distance per generation.
        """
        self.beta = beta

        def _order_crossover(p1, p2):
            """Order Crossover (OX): preserves the relative order of cities."""
            n           = len(p1)
            i, j        = sorted(np.random.choice(n, 2, replace=False))
            child       = np.full(n, -1, dtype=int)
            child[i:j+1] = p1[i:j+1]
            segment_set  = set(p1[i:j+1])
            fill_vals    = [c for c in p2 if c not in segment_set]
            pos          = list(range(0, i)) + list(range(j+1, n))
            for k, p in enumerate(pos):
                child[p] = fill_vals[k]
            return child

        def _swap_mutation(tour):
            """Swap two random positions in the tour."""
            new_tour = tour.copy()
            i, j     = np.random.choice(len(tour), 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            return new_tour

        def _tournament_select(population, fitnesses):
            """k-way tournament selection. Returns one parent tour."""
            idx      = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = idx[np.argmin(fitnesses[idx])]
            return population[best_idx].copy()

        population = np.array([self.tsp.random_tour() for _ in range(pop_size)])
        fitnesses  = np.array([self._penalised_distance(t) for t in population])

        best_idx  = np.argmin(fitnesses)
        best_tour = population[best_idx].copy()
        best_dist = fitnesses[best_idx]
        history   = []

        elitism_count  = max(1, int(pop_size * elitism_rate))
        offspring_size = pop_size - elitism_count

        for generation in range(1, max_iter + 1):
            elite_idx = np.argsort(fitnesses)[:elitism_count]
            elite_pop = population[elite_idx].copy()

            offspring = []
            while len(offspring) < offspring_size:
                p1    = _tournament_select(population, fitnesses)
                p2    = _tournament_select(population, fitnesses)
                child = _order_crossover(p1, p2) if np.random.rand() < CR else p1.copy()
                if np.random.rand() < mutation_rate:
                    child = _swap_mutation(child)
                offspring.append(child)

            offspring  = np.array(offspring[:offspring_size])
            population = np.vstack([elite_pop, offspring])
            fitnesses  = np.array([self._penalised_distance(t) for t in population])

            gen_best_idx  = np.argmin(fitnesses)
            gen_best_dist = fitnesses[gen_best_idx]

            if gen_best_dist < best_dist:
                best_dist = gen_best_dist
                best_tour = population[gen_best_idx].copy()
                if verbose:
                    print(
                        "[GA] Gen: %d  dist: %.4f  feasible: %s"
                        % (generation, self.tsp.total_distance(best_tour),
                           self.tsp.is_feasible(best_tour))
                    )

            history.append(best_dist)

        return best_tour, self.tsp.total_distance(best_tour), history


if __name__ == "__main__":
    N_CITIES = 15
    SEED     = 42

    tsp = TSP.load_from_file("src/testing/discrete_problems/testcases/tsp_clustered_40.txt")

    print("=" * 56)
    print("  TSP Demo  –  %d cities" % N_CITIES)
    print("  Time limit : %s" % tsp.time_limit)
    print("  Cost limit : %s" % tsp.cost_limit)
    print("=" * 56)
    
    solver = TSPSolver(tsp, beta=2.0)

    print("\n>>> Simulated Annealing")
    sa_tour, sa_dist, sa_history = solver.solve_sa(
        T0=500.0,
        T_min=1e-3,
        max_iter=15_000,
        alpha=0.003,
        verbose=True,
    )
    
    print("\n[SA] Best tour     :", sa_tour)
    print("[SA] Total distance: %.4f" % sa_dist)
    print("[SA] Total time    : %.4f" % tsp.total_time(sa_tour))
    print("[SA] Total cost    : %.4f" % tsp.total_cost(sa_tour))
    print("[SA] Feasible      :", tsp.is_feasible(sa_tour))

    print("\n>>> Genetic Algorithm")
    ga_tour, ga_dist, ga_history = solver.solve_ga(
        pop_size=120,
        max_iter=600,
        CR=0.9,
        mutation_rate=0.05,
        tournament_size=3,
        elitism_rate=0.1,
        beta=2.0,
        verbose=True,
    )
    print("\n[GA] Best tour     :", ga_tour)
    print("[GA] Total distance: %.4f" % ga_dist)
    print("[GA] Total time    : %.4f" % tsp.total_time(ga_tour))
    print("[GA] Total cost    : %.4f" % tsp.total_cost(ga_tour))
    print("[GA] Feasible      :", tsp.is_feasible(ga_tour))

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sa_history, ".-", markersize=2)
    axes[0].set_title("SA – Convergence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best Distance (penalised)")

    axes[1].plot(ga_history, ".-", markersize=2, color="tab:orange")
    axes[1].set_title("GA – Convergence")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Best Distance (penalised)")

    pyplot.suptitle("TSP Solver – %d cities" % N_CITIES)
    pyplot.tight_layout()
    pyplot.show()

import numpy as np
from collections import deque
import heapq
from matplotlib import pyplot
import json

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
        self.coords      = None  # (n_cities, 2) array set by generate(); None for manual instances
    
    def info(self):
        """
        Print information about the TSP instance.
        """

        print("====== TSP Instance Info ======")
        print("Cities:", self.n_cities)
        print("Distance matrix:", self.dist_matrix.shape)

        if self.time_matrix is not None:
            print("Time matrix available")

        if self.cost_matrix is not None:
            print("Cost matrix available")

        if self.time_limit is not None:
            print("Time limit:", self.time_limit)

        if self.cost_limit is not None:
            print("Cost limit:", self.cost_limit)

    def load_from_json(self, filepath):
        """
        Load TSP instance from a JSON file.
        """

        with open(filepath, "r") as f:
            data = json.load(f)

        # Distance matrix (required)
        self.dist_matrix = np.array(data["dist_matrix"], dtype=float)

        # Optional matrices
        self.time_matrix = None
        if data.get("time_matrix") is not None:
            self.time_matrix = np.array(data["time_matrix"], dtype=float)

        self.cost_matrix = None
        if data.get("cost_matrix") is not None:
            self.cost_matrix = np.array(data["cost_matrix"], dtype=float)

        # Optional constraints
        self.time_limit = data.get("time_limit", None)
        self.cost_limit = data.get("cost_limit", None)

        # Number of cities
        self.n_cities = self.dist_matrix.shape[0]


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
        """Total amount by which the tour exceeds active constraints (>= 0)."""
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

        Distances  : Euclidean distances between random (x, y) coordinates in [0, 100]^2.
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

        time_matrix = dist_matrix * rng.uniform(0.8, 1.2, size=(n_cities, n_cities))
        cost_matrix = dist_matrix * rng.uniform(0.5, 1.5, size=(n_cities, n_cities))

        time_matrix = (time_matrix + time_matrix.T) / 2
        cost_matrix = (cost_matrix + cost_matrix.T) / 2
        np.fill_diagonal(time_matrix, 0)
        np.fill_diagonal(cost_matrix, 0)

        instance = cls(
            dist_matrix=dist_matrix,
            time_matrix=time_matrix,
            cost_matrix=cost_matrix,
            time_limit=time_limit,
            cost_limit=cost_limit,
        )
        instance.coords = coords
        return instance


class TSPSolver:
    """
    Solver wrapper for TSP instances.

    Provides nine optimisation strategies:
        solve_sa    - Simulated Annealing with 2-opt neighbourhood
        solve_ga    - Genetic Algorithm with OX crossover and swap mutation
        solve_aco   - Ant Colony Optimization
        solve_cs    - Cuckoo Search (double-bridge Levy-inspired perturbations)
        solve_abc   - Artificial Bee Colony
        solve_fa    - Firefly Algorithm (permutation adaptation)
        solve_astar - A* Search with NN lower bound (exact for small n, greedy+2opt for large)
        solve_bfs   - Beam Search (BFS-style with bounded frontier)
        solve_dfs   - Depth-First Search with branch-and-bound pruning

    Parameters
    ----------
    tsp  : TSP   The problem instance to solve.
    beta : float Penalty multiplier for constraint violations (default 2.0).
    """

    def __init__(self, tsp, beta=2.0):
        self.tsp  = tsp
        self.beta = beta

    def _penalised_distance(self, tour):
        """Distance + beta * constraint_violation (shared by all solvers)."""
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
            return T0 * np.exp(-alpha * iteration)

        def _two_opt(tour):
            n    = len(tour)
            i, j = sorted(np.random.choice(n, 2, replace=False))
            new_tour        = tour.copy()
            new_tour[i:j+1] = tour[i:j+1][::-1]
            return new_tour

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

            if delta < 0 or np.random.rand() < np.exp(-delta / T):
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
            Selection  - binary tournament (minimise penalised distance)
            Crossover  - Order Crossover (OX)
            Mutation   - swap mutation
            Elitism    - top elitism_rate individuals survive each generation

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
            n            = len(p1)
            i, j         = sorted(np.random.choice(n, 2, replace=False))
            child        = np.full(n, -1, dtype=int)
            child[i:j+1] = p1[i:j+1]
            segment_set  = set(p1[i:j+1])
            fill_vals    = [c for c in p2 if c not in segment_set]
            pos          = list(range(0, i)) + list(range(j+1, n))
            for k, p in enumerate(pos):
                child[p] = fill_vals[k]
            return child

        def _swap_mutation(tour):
            new_tour = tour.copy()
            i, j     = np.random.choice(len(tour), 2, replace=False)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            return new_tour

        def _tournament_select(population, fitnesses):
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

    def solve_aco(self, n_ants=30, max_iter=200, alpha=1.0, beta_aco=3.0, rho=0.1, Q=100.0, tau_init=1.0, verbose=True):
        """
        Solve the TSP with Ant Colony Optimization.

        Each ant constructs a complete tour using pheromone (tau) and inverse-distance
        heuristic (eta). Pheromone evaporates each iteration and is deposited proportional
        to Q / tour_length.

        Parameters
        ----------
        n_ants   : int   Number of ants per iteration.
        max_iter : int   Number of iterations.
        alpha    : float Pheromone importance exponent (tau^alpha).
        beta_aco : float Heuristic importance exponent (eta^beta).
        rho      : float Pheromone evaporation rate in [0, 1].
        Q        : float Pheromone deposit constant.
        tau_init : float Initial pheromone on all edges.
        verbose  : bool  Print when a new best is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Best distance per iteration.
        """
        n = self.tsp.n_cities
        with np.errstate(divide='ignore'):
            eta = np.where(self.tsp.dist_matrix == 0, 0.0, 1.0 / self.tsp.dist_matrix)
        tau = np.full((n, n), tau_init)
        np.fill_diagonal(tau, 0.0)

        def _build_tour():
            start   = np.random.randint(n)
            tour    = [start]
            visited = {start}
            for _ in range(n - 1):
                current   = tour[-1]
                unvisited = [c for c in range(n) if c not in visited]
                scores    = np.array(
                    [(tau[current, c] ** alpha) * (eta[current, c] ** beta_aco)
                     for c in unvisited]
                )
                total = scores.sum()
                probs = np.ones(len(unvisited)) / len(unvisited) if total == 0 else scores / total
                next_city = unvisited[np.random.choice(len(unvisited), p=probs)]
                tour.append(next_city)
                visited.add(next_city)
            return np.array(tour)

        best_tour = self.tsp.random_tour()
        best_dist = self._penalised_distance(best_tour)
        history   = []

        for iteration in range(1, max_iter + 1):
            all_tours = [_build_tour() for _ in range(n_ants)]
            all_dists = [self._penalised_distance(t) for t in all_tours]

            tau *= (1.0 - rho)
            for tour, dist in zip(all_tours, all_dists):
                deposit = Q / max(dist, 1e-10)
                for i in range(n):
                    a = tour[i]; b = tour[(i + 1) % n]
                    tau[a, b] += deposit
                    tau[b, a] += deposit

            iter_best_idx = int(np.argmin(all_dists))
            if all_dists[iter_best_idx] < best_dist:
                best_dist = all_dists[iter_best_idx]
                best_tour = all_tours[iter_best_idx].copy()
                if verbose:
                    print(
                        "[ACO] Iter: %d  dist: %.4f  feasible: %s"
                        % (iteration, self.tsp.total_distance(best_tour),
                           self.tsp.is_feasible(best_tour))
                    )

            history.append(best_dist)

        return best_tour, self.tsp.total_distance(best_tour), history

    def solve_cs(self, n_nests=25, max_iter=300, pa=0.25, verbose=True):
        """
        Solve the TSP with Cuckoo Search using double-bridge and 2-opt perturbations.

        Each nest holds a tour. New solutions are generated via double-bridge moves
        (large perturbation, Levy-flight analog) or 2-opt moves (local search).
        The worst pa fraction of nests are abandoned and replaced each iteration.

        Parameters
        ----------
        n_nests  : int   Number of host nests.
        max_iter : int   Maximum number of iterations.
        pa       : float Fraction of worst nests abandoned each iteration.
        verbose  : bool  Print when a new best is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Best distance per iteration.
        """
        def _double_bridge(tour):
            """4-opt double-bridge move for large perturbation (Levy analog)."""
            n   = len(tour)
            pos = sorted(np.random.choice(n, 4, replace=False))
            a, b, c, d = pos
            return np.concatenate([
                tour[:a+1], tour[c+1:d+1], tour[b+1:c+1], tour[a+1:b+1], tour[d+1:]
            ])

        def _two_opt(tour):
            n    = len(tour)
            i, j = sorted(np.random.choice(n, 2, replace=False))
            new_tour        = tour.copy()
            new_tour[i:j+1] = tour[i:j+1][::-1]
            return new_tour

        nests = [self.tsp.random_tour() for _ in range(n_nests)]
        dists = [self._penalised_distance(t) for t in nests]

        best_idx  = int(np.argmin(dists))
        best_tour = nests[best_idx].copy()
        best_dist = dists[best_idx]
        history   = []

        for iteration in range(1, max_iter + 1):
            for i in range(n_nests):
                candidate = _double_bridge(nests[i]) if np.random.rand() < 0.5 else _two_opt(nests[i])
                cand_dist = self._penalised_distance(candidate)
                j = np.random.randint(n_nests)
                if cand_dist < dists[j]:
                    nests[j] = candidate
                    dists[j] = cand_dist

            n_abandon = max(1, int(pa * n_nests))
            worst_idx = np.argsort(dists)[-n_abandon:]
            for i in worst_idx:
                nests[i] = self.tsp.random_tour()
                dists[i] = self._penalised_distance(nests[i])

            iter_best_idx = int(np.argmin(dists))
            if dists[iter_best_idx] < best_dist:
                best_dist = dists[iter_best_idx]
                best_tour = nests[iter_best_idx].copy()
                if verbose:
                    print(
                        "[CS] Iter: %d  dist: %.4f  feasible: %s"
                        % (iteration, self.tsp.total_distance(best_tour),
                           self.tsp.is_feasible(best_tour))
                    )

            history.append(best_dist)

        return best_tour, self.tsp.total_distance(best_tour), history

    def solve_abc(self, n_bees=30, max_iter=300, limit=None, verbose=True):
        """
        Solve the TSP with Artificial Bee Colony.

        Food sources are permutation tours. Employed bees exploit via 2-opt
        neighbourhood moves. Onlooker bees select sources via inverse-fitness
        roulette-wheel. Exhausted sources (exceeding limit trials) are abandoned
        and replaced with random tours (scout phase).

        Parameters
        ----------
        n_bees   : int       Number of employed bees (= food sources).
        max_iter : int       Maximum number of foraging cycles.
        limit    : int/None  Trials before a source is abandoned.
        verbose  : bool      Print when a new best is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Best distance per iteration.
        """
        limit = limit if limit is not None else n_bees * self.tsp.n_cities

        def _two_opt(tour):
            n    = len(tour)
            i, j = sorted(np.random.choice(n, 2, replace=False))
            new_tour        = tour.copy()
            new_tour[i:j+1] = tour[i:j+1][::-1]
            return new_tour

        sources = [self.tsp.random_tour() for _ in range(n_bees)]
        fits    = [self._penalised_distance(t) for t in sources]
        trials  = [0] * n_bees

        best_idx  = int(np.argmin(fits))
        best_tour = sources[best_idx].copy()
        best_dist = fits[best_idx]
        history   = []

        for iteration in range(1, max_iter + 1):
            # Employed bee phase
            for i in range(n_bees):
                candidate = _two_opt(sources[i])
                cand_fit  = self._penalised_distance(candidate)
                if cand_fit <= fits[i]:
                    sources[i] = candidate; fits[i] = cand_fit; trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bee phase
            inv_fits = [1.0 / max(f, 1e-10) for f in fits]
            total    = sum(inv_fits)
            probs    = [v / total for v in inv_fits]
            for _ in range(n_bees):
                i         = np.random.choice(n_bees, p=probs)
                candidate = _two_opt(sources[i])
                cand_fit  = self._penalised_distance(candidate)
                if cand_fit <= fits[i]:
                    sources[i] = candidate; fits[i] = cand_fit; trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee phase
            for i in range(n_bees):
                if trials[i] >= limit:
                    sources[i] = self.tsp.random_tour()
                    fits[i]    = self._penalised_distance(sources[i])
                    trials[i]  = 0

            iter_best_idx = int(np.argmin(fits))
            if fits[iter_best_idx] < best_dist:
                best_dist = fits[iter_best_idx]
                best_tour = sources[iter_best_idx].copy()
                if verbose:
                    print(
                        "[ABC] Iter: %d  dist: %.4f  feasible: %s"
                        % (iteration, self.tsp.total_distance(best_tour),
                           self.tsp.is_feasible(best_tour))
                    )

            history.append(best_dist)

        return best_tour, self.tsp.total_distance(best_tour), history

    def solve_fa(self, n_fireflies=30, max_iter=200, alpha=0.5, beta0=1.0, gamma=0.1, alpha_decay=0.97, verbose=True):
        """
        Solve the TSP with the Firefly Algorithm adapted for permutations.

        Brightness is inversely proportional to tour length. A dimmer firefly moves
        toward a brighter one using swap moves scaled by attractiveness. Random
        perturbation is applied via additional swap moves scaled by alpha.

        Parameters
        ----------
        n_fireflies : int   Number of fireflies.
        max_iter    : int   Number of iterations.
        alpha       : float Randomness scale for perturbation swap moves.
        beta0       : float Base attractiveness at zero permutation distance.
        gamma       : float Light absorption coefficient.
        alpha_decay : float Multiplicative decay of alpha per iteration.
        verbose     : bool  Print when a new best is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Best distance per iteration.
        """
        def _perm_dist(t1, t2):
            """Fraction of positions that differ between two permutations."""
            return float(np.sum(t1 != t2)) / len(t1)

        def _swap_move(tour, n_swaps):
            new_tour = tour.copy()
            for _ in range(max(1, int(n_swaps))):
                i, j = np.random.choice(len(new_tour), 2, replace=False)
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            return new_tour

        positions = [self.tsp.random_tour() for _ in range(n_fireflies)]
        scores    = [self._penalised_distance(t) for t in positions]

        best_idx  = int(np.argmin(scores))
        best_tour = positions[best_idx].copy()
        best_dist = scores[best_idx]
        history   = []
        cur_alpha = alpha

        for iteration in range(1, max_iter + 1):
            new_positions = [t.copy() for t in positions]
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if scores[j] < scores[i]:
                        r_sq      = _perm_dist(positions[i], positions[j]) ** 2
                        beta      = beta0 * np.exp(-gamma * r_sq)
                        n_attract = beta * self.tsp.n_cities * 0.3
                        n_rand    = cur_alpha * self.tsp.n_cities * 0.1
                        new_positions[i] = _swap_move(new_positions[i], n_attract + n_rand)

            positions = new_positions
            scores    = [self._penalised_distance(t) for t in positions]
            cur_alpha *= alpha_decay

            iter_best_idx = int(np.argmin(scores))
            if scores[iter_best_idx] < best_dist:
                best_dist = scores[iter_best_idx]
                best_tour = positions[iter_best_idx].copy()
                if verbose:
                    print(
                        "[FA] Iter: %d  dist: %.4f  feasible: %s"
                        % (iteration, self.tsp.total_distance(best_tour),
                           self.tsp.is_feasible(best_tour))
                    )

            history.append(best_dist)

        return best_tour, self.tsp.total_distance(best_tour), history

    def solve_astar(self, verbose=True):
        """
        Solve the TSP with A* Search.

        For small instances (n_cities <= 12): exact A* with admissible lower bound
        (sum of minimum outgoing edges for unvisited cities).
        For larger instances: greedy nearest-neighbor construction from every start
        city, followed by full 2-opt improvement, returning the best tour found.

        Parameters
        ----------
        verbose : bool  Print when a solution or improvement is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Single-element list with the final penalised distance.
        """
        n = self.tsp.n_cities

        # Precompute min outgoing edge per city for admissible lower bound
        min_edge_out = np.array([
            np.min(self.tsp.dist_matrix[c, np.arange(n) != c]) for c in range(n)
        ])

        if n <= 12:
            start_city = 0
            # (f, g, current, partial_tuple, visited_frozenset)
            h0   = sum(min_edge_out[c] for c in range(n) if c != start_city)
            heap = [(h0, 0.0, start_city, (start_city,), frozenset({start_city}))]
            best_tour = self.tsp.random_tour()
            best_dist = self._penalised_distance(best_tour)

            while heap:
                f, g, current, partial, visited = heapq.heappop(heap)
                if g >= best_dist:
                    continue
                if len(partial) == n:
                    total     = g + self.tsp.dist_matrix[current, start_city]
                    full_tour = np.array(partial, dtype=int)
                    pen       = total + self.beta * self.tsp.constraint_violation(full_tour)
                    if pen < best_dist:
                        best_dist = pen
                        best_tour = full_tour
                        if verbose:
                            print(
                                "[A*] Tour found  dist: %.4f  feasible: %s"
                                % (self.tsp.total_distance(best_tour),
                                   self.tsp.is_feasible(best_tour))
                            )
                    continue
                for next_city in range(n):
                    if next_city in visited:
                        continue
                    new_g   = g + self.tsp.dist_matrix[current, next_city]
                    new_vis = visited | {next_city}
                    h       = sum(min_edge_out[c] for c in range(n) if c not in new_vis)
                    new_f   = new_g + h
                    if new_f < best_dist:
                        heapq.heappush(heap, (new_f, new_g, next_city, partial + (next_city,), new_vis))
        else:
            # Greedy nearest-neighbor from every start, then 2-opt
            best_dist = np.inf
            best_tour = None
            for start in range(n):
                tour    = [start]
                visited = {start}
                current = start
                for _ in range(n - 1):
                    unvisited = [c for c in range(n) if c not in visited]
                    next_city = min(unvisited, key=lambda c: self.tsp.dist_matrix[current, c])
                    tour.append(next_city)
                    visited.add(next_city)
                    current = next_city
                tour = np.array(tour, dtype=int)
                # 2-opt local improvement
                improved = True
                while improved:
                    improved = False
                    for i in range(1, n - 1):
                        for j in range(i + 1, n):
                            new_tour        = tour.copy()
                            new_tour[i:j+1] = tour[i:j+1][::-1]
                            if self._penalised_distance(new_tour) < self._penalised_distance(tour):
                                tour     = new_tour
                                improved = True
                d = self._penalised_distance(tour)
                if d < best_dist:
                    best_dist = d
                    best_tour = tour
            if verbose:
                print(
                    "[A*] NN+2opt dist: %.4f  feasible: %s"
                    % (self.tsp.total_distance(best_tour), self.tsp.is_feasible(best_tour))
                )

        return best_tour, self.tsp.total_distance(best_tour), [self._penalised_distance(best_tour)]

    def solve_bfs(self, beam_width=10, verbose=True):
        """
        Solve the TSP with Beam Search (BFS-style bounded frontier).

        Builds tours city-by-city. At each depth level keeps only the best
        beam_width partial tours ordered by g (actual cost so far) + h (greedy
        lower bound: sum of minimum outgoing edges for unvisited cities).

        Parameters
        ----------
        beam_width : int  Maximum number of partial tours kept at each depth.
        verbose    : bool Print when the final tour is selected.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Single-element list with the final penalised distance.
        """
        n = self.tsp.n_cities
        min_edge_out = np.array([
            np.min(self.tsp.dist_matrix[c, np.arange(n) != c]) for c in range(n)
        ])

        def _lb(visited):
            return sum(min_edge_out[c] for c in range(n) if c not in visited)

        # (f_score, g_score, partial_list, visited_set)
        beam = []
        for start in range(min(beam_width, n)):
            g = 0.0
            partial = [start]
            visited = {start}
            beam.append((_lb(visited), g, partial, visited))

        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        for _ in range(n - 1):
            candidates = []
            for f, g, partial, visited in beam:
                current   = partial[-1]
                unvisited = [c for c in range(n) if c not in visited]
                for next_city in unvisited:
                    new_g   = g + self.tsp.dist_matrix[current, next_city]
                    new_par = partial + [next_city]
                    new_vis = visited | {next_city}
                    h       = _lb(new_vis)
                    candidates.append((new_g + h, new_g, new_par, new_vis))
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        best_tour = None
        best_dist = np.inf
        for f, g, partial, visited in beam:
            tour = np.array(partial, dtype=int)
            d    = self._penalised_distance(tour)
            if d < best_dist:
                best_dist = d
                best_tour = tour

        if verbose:
            print(
                "[BFS] Beam tour  dist: %.4f  feasible: %s"
                % (self.tsp.total_distance(best_tour), self.tsp.is_feasible(best_tour))
            )

        return best_tour, self.tsp.total_distance(best_tour), [best_dist]

    def solve_dfs(self, max_nodes=50_000, verbose=True):
        """
        Solve the TSP with Depth-First Search and branch-and-bound pruning.

        Explores partial tours depth-first. Prunes branches when the lower bound
        (current cost + min edge per unvisited city) exceeds the best known solution.
        Initialises with a greedy nearest-neighbor tour as the upper bound.
        Stops after max_nodes expansions for tractability on large instances.

        Parameters
        ----------
        max_nodes : int  Maximum number of partial-tour nodes to expand.
        verbose   : bool Print when a new best complete tour is found.

        Returns
        -------
        best_tour : ndarray  Best permutation of city indices.
        best_dist : float    Total distance of that tour.
        history   : list     Improving best distances found during search.
        """
        n = self.tsp.n_cities
        min_edge_out = np.array([
            np.min(self.tsp.dist_matrix[c, np.arange(n) != c]) for c in range(n)
        ])

        # Warm-start with greedy nearest-neighbor
        nn_tour = [0]; nn_vis = {0}; nn_cur = 0
        for _ in range(n - 1):
            unvis  = [c for c in range(n) if c not in nn_vis]
            next_c = min(unvis, key=lambda c: self.tsp.dist_matrix[nn_cur, c])
            nn_tour.append(next_c); nn_vis.add(next_c); nn_cur = next_c
        best_tour = np.array(nn_tour, dtype=int)
        best_dist = self._penalised_distance(best_tour)
        history   = [best_dist]

        stack      = [([0], {0}, 0.0)]
        nodes_seen = 0

        while stack and nodes_seen < max_nodes:
            partial, visited, g = stack.pop()
            nodes_seen += 1
            current   = partial[-1]
            unvisited = [c for c in range(n) if c not in visited]

            if not unvisited:
                g_total = g + self.tsp.dist_matrix[current, partial[0]]
                full    = np.array(partial, dtype=int)
                pen     = g_total + self.beta * self.tsp.constraint_violation(full)
                if pen < best_dist:
                    best_dist = pen
                    best_tour = full.copy()
                    history.append(best_dist)
                    if verbose:
                        print(
                            "[DFS] Nodes: %d  dist: %.4f  feasible: %s"
                            % (nodes_seen, self.tsp.total_distance(best_tour),
                               self.tsp.is_feasible(best_tour))
                        )
                continue

            lb = g + min_edge_out[current] + sum(min_edge_out[c] for c in unvisited)
            if lb >= best_dist:
                continue

            children = sorted(unvisited, key=lambda c: self.tsp.dist_matrix[current, c], reverse=True)
            for next_city in children:
                new_g = g + self.tsp.dist_matrix[current, next_city]
                if new_g < best_dist:
                    stack.append((partial + [next_city], visited | {next_city}, new_g))

        return best_tour, self.tsp.total_distance(best_tour), history


if __name__ == "__main__":
    N_CITIES = 15
    SEED     = 42

    tsp = TSP.generate(n_cities=N_CITIES, seed=SEED)

    print("=" * 56)
    print("  TSP Demo  -  %d cities" % N_CITIES)
    print("  Time limit : %s" % tsp.time_limit)
    print("  Cost limit : %s" % tsp.cost_limit)
    print("=" * 56)

    solver = TSPSolver(tsp, beta=2.0)

    print("\n>>> Simulated Annealing")
    sa_tour, sa_dist, _ = solver.solve_sa(T0=500.0, T_min=1e-3, max_iter=10_000, alpha=0.003, verbose=True)
    print("[SA] Best dist: %.4f  feasible: %s" % (sa_dist, tsp.is_feasible(sa_tour)))

    print("\n>>> Genetic Algorithm")
    ga_tour, ga_dist, _ = solver.solve_ga(pop_size=100, max_iter=500, verbose=True)
    print("[GA] Best dist: %.4f  feasible: %s" % (ga_dist, tsp.is_feasible(ga_tour)))

    print("\n>>> Ant Colony Optimization")
    aco_tour, aco_dist, _ = solver.solve_aco(n_ants=30, max_iter=200, verbose=True)
    print("[ACO] Best dist: %.4f  feasible: %s" % (aco_dist, tsp.is_feasible(aco_tour)))

    print("\n>>> Cuckoo Search")
    cs_tour, cs_dist, _ = solver.solve_cs(n_nests=25, max_iter=300, verbose=True)
    print("[CS] Best dist: %.4f  feasible: %s" % (cs_dist, tsp.is_feasible(cs_tour)))

    print("\n>>> Artificial Bee Colony")
    abc_tour, abc_dist, _ = solver.solve_abc(n_bees=30, max_iter=200, verbose=True)
    print("[ABC] Best dist: %.4f  feasible: %s" % (abc_dist, tsp.is_feasible(abc_tour)))

    print("\n>>> Firefly Algorithm")
    fa_tour, fa_dist, _ = solver.solve_fa(n_fireflies=30, max_iter=150, verbose=True)
    print("[FA] Best dist: %.4f  feasible: %s" % (fa_dist, tsp.is_feasible(fa_tour)))

    print("\n>>> A* Search")
    astar_tour, astar_dist, _ = solver.solve_astar(verbose=True)
    print("[A*] Best dist: %.4f  feasible: %s" % (astar_dist, tsp.is_feasible(astar_tour)))

    print("\n>>> Beam Search (BFS)")
    bfs_tour, bfs_dist, _ = solver.solve_bfs(beam_width=10, verbose=True)
    print("[BFS] Best dist: %.4f  feasible: %s" % (bfs_dist, tsp.is_feasible(bfs_tour)))

    print("\n>>> DFS (Branch and Bound)")
    dfs_tour, dfs_dist, _ = solver.solve_dfs(max_nodes=50_000, verbose=True)
    print("[DFS] Best dist: %.4f  feasible: %s" % (dfs_dist, tsp.is_feasible(dfs_tour)))

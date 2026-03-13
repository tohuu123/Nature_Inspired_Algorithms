import numpy as np
import heapq
from collections import deque
from matplotlib import pyplot
import json

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
    
    def load_from_json(self, filepath):
        """
        Load graph coloring instance from JSON file.
        """

        with open(filepath, "r") as f:
            data = json.load(f)

        self.n_vertices = data["n_vertices"]
        self.edges = [tuple(e) for e in data["edges"]]
        self.adj_matrix = self._build_adjacency_matrix()
        
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

    Provides nine optimisation strategies:
        solve_sa    - Simulated Annealing
        solve_ga    - Genetic Algorithm
        solve_aco   - Ant Colony Optimization
        solve_cs    - Cuckoo Search
        solve_abc   - Artificial Bee Colony
        solve_fa    - Firefly Algorithm
        solve_astar - A* Search with conflict-guided heuristic
        solve_bfs   - Beam Search (BFS-style bounded frontier)
        solve_dfs   - DFS with backtracking and conflict pruning

    Parameters
    ----------
    problem  : GraphColoring  The problem instance to solve.
    n_colors : int            Number of colors to use.
    beta     : float          Penalty multiplier for conflict violations (default 1.0).
    """

    def __init__(self, problem, n_colors, beta=1.0):
        self.problem  = problem
        self.n_colors = n_colors
        self.beta     = beta

    def _fitness(self, coloring):
        """Penalised fitness: n_colors_used + beta * n_conflicts."""
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
            return T0 * np.exp(-alpha * iteration)

        def _neighbour(coloring):
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
            Selection  - tournament selection (minimise fitness)
            Crossover  - uniform crossover
            Mutation   - random gene reassignment
            Elitism    - top elitism_rate individuals survive each generation

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
            mask = np.random.rand(len(p1)) < CR
            return np.where(mask, p1, p2)

        def _mutate(coloring):
            new_coloring = coloring.copy()
            vertex = np.random.randint(self.problem.n_vertices)
            new_coloring[vertex] = np.random.randint(self.n_colors)
            return new_coloring

        def _tournament_select(population, fitnesses):
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

    def solve_aco(self, n_ants=30, max_iter=200, alpha=1.0, beta_aco=2.0, rho=0.1, Q=10.0, tau_init=1.0, verbose=True):
        """
        Solve graph coloring with Ant Colony Optimization.

        Each ant assigns colors vertex-by-vertex. The pheromone matrix tau[v][c]
        represents the desirability of assigning color c to vertex v. The heuristic
        for vertex v and color c is inversely proportional to the number of already-
        assigned conflicting neighbors using color c.

        Parameters
        ----------
        n_ants   : int   Number of ants per iteration.
        max_iter : int   Number of iterations.
        alpha    : float Pheromone importance exponent.
        beta_aco : float Heuristic importance exponent.
        rho      : float Pheromone evaporation rate in [0, 1].
        Q        : float Pheromone deposit constant.
        tau_init : float Initial pheromone level.
        verbose  : bool  Print when a new best is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Best fitness per iteration.
        """
        n   = self.problem.n_vertices
        k   = self.n_colors
        adj = self.problem.adj_matrix
        tau = np.full((n, k), tau_init)

        def _build_coloring():
            coloring = np.full(n, -1, dtype=int)
            order = np.random.permutation(n)
            for v in order:
                # Heuristic: penalize colors used by already-assigned neighbors
                neighbor_colors = np.zeros(k, dtype=int)
                for u in range(n):
                    if adj[v, u] == 1 and coloring[u] >= 0:
                        neighbor_colors[coloring[u]] += 1
                scores = np.array([
                    (tau[v, c] ** alpha) * (1.0 / (1 + neighbor_colors[c])) ** beta_aco
                    for c in range(k)
                ])
                total = scores.sum()
                probs = np.ones(k) / k if total == 0 else scores / total
                coloring[v] = np.random.choice(k, p=probs)
            return coloring

        best_coloring = self.problem.random_coloring(self.n_colors)
        best_fit      = self._fitness(best_coloring)
        history       = []

        for iteration in range(1, max_iter + 1):
            all_colorings = [_build_coloring() for _ in range(n_ants)]
            all_fits      = [self._fitness(c) for c in all_colorings]

            tau *= (1.0 - rho)
            for coloring, fit in zip(all_colorings, all_fits):
                deposit = Q / max(fit, 1e-10)
                for v in range(n):
                    tau[v, coloring[v]] += deposit

            iter_best_idx = int(np.argmin(all_fits))
            if all_fits[iter_best_idx] < best_fit:
                best_fit      = all_fits[iter_best_idx]
                best_coloring = all_colorings[iter_best_idx].copy()
                if verbose:
                    print(
                        "[ACO] Iter: %d  colors: %d  conflicts: %d  valid: %s"
                        % (iteration, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )

            history.append(best_fit)

        return best_coloring, self.problem.n_colors_used(best_coloring), history

    def solve_cs(self, n_nests=25, max_iter=300, pa=0.25, verbose=True):
        """
        Solve graph coloring with Cuckoo Search.

        Each nest holds a coloring vector. New solutions are generated by randomly
        reassigning a subset of vertices (Levy-inspired perturbation). The worst pa
        fraction of nests are abandoned and replaced with random colorings each iteration.

        Parameters
        ----------
        n_nests  : int   Number of host nests.
        max_iter : int   Maximum iterations.
        pa       : float Fraction of worst nests abandoned each iteration.
        verbose  : bool  Print when a new best is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Best fitness per iteration.
        """
        n = self.problem.n_vertices

        def _perturb(coloring):
            """Randomly re-color a geometric-Levy-sized subset of vertices."""
            new_col = coloring.copy()
            step_size = max(1, int(np.random.exponential(scale=n * 0.15)))
            vertices  = np.random.choice(n, min(step_size, n), replace=False)
            for v in vertices:
                new_col[v] = np.random.randint(self.n_colors)
            return new_col

        nests = [self.problem.random_coloring(self.n_colors) for _ in range(n_nests)]
        fits  = [self._fitness(c) for c in nests]

        best_idx      = int(np.argmin(fits))
        best_coloring = nests[best_idx].copy()
        best_fit      = fits[best_idx]
        history       = []

        for iteration in range(1, max_iter + 1):
            for i in range(n_nests):
                candidate = _perturb(nests[i])
                cand_fit  = self._fitness(candidate)
                j = np.random.randint(n_nests)
                if cand_fit < fits[j]:
                    nests[j] = candidate
                    fits[j]  = cand_fit

            n_abandon = max(1, int(pa * n_nests))
            worst_idx = np.argsort(fits)[-n_abandon:]
            for i in worst_idx:
                nests[i] = self.problem.random_coloring(self.n_colors)
                fits[i]  = self._fitness(nests[i])

            iter_best_idx = int(np.argmin(fits))
            if fits[iter_best_idx] < best_fit:
                best_fit      = fits[iter_best_idx]
                best_coloring = nests[iter_best_idx].copy()
                if verbose:
                    print(
                        "[CS] Iter: %d  colors: %d  conflicts: %d  valid: %s"
                        % (iteration, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )

            history.append(best_fit)

        return best_coloring, self.problem.n_colors_used(best_coloring), history

    def solve_abc(self, n_bees=30, max_iter=300, limit=None, verbose=True):
        """
        Solve graph coloring with Artificial Bee Colony.

        Food sources are coloring vectors. Employed bees exploit via single-vertex
        recoloring. Onlooker bees select sources proportional to inverse fitness.
        Exhausted sources are abandoned and replaced with random colorings.

        Parameters
        ----------
        n_bees   : int       Number of employed bees (= food sources).
        max_iter : int       Maximum number of foraging cycles.
        limit    : int/None  Trials before a source is abandoned.
        verbose  : bool      Print when a new best is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Best fitness per iteration.
        """
        limit = limit if limit is not None else n_bees * self.problem.n_vertices

        def _neighbour(coloring):
            new_col = coloring.copy()
            vertex  = np.random.randint(self.problem.n_vertices)
            new_col[vertex] = np.random.randint(self.n_colors)
            return new_col

        sources = [self.problem.random_coloring(self.n_colors) for _ in range(n_bees)]
        fits    = [self._fitness(c) for c in sources]
        trials  = [0] * n_bees

        best_idx      = int(np.argmin(fits))
        best_coloring = sources[best_idx].copy()
        best_fit      = fits[best_idx]
        history       = []

        for iteration in range(1, max_iter + 1):
            # Employed bee phase
            for i in range(n_bees):
                candidate = _neighbour(sources[i])
                cand_fit  = self._fitness(candidate)
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
                candidate = _neighbour(sources[i])
                cand_fit  = self._fitness(candidate)
                if cand_fit <= fits[i]:
                    sources[i] = candidate; fits[i] = cand_fit; trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee phase
            for i in range(n_bees):
                if trials[i] >= limit:
                    sources[i] = self.problem.random_coloring(self.n_colors)
                    fits[i]    = self._fitness(sources[i])
                    trials[i]  = 0

            iter_best_idx = int(np.argmin(fits))
            if fits[iter_best_idx] < best_fit:
                best_fit      = fits[iter_best_idx]
                best_coloring = sources[iter_best_idx].copy()
                if verbose:
                    print(
                        "[ABC] Iter: %d  colors: %d  conflicts: %d  valid: %s"
                        % (iteration, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )

            history.append(best_fit)

        return best_coloring, self.problem.n_colors_used(best_coloring), history

    def solve_fa(self, n_fireflies=30, max_iter=200, alpha=0.5, beta0=1.0, gamma=1.0, alpha_decay=0.97, verbose=True):
        """
        Solve graph coloring with the Firefly Algorithm.

        Brightness is inversely proportional to fitness (lower is brighter).
        A dimmer firefly moves toward a brighter one: the fraction of differing
        gene positions determines the 'distance'. Attraction is applied by copying
        that fraction of gene values from the brighter firefly, plus random re-coloring
        of alpha-scaled random vertices.

        Parameters
        ----------
        n_fireflies : int   Number of fireflies.
        max_iter    : int   Number of iterations.
        alpha       : float Randomness scale (fraction of vertices randomly re-colored).
        beta0       : float Base attractiveness at zero distance.
        gamma       : float Light absorption coefficient.
        alpha_decay : float Multiplicative decay of alpha per iteration.
        verbose     : bool  Print when a new best is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Best fitness per iteration.
        """
        n = self.problem.n_vertices

        def _coloring_dist(c1, c2):
            return float(np.sum(c1 != c2)) / n

        def _move_toward(ci, cj, beta, cur_alpha):
            new_ci = ci.copy()
            diff_mask = ci != cj
            diff_pos  = np.where(diff_mask)[0]
            # Copy a beta-scaled portion of differing positions from cj
            n_copy = max(0, int(beta * len(diff_pos)))
            if n_copy > 0:
                chosen = np.random.choice(diff_pos, n_copy, replace=False)
                new_ci[chosen] = cj[chosen]
            # Random perturbation
            n_rand = max(0, int(cur_alpha * n))
            if n_rand > 0:
                rand_v = np.random.choice(n, n_rand, replace=False)
                for v in rand_v:
                    new_ci[v] = np.random.randint(self.n_colors)
            return new_ci

        positions = [self.problem.random_coloring(self.n_colors) for _ in range(n_fireflies)]
        scores    = [self._fitness(c) for c in positions]

        best_idx      = int(np.argmin(scores))
        best_coloring = positions[best_idx].copy()
        best_fit      = scores[best_idx]
        history       = []
        cur_alpha     = alpha

        for iteration in range(1, max_iter + 1):
            new_positions = [c.copy() for c in positions]
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if scores[j] < scores[i]:
                        r_sq = _coloring_dist(positions[i], positions[j]) ** 2
                        beta = beta0 * np.exp(-gamma * r_sq)
                        new_positions[i] = _move_toward(new_positions[i], positions[j], beta, cur_alpha)

            positions = new_positions
            scores    = [self._fitness(c) for c in positions]
            cur_alpha *= alpha_decay

            iter_best_idx = int(np.argmin(scores))
            if scores[iter_best_idx] < best_fit:
                best_fit      = scores[iter_best_idx]
                best_coloring = positions[iter_best_idx].copy()
                if verbose:
                    print(
                        "[FA] Iter: %d  colors: %d  conflicts: %d  valid: %s"
                        % (iteration, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )

            history.append(best_fit)

        return best_coloring, self.problem.n_colors_used(best_coloring), history

    def solve_astar(self, verbose=True):
        """
        Solve graph coloring with A* Search.

        Assigns colors one vertex at a time in descending degree order.
        State: partial coloring vector (assigned vertices so far).
        g = number of conflicts introduced so far.
        h = lower bound on additional conflicts: for each unassigned vertex,
            count 0 (optimistic; coloring may still be conflict-free).

        For small instances this performs exact search. For larger instances it uses
        a greedy degree-based coloring (assign the color that minimizes new conflicts,
        then break ties by minimum color index) followed by restart hill-climbing.

        Parameters
        ----------
        verbose : bool  Print when a solution or improvement is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Single-element list with the final fitness.
        """
        n   = self.problem.n_vertices
        adj = self.problem.adj_matrix

        # Degree-ordered vertex sequence (high degree first)
        degrees = adj.sum(axis=1)
        order   = np.argsort(-degrees).tolist()

        def _greedy_color():
            coloring = np.full(n, -1, dtype=int)
            for v in order:
                # Count how many neighbors already use each color
                neighbor_counts = np.zeros(self.n_colors, dtype=int)
                for u in range(n):
                    if adj[v, u] == 1 and coloring[u] >= 0:
                        neighbor_counts[coloring[u]] += 1
                # Pick the color with fewest neighbor conflicts (greedy)
                coloring[v] = int(np.argmin(neighbor_counts))
            return coloring

        best_coloring = _greedy_color()
        best_fit      = self._fitness(best_coloring)

        # Hill-climbing restarts with random perturbations
        for restart in range(20):
            coloring = _greedy_color().copy()
            # Randomly perturb a subset of most-conflicted vertices
            for _ in range(self.problem.n_vertices * 2):
                conflict_v = [v for v in range(n) if any(
                    coloring[v] == coloring[u] for u in range(n) if adj[v, u] == 1
                )]
                if not conflict_v:
                    break
                v        = np.random.choice(conflict_v)
                neighbor_counts = np.zeros(self.n_colors, dtype=int)
                for u in range(n):
                    if adj[v, u] == 1:
                        neighbor_counts[coloring[u]] += 1
                coloring[v] = int(np.argmin(neighbor_counts))

            fit = self._fitness(coloring)
            if fit < best_fit:
                best_fit      = fit
                best_coloring = coloring.copy()
                if verbose:
                    print(
                        "[A*] Restart %d  colors: %d  conflicts: %d  valid: %s"
                        % (restart, self.problem.n_colors_used(best_coloring),
                           self.problem.n_conflicts(best_coloring),
                           self.problem.is_valid(best_coloring))
                    )
            if best_fit == 0:
                break

        if verbose and best_fit > 0:
            print(
                "[A*] Final  colors: %d  conflicts: %d  valid: %s"
                % (self.problem.n_colors_used(best_coloring),
                   self.problem.n_conflicts(best_coloring),
                   self.problem.is_valid(best_coloring))
            )

        return best_coloring, self.problem.n_colors_used(best_coloring), [best_fit]

    def solve_bfs(self, beam_width=20, verbose=True):
        """
        Solve graph coloring with Beam Search (BFS-style bounded frontier).

        Assigns colors vertex-by-vertex in descending degree order. At each
        depth level, keeps only the best beam_width partial colorings sorted by
        current conflict count (g) + 0 (optimistic h = 0).

        Parameters
        ----------
        beam_width : int  Maximum number of partial colorings kept at each level.
        verbose    : bool Print when the final coloring is selected.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Single-element list with the final fitness.
        """
        n   = self.problem.n_vertices
        adj = self.problem.adj_matrix
        degrees = adj.sum(axis=1)
        order   = np.argsort(-degrees).tolist()

        def _conflicts_so_far(partial, assigned_set):
            count = 0
            for v in assigned_set:
                for u in assigned_set:
                    if u < v and adj[v, u] == 1 and partial[v] == partial[u]:
                        count += 1
            return count

        # Beam: (conflict_count, partial_coloring_tuple, assigned_set)
        init = np.full(n, -1, dtype=int)
        beam = [(0, tuple(init), set())]

        for depth, v in enumerate(order):
            candidates = []
            for g, partial_t, assigned in beam:
                partial = list(partial_t)
                for c in range(self.n_colors):
                    new_partial = partial[:]
                    new_partial[v] = c
                    new_assigned   = assigned | {v}
                    new_g = _conflicts_so_far(new_partial, new_assigned)
                    candidates.append((new_g, tuple(new_partial), new_assigned))
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        best_coloring = None
        best_fit      = np.inf
        for g, partial_t, assigned in beam:
            coloring = np.array(partial_t, dtype=int)
            fit      = self._fitness(coloring)
            if fit < best_fit:
                best_fit      = fit
                best_coloring = coloring

        if verbose:
            print(
                "[BFS] Beam coloring  colors: %d  conflicts: %d  valid: %s"
                % (self.problem.n_colors_used(best_coloring),
                   self.problem.n_conflicts(best_coloring),
                   self.problem.is_valid(best_coloring))
            )

        return best_coloring, self.problem.n_colors_used(best_coloring), [best_fit]

    def solve_dfs(self, max_nodes=10_000, verbose=True):
        """
        Solve graph coloring with DFS and backtracking (constraint propagation).

        Traverses vertices in descending degree order and assigns colors DFS-style.
        Branches pruned when partial conflict count already exceeds best known.
        Stops after max_nodes expansions for tractability.

        Parameters
        ----------
        max_nodes : int  Maximum number of partial-coloring states to expand.
        verbose   : bool Print when a new best complete coloring is found.

        Returns
        -------
        best_coloring : ndarray  Best coloring vector found.
        best_colors   : int      Number of distinct colors used.
        history       : list     Improving fitness values found during search.
        """
        n   = self.problem.n_vertices
        adj = self.problem.adj_matrix
        degrees = adj.sum(axis=1)
        order   = np.argsort(-degrees).tolist()

        # Warm-start with greedy coloring
        best_coloring = np.full(n, -1, dtype=int)
        for v in order:
            neighbor_c = np.zeros(self.n_colors, dtype=int)
            for u in range(n):
                if adj[v, u] == 1 and best_coloring[u] >= 0:
                    neighbor_c[best_coloring[u]] += 1
            best_coloring[v] = int(np.argmin(neighbor_c))
        best_fit = self._fitness(best_coloring)
        history  = [best_fit]

        # DFS stack: (partial_coloring, depth_index, conflict_so_far)
        init  = np.full(n, -1, dtype=int)
        stack = [(init.copy(), 0, 0)]
        nodes_seen = 0

        while stack and nodes_seen < max_nodes:
            coloring, depth, conflicts = stack.pop()
            nodes_seen += 1

            if conflicts >= best_fit:
                continue

            if depth == n:
                fit = self._fitness(coloring)
                if fit < best_fit:
                    best_fit      = fit
                    best_coloring = coloring.copy()
                    history.append(best_fit)
                    if verbose:
                        print(
                            "[DFS] Nodes: %d  colors: %d  conflicts: %d  valid: %s"
                            % (nodes_seen, self.problem.n_colors_used(best_coloring),
                               self.problem.n_conflicts(best_coloring),
                               self.problem.is_valid(best_coloring))
                        )
                continue

            v = order[depth]
            # Try colors that minimize immediate conflicts first
            color_order = sorted(range(self.n_colors), key=lambda c: sum(
                1 for u in range(n) if adj[v, u] == 1 and coloring[u] == c
            ))
            for c in reversed(color_order):  # reversed so cheapest is last (LIFO)
                new_color = coloring.copy()
                new_color[v] = c
                new_conflicts = conflicts + sum(
                    1 for u in range(n) if adj[v, u] == 1 and coloring[u] == c
                )
                if new_conflicts < best_fit:
                    stack.append((new_color, depth + 1, new_conflicts))

        return best_coloring, self.problem.n_colors_used(best_coloring), history


if __name__ == "__main__":
    N_VERTICES = 20
    N_COLORS   = 5
    SEED       = 42

    problem = GraphColoring.generate(
        n_vertices=N_VERTICES,
        edge_probability=0.4,
        seed=SEED,
    )

    print("=" * 56)
    print("  Graph Coloring Demo  -  %d vertices" % N_VERTICES)
    print("  Edges  : %d" % len(problem.edges))
    print("  Colors : %d" % N_COLORS)
    print("=" * 56)

    solver = GraphColoringSolver(problem, n_colors=N_COLORS, beta=1.0)

    print("\n>>> Simulated Annealing")
    sa_col, sa_c, _ = solver.solve_sa(T0=100.0, T_min=1e-3, max_iter=10_000, alpha=0.003, verbose=True)
    print("[SA] Colors: %d  Conflicts: %d  Valid: %s" % (sa_c, problem.n_conflicts(sa_col), problem.is_valid(sa_col)))

    print("\n>>> Genetic Algorithm")
    ga_col, ga_c, _ = solver.solve_ga(pop_size=100, max_iter=500, verbose=True)
    print("[GA] Colors: %d  Conflicts: %d  Valid: %s" % (ga_c, problem.n_conflicts(ga_col), problem.is_valid(ga_col)))

    print("\n>>> Ant Colony Optimization")
    aco_col, aco_c, _ = solver.solve_aco(n_ants=30, max_iter=200, verbose=True)
    print("[ACO] Colors: %d  Conflicts: %d  Valid: %s" % (aco_c, problem.n_conflicts(aco_col), problem.is_valid(aco_col)))

    print("\n>>> Cuckoo Search")
    cs_col, cs_c, _ = solver.solve_cs(n_nests=25, max_iter=300, verbose=True)
    print("[CS] Colors: %d  Conflicts: %d  Valid: %s" % (cs_c, problem.n_conflicts(cs_col), problem.is_valid(cs_col)))

    print("\n>>> Artificial Bee Colony")
    abc_col, abc_c, _ = solver.solve_abc(n_bees=30, max_iter=300, verbose=True)
    print("[ABC] Colors: %d  Conflicts: %d  Valid: %s" % (abc_c, problem.n_conflicts(abc_col), problem.is_valid(abc_col)))

    print("\n>>> Firefly Algorithm")
    fa_col, fa_c, _ = solver.solve_fa(n_fireflies=30, max_iter=200, verbose=True)
    print("[FA] Colors: %d  Conflicts: %d  Valid: %s" % (fa_c, problem.n_conflicts(fa_col), problem.is_valid(fa_col)))

    print("\n>>> A* Search")
    astar_col, astar_c, _ = solver.solve_astar(verbose=True)
    print("[A*] Colors: %d  Conflicts: %d  Valid: %s" % (astar_c, problem.n_conflicts(astar_col), problem.is_valid(astar_col)))

    print("\n>>> BFS")
    bfs_col, bfs_c, _ = solver.solve_bfs(beam_width=20, verbose=True)
    print("[BFS] Colors: %d  Conflicts: %d  Valid: %s" % (bfs_c, problem.n_conflicts(bfs_col), problem.is_valid(bfs_col)))

    print("\n>>> DFS")
    dfs_col, dfs_c, _ = solver.solve_dfs(max_nodes=10_000, verbose=True)
    print("[DFS] Colors: %d  Conflicts: %d  Valid: %s" % (dfs_c, problem.n_conflicts(dfs_col), problem.is_valid(dfs_col)))

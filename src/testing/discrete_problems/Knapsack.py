import numpy as np
import heapq
from collections import deque
from matplotlib import pyplot
import json

class KnapsackProblem:
    """
    0/1 Knapsack Problem instance.

    Parameters
    ----------
    values   : array-like, shape (n_items,)  Profit/value of each item.
    weights  : array-like, shape (n_items,)  Weight of each item.
    capacity : float                          Maximum knapsack capacity.
    """

    def __init__(self, values, weights, capacity):
        self.values   = np.asarray(values,  dtype=float)
        self.weights  = np.asarray(weights, dtype=float)
        self.capacity = float(capacity)
        self.n_items  = len(self.values)

    def fitness(self, x):
        """
        Evaluate a binary solution vector.

        Returns total value if the weight constraint is satisfied, else 0.
        """
        x            = np.asarray(x, dtype=int)
        total_weight = float(x @ self.weights)
        total_value  = float(x @ self.values)
        return total_value if total_weight <= self.capacity else 0.0

    def is_feasible(self, x):
        """Return True iff total weight does not exceed capacity."""
        return float(np.asarray(x, dtype=int) @ self.weights) <= self.capacity

    def load_from_json(self, filepath):
        """
        Load knapsack instance from JSON file.
        """

        with open(filepath, "r") as f:
            data = json.load(f)

        self.values = np.asarray(data["values"], dtype=float)
        self.weights = np.asarray(data["weights"], dtype=float)
        self.capacity = float(data["capacity"])
        self.n_items = len(self.values)
        
    @classmethod
    def generate(cls, n_items=20, capacity_ratio=0.5, seed=None):
        """
        Create a random knapsack instance.

        Parameters
        ----------
        n_items        : int    Number of items.
        capacity_ratio : float  Capacity as a fraction of total weight.
        seed           : int    Random seed.
        """
        rng     = np.random.default_rng(seed)
        values  = rng.integers(1, 101, size=n_items).astype(float)
        weights = rng.integers(1, 51,  size=n_items).astype(float)
        return cls(values, weights, capacity_ratio * weights.sum())


class KnapsackSolver:
    """
    Solver wrapper for KnapsackProblem instances.

    Provides nine optimisation strategies:
        solve_sa    - Simulated Annealing (binary bit-flip)
        solve_ga    - Genetic Algorithm (binary encoding)
        solve_aco   - Ant Colony Optimization (binary item selection)
        solve_cs    - Cuckoo Search (binary Levy-inspired flips)
        solve_abc   - Artificial Bee Colony (binary bit-flip)
        solve_fa    - Firefly Algorithm (binary encoding)
        solve_astar - A* with LP-relaxation upper bound (branch and bound)
        solve_bfs   - Beam Search (BFS-style over partial solutions)
        solve_dfs   - DFS with branch-and-bound pruning

    All solvers MAXIMISE the knapsack value (feasible solutions only).

    Parameters
    ----------
    problem : KnapsackProblem  The problem instance to solve.
    """

    def __init__(self, problem):
        self.problem = problem

    def solve_sa(self, T0=500.0, T_min=1e-3, max_iter=10_000, alpha=0.003, verbose=True):
        """
        Solve the knapsack with Simulated Annealing.

        Neighbourhood: flip a single random bit.
        Cooling      : exponential  T(k) = T0 * exp(-alpha * k).

        Parameters
        ----------
        T0       : float  Initial temperature.
        T_min    : float  Stopping temperature.
        max_iter : int    Maximum number of iterations.
        alpha    : float  Exponential cooling rate.
        verbose  : bool   Print when a new best is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Best value per iteration.
        """
        n = self.problem.n_items

        def _flip(x):
            new_x = x.copy()
            new_x[np.random.randint(n)] ^= 1
            return new_x

        current     = np.random.randint(0, 2, n)
        current_fit = self.problem.fitness(current)
        best_sol    = current.copy()
        best_val    = current_fit
        history     = []

        for iteration in range(1, max_iter + 1):
            T = T0 * np.exp(-alpha * iteration)
            if T <= T_min:
                break
            candidate     = _flip(current)
            candidate_fit = self.problem.fitness(candidate)
            delta         = candidate_fit - current_fit
            # Maximise: accept if better, or probabilistically if worse
            if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
                current     = candidate
                current_fit = candidate_fit
            if current_fit > best_val:
                best_val = current_fit
                best_sol = current.copy()
                if verbose:
                    print(
                        "[SA] Iter: %d  T: %.4f  value: %.2f  weight: %.2f  feasible: %s"
                        % (iteration, T, best_val,
                           float(best_sol @ self.problem.weights),
                           self.problem.is_feasible(best_sol))
                    )
            history.append(best_val)

        return best_sol, best_val, history

    def solve_ga(self, pop_size=100, max_iter=500, CR=0.8, mutation_rate=0.02, tournament_size=3, elitism_rate=0.1, verbose=True):
        """
        Solve the knapsack with a Genetic Algorithm.

        Encoding  : binary vector (0/1 per item).
        Crossover : uniform crossover.
        Mutation  : bit-flip mutation.
        Elitism   : top elitism_rate individuals survive each generation.

        Parameters
        ----------
        pop_size        : int    Population size.
        max_iter        : int    Number of generations.
        CR              : float  Crossover probability per gene.
        mutation_rate   : float  Per-individual mutation probability.
        tournament_size : int    Number of candidates per tournament.
        elitism_rate    : float  Fraction of best individuals kept each gen.
        verbose         : bool   Print when a new best is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Best value per generation.
        """
        n = self.problem.n_items

        def _uniform_crossover(p1, p2):
            mask = np.random.rand(n) < CR
            return np.where(mask, p1, p2)

        def _mutate(x):
            new_x = x.copy()
            new_x[np.random.randint(n)] ^= 1
            return new_x

        def _tournament_select(population, fitnesses):
            idx      = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = idx[np.argmax(fitnesses[idx])]
            return population[best_idx].copy()

        population = np.random.randint(0, 2, (pop_size, n))
        fitnesses  = np.array([self.problem.fitness(x) for x in population])

        best_idx = np.argmax(fitnesses)
        best_sol = population[best_idx].copy()
        best_val = fitnesses[best_idx]
        history  = []

        elitism_count  = max(1, int(pop_size * elitism_rate))
        offspring_size = pop_size - elitism_count

        for generation in range(1, max_iter + 1):
            elite_idx = np.argsort(fitnesses)[-elitism_count:]
            elite_pop = population[elite_idx].copy()

            offspring = []
            while len(offspring) < offspring_size:
                p1    = _tournament_select(population, fitnesses)
                p2    = _tournament_select(population, fitnesses)
                child = _uniform_crossover(p1, p2)
                if np.random.rand() < mutation_rate:
                    child = _mutate(child)
                offspring.append(child)

            offspring  = np.array(offspring[:offspring_size])
            population = np.vstack([elite_pop, offspring])
            fitnesses  = np.array([self.problem.fitness(x) for x in population])

            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_val:
                best_val = fitnesses[gen_best_idx]
                best_sol = population[gen_best_idx].copy()
                if verbose:
                    print(
                        "[GA] Gen: %d  value: %.2f  weight: %.2f  feasible: %s"
                        % (generation, best_val,
                           float(best_sol @ self.problem.weights),
                           self.problem.is_feasible(best_sol))
                    )

            history.append(best_val)

        return best_sol, best_val, history

    def solve_aco(self, n_ants=30, max_iter=200, alpha=1.0, beta_aco=2.0, rho=0.1, Q=100.0, tau_init=1.0, verbose=True):
        """
        Solve the knapsack with Ant Colony Optimization.

        Each ant selects items one-by-one using pheromone (tau[i][0/1]) and a
        value/weight ratio heuristic. Pheromone evaporates each iteration and is
        deposited proportional to Q * solution_value.

        Parameters
        ----------
        n_ants   : int   Number of ants per iteration.
        max_iter : int   Number of iterations.
        alpha    : float Pheromone importance exponent.
        beta_aco : float Heuristic importance exponent.
        rho      : float Pheromone evaporation rate in [0, 1].
        Q        : float Pheromone deposit scale factor.
        tau_init : float Initial pheromone level.
        verbose  : bool  Print when a new best is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Best value per iteration.
        """
        n   = self.problem.n_items
        eta = self.problem.values / np.maximum(self.problem.weights, 1e-10)
        # tau[i, 0] = pheromone for not including item i
        # tau[i, 1] = pheromone for including item i
        tau = np.full((n, 2), tau_init)

        def _build_solution():
            x        = np.zeros(n, dtype=int)
            capacity = self.problem.capacity
            order    = np.random.permutation(n)
            for i in order:
                scores = np.array([
                    (tau[i, 0] ** alpha),
                    (tau[i, 1] ** alpha) * (eta[i] ** beta_aco)
                ])
                total = scores.sum()
                probs = scores / total if total > 0 else np.array([0.5, 0.5])
                choice = np.random.choice(2, p=probs)
                if choice == 1 and self.problem.weights[i] <= capacity:
                    x[i] = 1
                    capacity -= self.problem.weights[i]
            return x

        best_sol = np.zeros(n, dtype=int)
        best_val = 0.0
        history  = []

        for iteration in range(1, max_iter + 1):
            all_sols = [_build_solution() for _ in range(n_ants)]
            all_vals = [self.problem.fitness(s) for s in all_sols]

            tau *= (1.0 - rho)
            for sol, val in zip(all_sols, all_vals):
                deposit = Q * val / max(self.problem.values.sum(), 1e-10)
                for i in range(n):
                    tau[i, sol[i]] += deposit

            iter_best_idx = int(np.argmax(all_vals))
            if all_vals[iter_best_idx] > best_val:
                best_val = all_vals[iter_best_idx]
                best_sol = all_sols[iter_best_idx].copy()
                if verbose:
                    print(
                        "[ACO] Iter: %d  value: %.2f  weight: %.2f  feasible: %s"
                        % (iteration, best_val,
                           float(best_sol @ self.problem.weights),
                           self.problem.is_feasible(best_sol))
                    )

            history.append(best_val)

        return best_sol, best_val, history

    def solve_cs(self, n_nests=25, max_iter=300, pa=0.25, verbose=True):
        """
        Solve the knapsack with Cuckoo Search.

        Each nest is a binary solution vector. New solutions are generated
        by flipping a geometrically-distributed number of bits (Levy analog).
        The worst pa fraction of nests are abandoned and replaced with random
        feasibility-repaired solutions each iteration.

        Parameters
        ----------
        n_nests  : int   Number of nests.
        max_iter : int   Maximum iterations.
        pa       : float Fraction of worst nests abandoned each iteration.
        verbose  : bool  Print when a new best is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Best value per iteration.
        """
        n = self.problem.n_items

        def _repair(x):
            """Greedy repair: drop lowest value/weight items until feasible."""
            x = x.copy()
            while not self.problem.is_feasible(x):
                on_idx = np.where(x == 1)[0]
                if len(on_idx) == 0:
                    break
                ratio  = self.problem.values[on_idx] / np.maximum(self.problem.weights[on_idx], 1e-10)
                x[on_idx[np.argmin(ratio)]] = 0
            return x

        def _perturb(x):
            new_x     = x.copy()
            step_size = max(1, int(np.random.exponential(scale=n * 0.15)))
            flips     = np.random.choice(n, min(step_size, n), replace=False)
            new_x[flips] ^= 1
            return _repair(new_x)

        nests = [_repair(np.random.randint(0, 2, n)) for _ in range(n_nests)]
        fits  = [self.problem.fitness(s) for s in nests]

        best_idx = int(np.argmax(fits))
        best_sol = nests[best_idx].copy()
        best_val = fits[best_idx]
        history  = []

        for iteration in range(1, max_iter + 1):
            for i in range(n_nests):
                candidate = _perturb(nests[i])
                cand_fit  = self.problem.fitness(candidate)
                j = np.random.randint(n_nests)
                if cand_fit > fits[j]:
                    nests[j] = candidate
                    fits[j]  = cand_fit

            n_abandon = max(1, int(pa * n_nests))
            worst_idx = np.argsort(fits)[:n_abandon]
            for i in worst_idx:
                nests[i] = _repair(np.random.randint(0, 2, n))
                fits[i]  = self.problem.fitness(nests[i])

            iter_best_idx = int(np.argmax(fits))
            if fits[iter_best_idx] > best_val:
                best_val = fits[iter_best_idx]
                best_sol = nests[iter_best_idx].copy()
                if verbose:
                    print(
                        "[CS] Iter: %d  value: %.2f  weight: %.2f  feasible: %s"
                        % (iteration, best_val,
                           float(best_sol @ self.problem.weights),
                           self.problem.is_feasible(best_sol))
                    )

            history.append(best_val)

        return best_sol, best_val, history

    def solve_abc(self, n_bees=30, max_iter=300, limit=None, verbose=True):
        """
        Solve the knapsack with Artificial Bee Colony.

        Food sources are binary solution vectors. Employed bees exploit via
        single-bit flipping. Onlooker bees select sources proportional to their
        value. Exhausted sources (exceeding limit trials) are replaced with
        random feasible solutions.

        Parameters
        ----------
        n_bees   : int       Number of employed bees.
        max_iter : int       Maximum number of foraging cycles.
        limit    : int/None  Trials before a source is abandoned.
        verbose  : bool      Print when a new best is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Best value per iteration.
        """
        n     = self.problem.n_items
        limit = limit if limit is not None else n_bees * n

        def _random_source():
            x = np.random.randint(0, 2, n)
            while not self.problem.is_feasible(x):
                x[np.random.randint(n)] = 0
            return x

        def _flip_bit(x):
            new_x = x.copy()
            new_x[np.random.randint(n)] ^= 1
            return new_x

        sources = [_random_source() for _ in range(n_bees)]
        fits    = [self.problem.fitness(s) for s in sources]
        trials  = [0] * n_bees

        best_idx = int(np.argmax(fits))
        best_sol = sources[best_idx].copy()
        best_val = fits[best_idx]
        history  = []

        for iteration in range(1, max_iter + 1):
            # Employed bee phase
            for i in range(n_bees):
                candidate = _flip_bit(sources[i])
                cand_fit  = self.problem.fitness(candidate)
                if cand_fit >= fits[i]:
                    sources[i] = candidate; fits[i] = cand_fit; trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bee phase
            total = sum(fits) + 1e-10
            probs = [f / total for f in fits]
            for _ in range(n_bees):
                i         = np.random.choice(n_bees, p=probs)
                candidate = _flip_bit(sources[i])
                cand_fit  = self.problem.fitness(candidate)
                if cand_fit >= fits[i]:
                    sources[i] = candidate; fits[i] = cand_fit; trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee phase
            for i in range(n_bees):
                if trials[i] >= limit:
                    sources[i] = _random_source()
                    fits[i]    = self.problem.fitness(sources[i])
                    trials[i]  = 0

            iter_best_idx = int(np.argmax(fits))
            if fits[iter_best_idx] > best_val:
                best_val = fits[iter_best_idx]
                best_sol = sources[iter_best_idx].copy()
                if verbose:
                    print(
                        "[ABC] Iter: %d  value: %.2f  weight: %.2f  feasible: %s"
                        % (iteration, best_val,
                           float(best_sol @ self.problem.weights),
                           self.problem.is_feasible(best_sol))
                    )

            history.append(best_val)

        return best_sol, best_val, history

    def solve_fa(self, n_fireflies=30, max_iter=200, alpha=0.5, beta0=1.0, gamma=1.0, alpha_decay=0.97, verbose=True):
        """
        Solve the knapsack with the Firefly Algorithm adapted for binary solutions.

        Brightness is proportional to solution value. A dimmer firefly moves
        toward a brighter one via probabilistic bit-copying scaled by attractiveness.
        Random bit-flips are applied proportional to the current alpha.

        Parameters
        ----------
        n_fireflies : int   Number of fireflies.
        max_iter    : int   Number of iterations.
        alpha       : float Randomness scale (fraction of random bit-flips).
        beta0       : float Base attractiveness at zero binary distance.
        gamma       : float Light absorption coefficient.
        alpha_decay : float Multiplicative alpha decay per iteration.
        verbose     : bool  Print when a new best is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Best value per iteration.
        """
        n = self.problem.n_items

        def _hamming_dist(x1, x2):
            return float(np.sum(x1 != x2)) / n

        def _move_toward(xi, xj, beta, cur_alpha):
            new_xi = xi.copy()
            diff   = np.where(xi != xj)[0]
            n_copy = max(0, int(beta * len(diff)))
            if n_copy > 0:
                chosen = np.random.choice(diff, n_copy, replace=False)
                new_xi[chosen] = xj[chosen]
            n_rand = max(0, int(cur_alpha * n))
            if n_rand > 0:
                rand_bits = np.random.choice(n, n_rand, replace=False)
                new_xi[rand_bits] ^= 1
            # Repair: drop lowest ratio items until feasible
            while not self.problem.is_feasible(new_xi):
                on_idx = np.where(new_xi == 1)[0]
                if len(on_idx) == 0:
                    break
                ratio  = self.problem.values[on_idx] / np.maximum(self.problem.weights[on_idx], 1e-10)
                new_xi[on_idx[np.argmin(ratio)]] = 0
            return new_xi

        positions = [np.random.randint(0, 2, n) for _ in range(n_fireflies)]
        scores    = [self.problem.fitness(x) for x in positions]

        best_idx = int(np.argmax(scores))
        best_sol = positions[best_idx].copy()
        best_val = scores[best_idx]
        history  = []
        cur_alpha = alpha

        for iteration in range(1, max_iter + 1):
            new_positions = [x.copy() for x in positions]
            for i in range(n_fireflies):
                for j in range(n_fireflies):
                    if scores[j] > scores[i]:
                        r_sq = _hamming_dist(positions[i], positions[j]) ** 2
                        beta = beta0 * np.exp(-gamma * r_sq)
                        new_positions[i] = _move_toward(new_positions[i], positions[j], beta, cur_alpha)

            positions = new_positions
            scores    = [self.problem.fitness(x) for x in positions]
            cur_alpha *= alpha_decay

            iter_best_idx = int(np.argmax(scores))
            if scores[iter_best_idx] > best_val:
                best_val = scores[iter_best_idx]
                best_sol = positions[iter_best_idx].copy()
                if verbose:
                    print(
                        "[FA] Iter: %d  value: %.2f  weight: %.2f  feasible: %s"
                        % (iteration, best_val,
                           float(best_sol @ self.problem.weights),
                           self.problem.is_feasible(best_sol))
                    )

            history.append(best_val)

        return best_sol, best_val, history

    def solve_astar(self, verbose=True):
        """
        Solve the knapsack with A* Search (branch and bound).

        Assigns items one-by-one in descending value/weight ratio order.
        g = value collected so far; h = LP-relaxation upper bound on remaining
        items (fractional knapsack bound, always admissible). Prunes branches
        exceeding the best integer solution found so far.

        Parameters
        ----------
        verbose : bool  Print when a new best solution is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Single-element list with the final best value.
        """
        n = self.problem.n_items
        # Sort items by value/weight ratio descending
        ratio = self.problem.values / np.maximum(self.problem.weights, 1e-10)
        order = np.argsort(-ratio).tolist()

        def _lp_bound(depth, g, remaining_capacity):
            """LP relaxation upper bound for remaining items from `depth` onward."""
            ub = g
            cap = remaining_capacity
            for idx in range(depth, n):
                i = order[idx]
                if self.problem.weights[i] <= cap:
                    ub  += self.problem.values[i]
                    cap -= self.problem.weights[i]
                else:
                    ub  += ratio[i] * cap
                    break
            return ub

        best_sol = np.zeros(n, dtype=int)
        best_val = 0.0

        # Priority queue: (-f, depth, g, remaining_capacity, partial_x_list)
        # We negate f because heapq is a min-heap; we want max-first.
        h0    = _lp_bound(0, 0.0, self.problem.capacity)
        heap  = [(-h0, 0, 0.0, self.problem.capacity, [0] * n)]

        while heap:
            neg_f, depth, g, cap, x_list = heapq.heappop(heap)
            f = -neg_f
            if f <= best_val:
                continue
            if depth == n:
                if g > best_val:
                    best_val = g
                    best_sol = np.array(x_list, dtype=int)
                    if verbose:
                        print(
                            "[A*] value: %.2f  weight: %.2f  feasible: %s"
                            % (best_val,
                               float(best_sol @ self.problem.weights),
                               self.problem.is_feasible(best_sol))
                        )
                continue

            i = order[depth]

            # Branch: include item i (if weight allows)
            if self.problem.weights[i] <= cap:
                new_x       = x_list[:]
                new_x[i]    = 1
                new_g       = g + self.problem.values[i]
                new_cap     = cap - self.problem.weights[i]
                new_f       = _lp_bound(depth + 1, new_g, new_cap)
                if new_f > best_val:
                    heapq.heappush(heap, (-new_f, depth + 1, new_g, new_cap, new_x))

            # Branch: exclude item i
            excl_f = _lp_bound(depth + 1, g, cap)
            if excl_f > best_val:
                heapq.heappush(heap, (-excl_f, depth + 1, g, cap, x_list[:]))

        return best_sol, best_val, [best_val]

    def solve_bfs(self, beam_width=20, verbose=True):
        """
        Solve the knapsack with Beam Search (BFS-style bounded frontier).

        Considers items in descending value/weight ratio order. At each depth
        level (one item), expands each beam element into include/exclude branches
        and keeps only the best beam_width partial solutions by LP upper bound.

        Parameters
        ----------
        beam_width : int  Maximum partial solutions kept at each depth.
        verbose    : bool Print when the final solution is selected.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Single-element list with the final best value.
        """
        n = self.problem.n_items
        ratio = self.problem.values / np.maximum(self.problem.weights, 1e-10)
        order = np.argsort(-ratio).tolist()

        def _lp_bound(depth, g, cap):
            ub = g; c = cap
            for idx in range(depth, n):
                i = order[idx]
                if self.problem.weights[i] <= c:
                    ub += self.problem.values[i]; c -= self.problem.weights[i]
                else:
                    ub += ratio[i] * c; break
            return ub

        # Beam: (neg_lb, g, cap, x_list)  – neg_lb for max-first sorting
        init_lb = _lp_bound(0, 0.0, self.problem.capacity)
        beam    = [(-init_lb, 0.0, self.problem.capacity, [0] * n)]

        for depth in range(n):
            i          = order[depth]
            candidates = []
            for neg_lb, g, cap, x_list in beam:
                # Exclude
                excl_lb = _lp_bound(depth + 1, g, cap)
                candidates.append((-excl_lb, g, cap, x_list[:]))
                # Include
                if self.problem.weights[i] <= cap:
                    new_x    = x_list[:]
                    new_x[i] = 1
                    new_g    = g + self.problem.values[i]
                    new_cap  = cap - self.problem.weights[i]
                    incl_lb  = _lp_bound(depth + 1, new_g, new_cap)
                    candidates.append((-incl_lb, new_g, new_cap, new_x))
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        best_sol = np.zeros(n, dtype=int)
        best_val = 0.0
        for neg_lb, g, cap, x_list in beam:
            x   = np.array(x_list, dtype=int)
            val = self.problem.fitness(x)
            if val > best_val:
                best_val = val
                best_sol = x

        if verbose:
            print(
                "[BFS] Beam sol  value: %.2f  weight: %.2f  feasible: %s"
                % (best_val,
                   float(best_sol @ self.problem.weights),
                   self.problem.is_feasible(best_sol))
            )

        return best_sol, best_val, [best_val]

    def solve_dfs(self, max_nodes=100_000, verbose=True):
        """
        Solve the knapsack with Depth-First Search and branch-and-bound pruning.

        Items are considered in descending value/weight ratio order. DFS explores
        include/exclude branches. LP-relaxation upper bounds prune branches that
        cannot improve the current best. Stops after max_nodes expansions.

        Parameters
        ----------
        max_nodes : int  Maximum number of DFS nodes to expand.
        verbose   : bool Print when a new best complete solution is found.

        Returns
        -------
        best_sol   : ndarray  Best binary solution vector.
        best_value : float    Total value of best solution.
        history    : list     Improving best values found during search.
        """
        n = self.problem.n_items
        ratio = self.problem.values / np.maximum(self.problem.weights, 1e-10)
        order = np.argsort(-ratio).tolist()

        def _lp_bound(depth, g, cap):
            ub = g; c = cap
            for idx in range(depth, n):
                i = order[idx]
                if self.problem.weights[i] <= c:
                    ub += self.problem.values[i]; c -= self.problem.weights[i]
                else:
                    ub += ratio[i] * c; break
            return ub

        # Warm-start with greedy fractional knapsack solution (integer constrained)
        best_sol = np.zeros(n, dtype=int)
        best_val = 0.0
        cap = self.problem.capacity
        for idx in range(n):
            i = order[idx]
            if self.problem.weights[i] <= cap:
                best_sol[i] = 1
                best_val   += self.problem.values[i]
                cap        -= self.problem.weights[i]
        history = [best_val]

        # DFS stack: (depth, g, cap, x_list)
        stack      = [(0, 0.0, self.problem.capacity, [0] * n)]
        nodes_seen = 0

        while stack and nodes_seen < max_nodes:
            depth, g, cap, x_list = stack.pop()
            nodes_seen += 1

            lb = _lp_bound(depth, g, cap)
            if lb <= best_val:
                continue

            if depth == n:
                val = g
                if val > best_val:
                    best_val = val
                    best_sol = np.array(x_list, dtype=int)
                    history.append(best_val)
                    if verbose:
                        print(
                            "[DFS] Nodes: %d  value: %.2f  weight: %.2f  feasible: %s"
                            % (nodes_seen, best_val,
                               float(best_sol @ self.problem.weights),
                               self.problem.is_feasible(best_sol))
                        )
                continue

            i = order[depth]
            # Push exclude branch first (include branch explored first via LIFO)
            excl_lb = _lp_bound(depth + 1, g, cap)
            if excl_lb > best_val:
                stack.append((depth + 1, g, cap, x_list[:]))

            if self.problem.weights[i] <= cap:
                new_x    = x_list[:]
                new_x[i] = 1
                new_g    = g + self.problem.values[i]
                new_cap  = cap - self.problem.weights[i]
                incl_lb  = _lp_bound(depth + 1, new_g, new_cap)
                if incl_lb > best_val:
                    stack.append((depth + 1, new_g, new_cap, new_x))

        return best_sol, best_val, history


if __name__ == "__main__":
    N_ITEMS = 20
    SEED    = 42

    problem = KnapsackProblem.generate(n_items=N_ITEMS, capacity_ratio=0.5, seed=SEED)

    print("=" * 56)
    print("  Knapsack Demo  -  %d items" % N_ITEMS)
    print("  Capacity : %.1f" % problem.capacity)
    print("=" * 56)

    solver = KnapsackSolver(problem)

    print("\n>>> Simulated Annealing")
    sa_sol, sa_val, _ = solver.solve_sa(T0=500.0, T_min=1e-3, max_iter=10_000, alpha=0.003, verbose=True)
    print("[SA] Value: %.2f  Weight: %.2f  Feasible: %s" % (sa_val, float(sa_sol @ problem.weights), problem.is_feasible(sa_sol)))

    print("\n>>> Genetic Algorithm")
    ga_sol, ga_val, _ = solver.solve_ga(pop_size=100, max_iter=500, verbose=True)
    print("[GA] Value: %.2f  Weight: %.2f  Feasible: %s" % (ga_val, float(ga_sol @ problem.weights), problem.is_feasible(ga_sol)))

    print("\n>>> Ant Colony Optimization")
    aco_sol, aco_val, _ = solver.solve_aco(n_ants=30, max_iter=200, verbose=True)
    print("[ACO] Value: %.2f  Weight: %.2f  Feasible: %s" % (aco_val, float(aco_sol @ problem.weights), problem.is_feasible(aco_sol)))

    print("\n>>> Cuckoo Search")
    cs_sol, cs_val, _ = solver.solve_cs(n_nests=25, max_iter=300, verbose=True)
    print("[CS] Value: %.2f  Weight: %.2f  Feasible: %s" % (cs_val, float(cs_sol @ problem.weights), problem.is_feasible(cs_sol)))

    print("\n>>> Artificial Bee Colony")
    abc_sol, abc_val, _ = solver.solve_abc(n_bees=30, max_iter=300, verbose=True)
    print("[ABC] Value: %.2f  Weight: %.2f  Feasible: %s" % (abc_val, float(abc_sol @ problem.weights), problem.is_feasible(abc_sol)))

    print("\n>>> Firefly Algorithm")
    fa_sol, fa_val, _ = solver.solve_fa(n_fireflies=30, max_iter=200, verbose=True)
    print("[FA] Value: %.2f  Weight: %.2f  Feasible: %s" % (fa_val, float(fa_sol @ problem.weights), problem.is_feasible(fa_sol)))

    print("\n>>> A* Search (Branch and Bound)")
    astar_sol, astar_val, _ = solver.solve_astar(verbose=True)
    print("[A*] Value: %.2f  Weight: %.2f  Feasible: %s" % (astar_val, float(astar_sol @ problem.weights), problem.is_feasible(astar_sol)))

    print("\n>>> Beam Search (BFS)")
    bfs_sol, bfs_val, _ = solver.solve_bfs(beam_width=20, verbose=True)
    print("[BFS] Value: %.2f  Weight: %.2f  Feasible: %s" % (bfs_val, float(bfs_sol @ problem.weights), problem.is_feasible(bfs_sol)))

    print("\n>>> DFS (Branch and Bound)")
    dfs_sol, dfs_val, _ = solver.solve_dfs(max_nodes=100_000, verbose=True)
    print("[DFS] Value: %.2f  Weight: %.2f  Feasible: %s" % (dfs_val, float(dfs_sol @ problem.weights), problem.is_feasible(dfs_sol)))

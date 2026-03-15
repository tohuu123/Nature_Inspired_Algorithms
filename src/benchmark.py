"""
Benchmark runner
================
Runs all Nature-Inspired and Traditional algorithms on continuous benchmark
functions and discrete problems, collects statistics, and returns structured
results that can be fed directly into the visualization module.

"""

import numpy as np
import time
import sys
import os

# ---------------------------------------------------------------------------
# Path handling — ensure 'src' package is importable regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "src" else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Continuous test functions ──────────────────────────────────────────────
from src.testing.continous.test_functions import sphere, rastrigin, rosenbrock, griewank, ackley

# ── Nature-inspired / metaheuristic algorithms ────────────────────────────
from src.biology.PSO import PSO
from src.biology.CS  import CS
from src.biology.FA  import FA
from src.biology.ABC import ABC
from src.biology.ACO import ACO
from src.evolution.GA   import GA
from src.evolution.DE   import DE
from src.human.TLBO import TLBO
from src.physics import SA

# ── Traditional algorithms ────────────────────────────────────────────────
from src.traditional.hill_climbing import HillClimbing
from src.traditional.graph_search  import (Grid, BFS, DFS, UCS, GBFS,
                                            AStar, Heuristic)

# ── Discrete problems ────────────────────────────────────────────────────
from src.testing.discrete_problems.TSP           import TSP, TSPSolver
from src.testing.discrete_problems.Knapsack      import KnapsackProblem, KnapsackSolver
from src.testing.discrete_problems.GraphColoring import GraphColoring, GraphColoringSolver

def _make_bounds(lo, hi, dim):
    return np.array([[lo, hi]] * dim, dtype=float)



# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                          BenchmarkRunner                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class BenchmarkRunner:
    """
    Orchestrates benchmarks across three domains:
        1. Continuous optimisation  (Sphere, Rastrigin, Rosenbrock, Griewank, Ackley)
        2. Discrete problems        (TSP, Knapsack, Graph Coloring)
        3. Graph-search / pathfinding (BFS, DFS, UCS, GBFS, A*, Hill Climbing)

    Parameters
    ----------
    n_trials  : int   Number of independent trials per algorithm (for statistical robustness).
    dim       : int   Dimensionality for the continuous test functions.
    max_iter  : int   Max iterations budget for metaheuristic algorithms.
    seed      : int   Base random seed (incremented per trial).
    """

    def __init__(self, n_trials=5, dim=10, max_iter=200, seed=42):
        self.n_trials = n_trials
        self.dim      = dim
        self.max_iter = max_iter
        self.seed     = seed

    def _as_1d_float_array(self, values, name):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"'{name}' must contain at least one value.")
        return arr

    def _parse_sensitivity_parameters(self, parameters):
        if isinstance(parameters, dict):
            needed = ["mutation_rate", "rho", "alpha"]
            missing = [k for k in needed if k not in parameters]
            if missing:
                raise ValueError(f"Missing sensitivity parameters: {missing}")
            return {
                "mutation_rate": self._as_1d_float_array(parameters["mutation_rate"], "mutation_rate"),
                "rho": self._as_1d_float_array(parameters["rho"], "rho"),
                "alpha": self._as_1d_float_array(parameters["alpha"], "alpha"),
            }

        if isinstance(parameters, (list, tuple)) and len(parameters) == 3:
            return {
                "mutation_rate": self._as_1d_float_array(parameters[0], "mutation_rate"),
                "rho": self._as_1d_float_array(parameters[1], "rho"),
                "alpha": self._as_1d_float_array(parameters[2], "alpha"),
            }

        raise ValueError(
            "parameters must be a dict with keys 'mutation_rate', 'rho', 'alpha' "
            "or a (mutation_rate, rho, alpha) tuple/list."
        )

    def run_parameters_sensitivity(self, parameters, continuous_func=sphere, TEST_CASE="test_case/medium_tsp_30.json", continuous_bounds=(-5.12, 5.12), ga_chrom_len=100, ga_pop_size=200, aco_n_cities=20, aco_n_ants=20, verbose=False):
        parsed = self._parse_sensitivity_parameters(parameters)
        mutation_rates = parsed["mutation_rate"]
        rho_values = parsed["rho"]
        alpha_values = parsed["alpha"]

        lo, hi = float(continuous_bounds[0]), float(continuous_bounds[1])
        if lo >= hi:
            raise ValueError("continuous_bounds must satisfy low < high.")

        bounds = _make_bounds(lo, hi, self.dim)
        
        # load the problem on small_tsp_10.json
        tsp = TSP(np.zeros((1,1))) # dummy init
        tsp.load_from_json(TEST_CASE)

        results = {
            "meta": {
                "n_trials": int(self.n_trials),
                "dim": int(self.dim),
                "max_iter": int(self.max_iter),
                "seed": int(self.seed),
                "aco_n_cities": int(aco_n_cities),
            },
            "GA": {
                "parameter_name": "mutation_rate",
                "objective": "maximize",
                "records": [],
            },
            "ACO": {
                "parameter_name": "rho",
                "objective": "minimize",
                "records": [],
            },
            "SA": {
                "parameter_name": "alpha",
                "objective": "minimize",
                "records": [],
            },
        }
        TSP_solver = TSPSolver(tsp)

        for rate in mutation_rates:
            runs = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    out = TSP_solver.solve_ga(
                        pop_size=int(ga_pop_size),
                        max_iter=int(self.max_iter),
                        mutation_rate=float(rate),
                    )
                    
                    t0 = time.perf_counter()
                    elapsed = time.perf_counter() - t0
                    runs.append({
                        "score": float(out[1]),
                        "time": float(elapsed),
                        "history": list(out[2]) if len(out) > 2 else [],
                    })
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] GA/mutation_rate={rate} trial {t}: {e}")
                    runs.append({"score": float("-inf"), "time": 0.0, "history": []})

            scores = [r["score"] for r in runs if np.isfinite(r["score"])]
            times = [r["time"] for r in runs]
            results["GA"]["records"].append({
                "value": float(rate),
                "trials": runs,
                "mean_score": float(np.mean(scores)) if scores else float("nan"),
                "std_score": float(np.std(scores)) if scores else float("nan"),
                "mean_time": float(np.mean(times)) if times else 0.0,
            })

        for rho in rho_values:
            runs = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    aco = ACO(
                        dist_matrix=tsp.dist_matrix,
                        n_ants=int(aco_n_ants),
                        n_iter=int(self.max_iter),
                        rho=float(rho),
                    )
                    t0 = time.perf_counter()
                    out = aco.run(verbose=False)
                    elapsed = time.perf_counter() - t0
                    runs.append({
                        "score": float(out[1]),
                        "time": float(elapsed),
                        "history": list(out[2]) if len(out) > 2 else [],
                    })
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] ACO/rho={rho} trial {t}: {e}")
                    runs.append({"score": float("inf"), "time": 0.0, "history": []})

            scores = [r["score"] for r in runs if np.isfinite(r["score"])]
            times = [r["time"] for r in runs]
            results["ACO"]["records"].append({
                "value": float(rho),
                "trials": runs,
                "mean_score": float(np.mean(scores)) if scores else float("nan"),
                "std_score": float(np.std(scores)) if scores else float("nan"),
                "mean_time": float(np.mean(times)) if times else 0.0,
            })

        for alpha in alpha_values:
            runs = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:    
                    sa = SA(max_iter=int(self.max_iter), alpha=float(alpha))
                    t0 = time.perf_counter()
                    out = sa.run(obj_func=continuous_func, bounds=bounds, verbose=True)
                    elapsed = time.perf_counter() - t0
                    runs.append({
                        "score": float(out[1]),
                        "time": float(elapsed),
                        "history": list(out[2]) if len(out) > 2 else [],
                    })
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] SA/alpha={alpha} trial {t}: {e}")
                    runs.append({"score": float("inf"), "time": 0.0, "history": []})

            scores = [r["score"] for r in runs if np.isfinite(r["score"])]
            times = [r["time"] for r in runs]
            results["SA"]["records"].append({
                "value": float(alpha),
                "trials": runs,
                "mean_score": float(np.mean(scores)) if scores else float("nan"),
                "std_score": float(np.std(scores)) if scores else float("nan"),
                "mean_time": float(np.mean(times)) if times else 0.0,
            })

        return results

    def _population_diversity(self, population):
        return float(np.mean(np.std(population, axis=0)))

    def _track_pso_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        n_dims = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]
        v_max = algo.v_max_ratio * (hi - lo)
        v_min = -v_max

        positions = lo + np.random.rand(algo.n_particles, n_dims) * (hi - lo)
        velocities = np.zeros((algo.n_particles, n_dims))
        scores = np.array([obj_func(p) for p in positions])
        p_best_pos = positions.copy()
        p_best_score = scores.copy()
        g_best_idx = int(np.argmin(scores))
        g_best = p_best_pos[g_best_idx].copy()

        diversity = []
        for _ in range(algo.max_iter):
            r1 = np.random.rand(algo.n_particles, n_dims)
            r2 = np.random.rand(algo.n_particles, n_dims)
            cognitive = algo.c1 * r1 * (p_best_pos - positions)
            social = algo.c2 * r2 * (g_best - positions)
            velocities = algo.w * velocities + cognitive + social
            velocities = np.clip(velocities, v_min, v_max)

            positions = positions + velocities
            positions = np.clip(positions, lo, hi)
            scores = np.array([obj_func(p) for p in positions])

            improved = scores < p_best_score
            p_best_pos[improved] = positions[improved].copy()
            p_best_score[improved] = scores[improved]
            g_best = p_best_pos[int(np.argmin(p_best_score))].copy()
            diversity.append(self._population_diversity(positions))

        return diversity

    def _track_cs_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        n_dims = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]
        from math import gamma as gamma_func, pi

        num = gamma_func(1 + algo.beta_levy) * np.sin(pi * algo.beta_levy / 2)
        den = gamma_func((1 + algo.beta_levy) / 2) * algo.beta_levy * (2 ** ((algo.beta_levy - 1) / 2))
        sigma_u = (num / den) ** (1.0 / algo.beta_levy)

        nests = lo + np.random.rand(algo.n_nests, n_dims) * (hi - lo)
        scores = np.array([obj_func(n) for n in nests])
        diversity = []

        for _ in range(algo.max_iter):
            for i in range(algo.n_nests):
                u = np.random.randn(n_dims) * sigma_u
                v = np.random.randn(n_dims)
                step = u / (np.abs(v) ** (1.0 / algo.beta_levy))
                candidate = np.clip(nests[i] + algo.alpha * step, lo, hi)
                cand_score = obj_func(candidate)

                j = np.random.randint(algo.n_nests)
                if cand_score < scores[j]:
                    nests[j] = candidate
                    scores[j] = cand_score

            n_abandon = max(1, int(algo.pa * algo.n_nests))
            worst_idx = np.argsort(scores)[-n_abandon:]
            for idx in worst_idx:
                nests[idx] = lo + np.random.rand(n_dims) * (hi - lo)
                scores[idx] = obj_func(nests[idx])

            diversity.append(self._population_diversity(nests))

        return diversity

    def _track_fa_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        n_dims = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]

        positions = lo + np.random.rand(algo.n_fireflies, n_dims) * (hi - lo)
        scores = np.array([obj_func(p) for p in positions])
        alpha = algo.alpha
        diversity = []

        for _ in range(algo.max_iter):
            new_positions = positions.copy()
            for i in range(algo.n_fireflies):
                for j in range(algo.n_fireflies):
                    if scores[j] < scores[i]:
                        diff = positions[j] - positions[i]
                        r_sq = float(np.dot(diff, diff))
                        beta = algo.beta0 * np.exp(-algo.gamma * r_sq)
                        epsilon = np.random.rand(n_dims) - 0.5
                        step = beta * diff + alpha * epsilon
                        new_positions[i] = np.clip(new_positions[i] + step, lo, hi)

            positions = new_positions
            scores = np.array([obj_func(p) for p in positions])
            alpha *= algo.alpha_decay
            diversity.append(self._population_diversity(positions))

        return diversity

    def _track_de_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        dims = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]

        pop = lo + np.random.rand(algo.pop_size, dims) * (hi - lo)
        obj_all = np.array([obj_func(ind) for ind in pop])
        diversity = []

        for _ in range(algo.max_iter):
            for j in range(algo.pop_size):
                candidates = [c for c in range(algo.pop_size) if c != j]
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
                mutated = np.clip(a + algo.F * (b - c), lo, hi)
                p = np.random.rand(dims)
                trial = np.array([mutated[d] if p[d] < algo.CR else pop[j, d] for d in range(dims)])
                obj_trial = obj_func(trial)
                if obj_trial < obj_all[j]:
                    pop[j] = trial
                    obj_all[j] = obj_trial

            diversity.append(self._population_diversity(pop))

        return diversity

    def _track_tlbo_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        dim = len(bounds)
        lb, ub = bounds[:, 0], bounds[:, 1]

        pop = lb + np.random.rand(algo.pop_size, dim) * (ub - lb)
        fitness = np.array([obj_func(ind) for ind in pop])
        diversity = []

        for _ in range(algo.max_iter):
            teacher = pop[np.argmin(fitness)].copy()
            mean_pop = pop.mean(axis=0)
            tf = np.random.randint(1, 3)

            for i in range(algo.pop_size):
                r = np.random.rand(dim)
                new = np.clip(pop[i] + r * (teacher - tf * mean_pop), lb, ub)
                new_fit = obj_func(new)
                if new_fit < fitness[i]:
                    pop[i] = new
                    fitness[i] = new_fit

            for i in range(algo.pop_size):
                j = i
                while j == i:
                    j = np.random.randint(0, algo.pop_size)
                r = np.random.rand(dim)
                if fitness[i] < fitness[j]:
                    new = pop[i] + r * (pop[i] - pop[j])
                else:
                    new = pop[i] + r * (pop[j] - pop[i])
                new = np.clip(new, lb, ub)
                new_fit = obj_func(new)
                if new_fit < fitness[i]:
                    pop[i] = new
                    fitness[i] = new_fit

            diversity.append(self._population_diversity(pop))

        return diversity

    def _track_hc_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        dim = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]

        current_sol = lo + np.random.rand(dim) * (hi - lo)
        current_cost = obj_func(current_sol)
        diversity = []

        for _ in range(algo.max_iter):
            sigma = algo.step_size * (hi - lo)
            deltas = np.random.randn(algo.n_neighbours, dim) * sigma
            candidates = np.clip(current_sol + deltas, lo, hi)
            candidate_costs = np.array([obj_func(c) for c in candidates])

            diversity.append(self._population_diversity(candidates))

            best_idx = int(np.argmin(candidate_costs))
            best_cost = candidate_costs[best_idx]
            if best_cost >= current_cost:
                break

            current_sol = candidates[best_idx]
            current_cost = best_cost

        return diversity

    def _track_sa_diversity(self, algo, obj_func, bounds):
        bounds = np.asarray(bounds, dtype=float)
        dim = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]

        current_sol = lo + np.random.rand(dim) * (hi - lo)
        current_cost = obj_func(current_sol)
        diversity = []

        for iteration in range(1, algo.max_iter + 1):
            t = algo.T0 * np.exp(-algo.alpha * iteration)
            if t <= algo.T_min:
                break

            sigma = algo.step_scale * (hi - lo) * (t / algo.T0)
            probe_count = max(8, dim)
            probes = current_sol + np.random.randn(probe_count, dim) * sigma
            probes = np.clip(probes, lo, hi)
            diversity.append(self._population_diversity(probes))

            candidate_sol = probes[np.random.randint(probe_count)]
            candidate_cost = obj_func(candidate_sol)
            delta = candidate_cost - current_cost

            if delta < 0 or np.random.rand() < np.exp(-delta / t):
                current_sol = candidate_sol
                current_cost = candidate_cost

        return diversity

    def _track_algorithm_diversity(self, algo, obj_func, bounds, algo_name=None):
        name = str(algo_name).strip().lower() if algo_name is not None else ""
        if isinstance(algo, PSO):
            return self._track_pso_diversity(algo, obj_func, bounds)
        if isinstance(algo, CS):
            return self._track_cs_diversity(algo, obj_func, bounds)
        if isinstance(algo, FA):
            return self._track_fa_diversity(algo, obj_func, bounds)
        if isinstance(algo, DE):
            return self._track_de_diversity(algo, obj_func, bounds)
        if isinstance(algo, TLBO):
            return self._track_tlbo_diversity(algo, obj_func, bounds)
        if isinstance(algo, HillClimbing):
            return self._track_hc_diversity(algo, obj_func, bounds)
        if isinstance(algo, SA):
            return self._track_sa_diversity(algo, obj_func, bounds)

        if name in ("pso",):
            return self._track_pso_diversity(algo, obj_func, bounds)
        if name in ("cs",):
            return self._track_cs_diversity(algo, obj_func, bounds)
        if name in ("fa",):
            return self._track_fa_diversity(algo, obj_func, bounds)
        if name in ("de",):
            return self._track_de_diversity(algo, obj_func, bounds)
        if name in ("tlbo",):
            return self._track_tlbo_diversity(algo, obj_func, bounds)
        if name in ("hc", "hill climbing", "hill_climbing"):
            return self._track_hc_diversity(algo, obj_func, bounds)
        if name in ("sa", "simulated annealing", "simulated_annealing"):
            return self._track_sa_diversity(algo, obj_func, bounds)
        raise ValueError("Unsupported algorithm for population-diversity tracking")

    def bench_exploration_exploitation(self, CONTINUOUS_FUNCTIONS, continuous_algorithms, verbose=False):
        """
        Benchmark exploration vs exploitation using population diversity.

        Diversity at each iteration is measured as:
            mean(std(population, axis=0))

        Returns
        -------
        dict
            results[function_name][algorithm_name] = [
                {"diversity": [d1, d2, ...], "time": elapsed_seconds}, ...
            ]
        """
        results = {}
        for fname, (funcc, (lo, hi)) in CONTINUOUS_FUNCTIONS.items():
            bounds = _make_bounds(lo, hi, self.dim)
            results[fname] = {}

            for aname, algo in continuous_algorithms.items():
                trials = []
                for t in range(self.n_trials):
                    np.random.seed(self.seed + t)
                    try:
                        t0 = time.perf_counter()
                        diversity_hist = self._track_algorithm_diversity(algo, funcc, bounds, algo_name=aname)
                        elapsed = time.perf_counter() - t0
                        trials.append({"diversity": list(diversity_hist), "time": elapsed})
                    except Exception as e:
                        if verbose:
                            print(f"  [WARN] Diversity/{aname} on {fname} trial {t}: {e}")
                        trials.append({"diversity": [], "time": 0})

                results[fname][aname] = trials
                if verbose:
                    lengths = [len(tr["diversity"]) for tr in trials if tr["diversity"]]
                    mean_len = float(np.mean(lengths)) if lengths else 0.0
                    print(f"  {fname:>12s} | {aname:<12s} | avg_len={mean_len:.1f}")

        return results

    # ──────────────────────────────────────────────────────────────────────
    #  1.  Continuous Benchmark
    # ──────────────────────────────────────────────────────────────────────
    def run_continuous_benchmarks(self, CONTINUOUS_FUNCTIONS, continuous_algorithms, verbose=False):
        """
        Run all metaheuristic + Hill Climbing + SA on every continuous function.

        Returns
        -------
        results : dict
            results[func_name][algo_name] = list of dicts
            Each dict: {score, time, history}
        """
        results = {}
        for fname, (funcc, (lo, hi)) in CONTINUOUS_FUNCTIONS.items():
            boundds = _make_bounds(lo, hi, self.dim)
            results[fname] = {} 
            
            algo_factories = continuous_algorithms
            
            for aname, algo in algo_factories.items():
                trials = []
                for t in range(self.n_trials):
                    np.random.seed(self.seed + t)
                    try:
                        t0 = time.perf_counter()
                        out = algo.run(obj_func = funcc, bounds = boundds, verbose=False)
                        elapsed = time.perf_counter() - t0
                        # out = (best_sol, best_score, history)
                        trials.append({
                            "score":   out[1],
                            "time":    elapsed,
                            "history": list(out[2]) if len(out) > 2 else [],
                        })
                    except Exception as e:
                        if verbose:
                            print(f"  [WARN] {aname} on {fname} trial {t}: {e}")
                        trials.append({"score": float("inf"), "time": 0, "history": []})

                results[fname][aname] = trials
                if verbose:
                    scores = [tr["score"] for tr in trials]
                    print(f"  {fname:>12s} | {aname:<1s} | mean={np.mean(scores):.4e}  std={np.std(scores):.4e}")

        return results

    # ──────────────────────────────────────────────────────────────────────
    #  1b. Scalability Benchmark (sweep over multiple dimensions)
    # ──────────────────────────────────────────────────────────────────────
    def run_scalability_benchmarks(self, dims, CONTINUOUS_FUNCTIONS,
                                   continuous_algorithms, verbose=False):
        """
        Run continuous benchmarks across multiple problem dimensions.

        Parameters
        ----------
        dims : list[int]
            Dimensions to sweep, e.g. ``[10, 30, 50]``.
        CONTINUOUS_FUNCTIONS : dict
            Same format as ``run_continuous_benchmarks``.
        continuous_algorithms : dict
            Same format as ``run_continuous_benchmarks``.
        verbose : bool
            If True, print progress for every (dim, function, algorithm).

        Returns
        -------
        results : dict
            ``results[dim][func_name][algo_name]`` = list of trial dicts,
            each containing ``{score, time, history}``.
            Identical nested structure to ``run_continuous_benchmarks``,
            just wrapped by an outer dimension key.

        Example
        -------
        >>> scalability = runner.run_scalability_benchmarks(
        ...     dims=[10, 30, 50],
        ...     CONTINUOUS_FUNCTIONS=CONTINUOUS_FUNCTIONS,
        ...     continuous_algorithms=continuous_algorithms,
        ... )
        >>> scalability[10]["Sphere"]["PSO"]   # list of trial dicts for dim=10
        """
        scalability_results = {}
        original_dim = self.dim

        for dim in dims:
            if verbose:
                print(f"\n{'='*50}")
                print(f"  Dimension = {dim}")
                print(f"{'='*50}")
            self.dim = dim
            scalability_results[dim] = self.run_continuous_benchmarks(
                CONTINUOUS_FUNCTIONS, continuous_algorithms, verbose=verbose
            )

        # Restore original dimension
        self.dim = original_dim
        return scalability_results


    # ──────────────────────────────────────────────────────────────────────
    #  2.  Discrete Benchmark  (TSP, Knapsack, Graph Coloring)
    # ──────────────────────────────────────────────────────────────────────
    def _bench_tsp(self, algo_runs, tsp, verbose=False):
        """Benchmark all TSPSolver methods. Results include the TSP instance and best tour per trial."""
        res = {"problem": tsp}
        for aname, run_fn in algo_runs.items():
            trials = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    t0 = time.perf_counter()
                    out = run_fn()
                    elapsed = time.perf_counter() - t0
                    # out = (best_tour, best_dist, history)
                    trials.append({"score": out[1], "time": elapsed,
                                   "history": list(out[2]) if len(out) > 2 else [],
                                   "solution": out[0]})
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] TSP/{aname} trial {t}: {e}")
                    trials.append({"score": float("inf"), "time": 0, "history": [], "solution": None})
            res[aname] = trials
            if verbose:
                scores = [tr["score"] for tr in trials]
                print(f"  TSP | {aname:<20s} | mean={np.mean(scores):.2f}")
        return res

    def _bench_knapsack(self, algo_runs, knapsack, verbose):
        """Benchmark all KnapsackSolver methods. Results include the problem instance and best solution per trial."""
        res = {"problem": knapsack}

        for aname, run_fn in algo_runs.items():
            trials = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    t0 = time.perf_counter()
                    out = run_fn()
                    elapsed = time.perf_counter() - t0
                    # out = (best_sol, best_value, history)
                    trials.append({"score": out[1], "time": elapsed,
                                   "history": list(out[2]) if len(out) > 2 else [],
                                   "solution": out[0]})
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] Knapsack/{aname} trial {t}: {e}")
                    trials.append({"score": 0, "time": 0, "history": [], "solution": None})
            res[aname] = trials
            if verbose:
                scores = [tr["score"] for tr in trials]
                print(f"  Knapsack | {aname:<20s} | mean={np.mean(scores):.2f}")
        return res

    def _bench_graph_coloring(self, algo_runs, gc, verbose=False):
        """Benchmark all GraphColoringSolver methods. Results include the problem instance and best coloring per trial."""
        res = {"problem": gc}

        for aname, run_fn in algo_runs.items():
            trials = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    t0 = time.perf_counter()
                    out = run_fn()
                    elapsed = time.perf_counter() - t0
                    # out = (best_coloring, best_colors, history)
                    trials.append({"score": out[1], "time": elapsed,
                                   "history": list(out[2]) if len(out) > 2 else [],
                                   "solution": out[0]})
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] GC/{aname} trial {t}: {e}")
                    trials.append({"score": float("inf"), "time": 0, "history": [], "solution": None})
            res[aname] = trials
            if verbose:
                scores = [tr["score"] for tr in trials]
                print(f"  Graph Coloring | {aname:<20s} | mean={np.mean(scores):.2f}")
        return res

    # ──────────────────────────────────────────────────────────────────────
    #  3.  Graph-Search Pathfinding Benchmark
    # ──────────────────────────────────────────────────────────────────────
    def run_graph_search_benchmarks(self, grid_configs=None, verbose=False):
        """
        Run BFS, DFS, UCS, GBFS, A* (Manhattan), A* (Euclidean) on grids.

        Parameters
        ----------
        grid_configs : list of dict, optional
            Each dict contains kwargs for Grid constructor.
            Defaults to three grids of increasing size.

        Returns
        -------
        results : dict
            results[grid_label][algo_name] = {path_length, nodes_expanded, elapsed_ms}
        """
        if grid_configs is None:
            grid_configs = [
                {"label": "Small 15×15",  "rows": 15, "cols": 15, "obstacle_ratio": 0.25, "seed": self.seed},
                {"label": "Medium 25×25", "rows": 25, "cols": 25, "obstacle_ratio": 0.30, "seed": self.seed},
                {"label": "Large 40×40",  "rows": 40, "cols": 40, "obstacle_ratio": 0.25, "seed": self.seed},
            ]

        results = {}
        for cfg in grid_configs:
            label = cfg.pop("label", f"{cfg.get('rows')}x{cfg.get('cols')}")
            grid = Grid(**cfg)
            cfg["label"] = label  # restore

            algorithms = [
                ("BFS",              BFS(grid)),
                ("DFS",              DFS(grid)),
                ("UCS",              UCS(grid)),
                ("GBFS",             GBFS(grid, Heuristic.manhattan)),
                ("A* (Manhattan)",   AStar(grid, Heuristic.manhattan)),
                ("A* (Euclidean)",   AStar(grid, Heuristic.euclidean)),
            ]

            grid_results = {}
            for aname, algo in algorithms:
                path, visited = algo.run()
                grid_results[aname] = {
                    "path_length":    len(path),
                    "nodes_expanded": algo.nodes_expanded,
                    "elapsed_ms":     algo.elapsed * 1000,
                    "path":           path,
                    "visited":        visited,
                }
                if verbose:
                    print(f"  {label} | {aname:<18s} | path={len(path):>4d}  expanded={algo.nodes_expanded:>5d}  time={algo.elapsed*1000:.3f}ms")

            results[label] = {"grid": grid, "algos": grid_results}

        return results

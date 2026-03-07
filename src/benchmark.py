"""
Benchmark runner
================
Runs all Nature-Inspired and Traditional algorithms on continuous benchmark
functions and discrete problems, collects statistics, and returns structured
results that can be fed directly into the visualization module.

Usage
-----
    from src.benchmark import BenchmarkRunner
    runner = BenchmarkRunner(n_trials=5, seed=42)
    results = runner.run_continuous_benchmarks()
    results_discrete = runner.run_discrete_benchmarks()
    results_graph = runner.run_graph_search_benchmarks()
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
from src.physics.SA  import SimulatedAnnealing

# ── Traditional algorithms ────────────────────────────────────────────────
from src.traditional.hill_climbing import HillClimbing
from src.traditional.graph_search  import (Grid, BFS, DFS, UCS, GBFS,
                                            AStar, Heuristic)

# ── Discrete problems ────────────────────────────────────────────────────
from src.testing.discrete_problems.TSP           import TSP, TSPSolver
from src.testing.discrete_problems.Knapsack      import KnapsackProblem
from src.testing.discrete_problems.GraphColoring import GraphColoring, GraphColoringSolver


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                       Helper: function bounds                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝
CONTINUOUS_FUNCTIONS = {
    "Sphere":     (sphere,     (-100, 100)),
    "Rastrigin":  (rastrigin,  (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, (-5, 10)),
    "Griewank":   (griewank,   (-600, 600)),
    "Ackley":     (ackley,     (-32.768, 32.768)),
}


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

    # ──────────────────────────────────────────────────────────────────────
    #  1.  Continuous Benchmark
    # ──────────────────────────────────────────────────────────────────────
    def run_continuous_benchmarks(self, verbose=False):
        """
        Run all metaheuristic + Hill Climbing + SA on every continuous function.

        Returns
        -------
        results : dict
            results[func_name][algo_name] = list of dicts
            Each dict: {score, time, history}
        """
        results = {}
        for fname, (func, (lo, hi)) in CONTINUOUS_FUNCTIONS.items():
            bounds = _make_bounds(lo, hi, self.dim)
            results[fname] = {}

            algo_factories = self._continuous_algo_factories(func, bounds)

            for aname, factory in algo_factories.items():
                trials = []
                for t in range(self.n_trials):
                    np.random.seed(self.seed + t)
                    try:
                        algo = factory()
                        t0 = time.perf_counter()
                        out = algo.run(verbose=False)
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
                    print(f"  {fname:>12s} | {aname:<20s} | mean={np.mean(scores):.4e}  std={np.std(scores):.4e}")

        return results

    def _continuous_algo_factories(self, func, bounds):
        """Return {name: lambda()->algo} for every continuous algorithm."""
        dim = self.dim
        mi  = self.max_iter
        return {
            "PSO":  lambda f=func, b=bounds: PSO(obj_func=f, bounds=b, n_particles=30, max_iter=mi),
            "CS":   lambda f=func, b=bounds: CS(obj_func=f, bounds=b, n_nests=25, max_iter=mi),
            "FA":   lambda f=func, b=bounds: FA(obj_func=f, bounds=b, n_fireflies=25, max_iter=mi),
            "DE":   lambda f=func, b=bounds: DE(bounds=b, obj=f, pop_size=30, max_iter=mi),
            "SA":   lambda f=func, b=bounds: SimulatedAnnealing(obj_func=f, bounds=b, max_iter=mi*50, T0=1000, alpha=0.005),
            "TLBO": lambda f=func, b=bounds: TLBO(obj_func=f, bounds=b, pop_size=30, max_iter=mi),
            "Hill Climbing": lambda f=func, b=bounds: HillClimbing(obj_func=f, bounds=b, dim=dim, max_iter=mi*10),
        }

    # ──────────────────────────────────────────────────────────────────────
    #  2.  Discrete Benchmark  (TSP, Knapsack, Graph Coloring)
    # ──────────────────────────────────────────────────────────────────────
    def run_discrete_benchmarks(self, verbose=False):
        """
        Run applicable algorithms on discrete problems.

        Returns
        -------
        results : dict
            results[problem_name][algo_name] = list of {score, time, history}
        """
        results = {}

        # ── TSP ───────────────────────────────────────────────────────────
        tsp = TSP.generate(n_cities=15, seed=self.seed)
        results["TSP"] = self._bench_tsp(tsp, verbose)

        # ── Knapsack ─────────────────────────────────────────────────────
        knapsack = KnapsackProblem.generate(n_items=20, capacity_ratio=0.5, seed=self.seed)
        results["Knapsack"] = self._bench_knapsack(knapsack, verbose)

        # ── Graph Coloring ───────────────────────────────────────────────
        gc = GraphColoring.generate(n_vertices=20, edge_probability=0.4, seed=self.seed)
        results["Graph Coloring"] = self._bench_graph_coloring(gc, verbose)

        return results

    def _bench_tsp(self, tsp, verbose):
        """Benchmark algorithms on TSP."""
        res = {}
        solver = TSPSolver(tsp)
        mi = min(self.max_iter, 500)

        algo_runs = {
            "ACO": lambda: ACO(dist_matrix=tsp.dist_matrix, n_ants=20, n_iter=mi).run(verbose=False),
            "SA (TSP)":  lambda: solver.solve_sa(max_iter=mi*20, verbose=False),
            "GA (TSP)":  lambda: solver.solve_ga(pop_size=100, max_iter=mi, verbose=False),
        }

        for aname, run_fn in algo_runs.items():
            trials = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    t0 = time.perf_counter()
                    out = run_fn()
                    elapsed = time.perf_counter() - t0
                    trials.append({"score": out[1], "time": elapsed,
                                   "history": list(out[2]) if len(out) > 2 else []})
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] TSP/{aname} trial {t}: {e}")
                    trials.append({"score": float("inf"), "time": 0, "history": []})
            res[aname] = trials
            if verbose:
                scores = [tr["score"] for tr in trials]
                print(f"  TSP | {aname:<20s} | mean={np.mean(scores):.2f}")
        return res

    def _bench_knapsack(self, knapsack, verbose):
        """Benchmark GA and ABC on Knapsack (maximisation → negate for consistency)."""
        res = {}
        n_items = knapsack.n_items

        algo_runs = {
            "GA (Knapsack)":  lambda: GA(fitness_func=knapsack.fitness, chrom_len=n_items,
                                          pop_size=100, max_iter=self.max_iter).run(verbose=False),
            "ABC (Knapsack)": lambda: ABC(fitness_func=knapsack.fitness, n_dims=n_items,
                                           n_bees=30, max_iter=self.max_iter).run(verbose=False),
        }
        for aname, run_fn in algo_runs.items():
            trials = []
            for t in range(self.n_trials):
                np.random.seed(self.seed + t)
                try:
                    t0 = time.perf_counter()
                    out = run_fn()
                    elapsed = time.perf_counter() - t0
                    # GA/ABC maximise → store as-is (higher = better)
                    trials.append({"score": out[1], "time": elapsed,
                                   "history": list(out[2]) if len(out) > 2 else []})
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] Knapsack/{aname} trial {t}: {e}")
                    trials.append({"score": 0, "time": 0, "history": []})
            res[aname] = trials
            if verbose:
                scores = [tr["score"] for tr in trials]
                print(f"  Knapsack | {aname:<20s} | mean={np.mean(scores):.2f}")
        return res

    def _bench_graph_coloring(self, gc, verbose):
        """Benchmark SA and GA on Graph Coloring."""
        res = {}
        n_colors = gc.n_vertices  # upper-bound colors
        solver = GraphColoringSolver(gc, n_colors=n_colors)
        mi = self.max_iter

        algo_runs = {
            "SA (GC)": lambda: solver.solve_sa(max_iter=mi*20, verbose=False),
            "GA (GC)": lambda: solver.solve_ga(pop_size=100, max_iter=mi, verbose=False),
        }
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
                                   "history": list(out[2]) if len(out) > 2 else []})
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] GC/{aname} trial {t}: {e}")
                    trials.append({"score": float("inf"), "time": 0, "history": []})
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

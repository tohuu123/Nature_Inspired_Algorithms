#!/usr/bin/env python
"""
run_benchmark.py  -  Main entry point
=====================================
Chạy tất cả benchmark, thống kê, va xuất biểu đồ PNG vao thư mục ``output/``.

Cách dùng
---------
    python run_benchmark.py                   # chạy tất cả (mặc định 5 trials)
    python run_benchmark.py --trials 10       # 10 trials
    python run_benchmark.py --dim 20          # 20-D test functions
    python run_benchmark.py --fast            # 3 trials, 100 iter (nhanh để test)
    python run_benchmark.py --only continuous # chỉ benchmark continuous
    python run_benchmark.py --only discrete   # chỉ discrete problems
    python run_benchmark.py --only graph      # chỉ graph search
"""

import argparse
import os
import sys
import time
import csv

# ── ensure project root is on sys.path ─────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.benchmark import BenchmarkRunner
from src.visualization.visualize import (
    plot_convergence, plot_box_scores, plot_time_comparison,
    plot_heatmap_ranking, plot_summary_table, plot_overall_comparison,
    plot_discrete_convergence, plot_discrete_bar,
    plot_pathfinding_grids, plot_pathfinding_metrics, plot_pathfinding_summary_table,
)

import numpy as np


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                        CSV export helpers                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def export_continuous_csv(results, save_dir):
    """Write continuous benchmark stats to CSV."""
    path = os.path.join(save_dir, "continuous_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Algorithm", "Mean", "Std", "Best", "Worst",
                     "Median", "MeanTime_s"])
        for fname, algos in results.items():
            for aname, trials in algos.items():
                scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
                times  = [tr["time"]  for tr in trials]
                if not scores:
                    continue
                w.writerow([
                    fname, aname,
                    f"{np.mean(scores):.6e}", f"{np.std(scores):.6e}",
                    f"{np.min(scores):.6e}", f"{np.max(scores):.6e}",
                    f"{np.median(scores):.6e}", f"{np.mean(times):.4f}",
                ])
    print(f"  [OK]  {path}")


def export_discrete_csv(results, save_dir):
    """Write discrete benchmark stats to CSV."""
    path = os.path.join(save_dir, "discrete_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Problem", "Algorithm", "Mean", "Std", "Best", "Worst", "MeanTime_s"])
        for pname, algos in results.items():
            for aname, trials in algos.items():
                scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
                times  = [tr["time"]  for tr in trials]
                if not scores:
                    continue
                w.writerow([
                    pname, aname,
                    f"{np.mean(scores):.4f}", f"{np.std(scores):.4f}",
                    f"{np.min(scores):.4f}", f"{np.max(scores):.4f}",
                    f"{np.mean(times):.4f}",
                ])
    print(f"  [OK]  {path}")


def export_graph_csv(results, save_dir):
    """Write graph search stats to CSV."""
    path = os.path.join(save_dir, "graph_search_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Grid", "Algorithm", "PathLength", "NodesExpanded", "Time_ms"])
        for label, data in results.items():
            for aname, m in data["algos"].items():
                w.writerow([
                    label, aname,
                    m["path_length"], m["nodes_expanded"],
                    f"{m['elapsed_ms']:.4f}",
                ])
    print(f"  [OK]  {path}")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                               main()                                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Nature-Inspired vs Traditional algorithms and generate PNG charts.")
    parser.add_argument("--trials",  type=int, default=5,   help="Number of independent trials (default: 5)")
    parser.add_argument("--dim",     type=int, default=10,  help="Dimensionality for continuous functions (default: 10)")
    parser.add_argument("--iter",    type=int, default=200, help="Max iterations for metaheuristics (default: 200)")
    parser.add_argument("--seed",    type=int, default=42,  help="Base random seed (default: 42)")
    parser.add_argument("--outdir",  type=str, default="output", help="Output directory (default: output)")
    parser.add_argument("--fast",    action="store_true",   help="Quick test: 3 trials, 100 iter")
    parser.add_argument("--only",    type=str, choices=["continuous", "discrete", "graph", "all"],
                        default="all", help="Run only a specific benchmark type")
    args = parser.parse_args()

    if args.fast:
        args.trials = 3
        args.iter   = 100

    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)

    runner = BenchmarkRunner(
        n_trials=args.trials,
        dim=args.dim,
        max_iter=args.iter,
        seed=args.seed,
    )

    total_t0 = time.perf_counter()

    # ── 1. Continuous ─────────────────────────────────────────────────────
    if args.only in ("all", "continuous"):
        print("\n" + "=" * 60)
        print("  CONTINUOUS OPTIMISATION BENCHMARKS")
        print("=" * 60)
        cont_results = runner.run_continuous_benchmarks(verbose=True)
        print("\n  Generating continuous charts ...")
        plot_convergence(cont_results, save_dir)
        plot_box_scores(cont_results, save_dir)
        plot_time_comparison(cont_results, save_dir)
        plot_heatmap_ranking(cont_results, save_dir)
        plot_summary_table(cont_results, save_dir)
        plot_overall_comparison(cont_results, save_dir)
        export_continuous_csv(cont_results, save_dir)

    # ── 2. Discrete ──────────────────────────────────────────────────────
    if args.only in ("all", "discrete"):
        print("\n" + "=" * 60)
        print("  DISCRETE PROBLEM BENCHMARKS")
        print("=" * 60)
        disc_results = runner.run_discrete_benchmarks(verbose=True)
        print("\n  Generating discrete charts ...")
        plot_discrete_convergence(disc_results, save_dir)
        plot_discrete_bar(disc_results, save_dir)
        export_discrete_csv(disc_results, save_dir)

    # ── 3. Graph Search ──────────────────────────────────────────────────
    if args.only in ("all", "graph"):
        print("\n" + "=" * 60)
        print("  GRAPH-SEARCH PATHFINDING BENCHMARKS")
        print("=" * 60)
        graph_results = runner.run_graph_search_benchmarks(verbose=True)
        print("\n  Generating pathfinding charts ...")
        plot_pathfinding_grids(graph_results, save_dir)
        plot_pathfinding_metrics(graph_results, save_dir)
        plot_pathfinding_summary_table(graph_results, save_dir)
        export_graph_csv(graph_results, save_dir)

    elapsed = time.perf_counter() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  DONE  — total time {elapsed:.1f}s")
    print(f"  All outputs saved to:  {os.path.abspath(save_dir)}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

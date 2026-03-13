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

# ── ensure project root is on sys.path ─────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.logger import export_continuous_csv, export_discrete_csv, export_graph_csv
from src.benchmark import BenchmarkRunner
from src.visualization.visualize import (
    plot_convergence, plot_box_scores, plot_time_comparison,
    plot_heatmap_ranking, plot_summary_table, plot_overall_comparison,
    plot_discrete_convergence, plot_discrete_bar,
    plot_pathfinding_grids, plot_pathfinding_metrics, plot_pathfinding_summary_table,
)

import numpy as np
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
    parser.add_argument("--save-json", action="store_true",
                        help="Save continuous & discrete benchmark results to <outdir>/*.json")
    parser.add_argument("--load-json", type=str, default=None, metavar="DIR",
                        help="Load results from <DIR>/*.json and skip re-running")
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
        if args.load_json:
            print(f"\n  Loading continuous results from '{args.load_json}/continuous_results.json' ...")
            cont_results = runner.load_continuous_benchmarks(args.load_json)
        else:
            cont_results = runner.run_continuous_benchmarks(verbose=True)
            if args.save_json:
                path = runner.save_continuous_benchmarks(cont_results, save_dir)
                print(f"\n  Continuous results saved to: {path}")
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
        if args.load_json:
            print(f"\n  Loading discrete results from '{args.load_json}/discrete_results.json' ...")
            disc_results = runner.load_discrete_benchmarks(args.load_json)
        else:
            disc_results = runner.run_discrete_benchmarks(verbose=True)
            if args.save_json:
                path = runner.save_discrete_benchmarks(disc_results, save_dir)
                print(f"\n  Discrete results saved to: {path}")
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

"""
Visualization module for benchmark results (for plotting)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

# ── colour palettes ──────────────────────────────────────────────────────
NATURE_COLORS = {
    "PSO":  "#1f77b4",
    "CS":   "#ff7f0e",
    "FA":   "#2ca02c",
    "DE":   "#d62728",
    "SA":   "#9467bd",
    "TLBO": "#8c564b",
    "Hill Climbing": "#e377c2",
    "GA":   "#7f7f7f",
    "ABC":  "#bcbd22",
    "ACO":  "#17becf",
    # TSP solver variants
    "SA (TSP)":  "#9467bd",
    "GA (TSP)":  "#7f7f7f",
    "ACO (TSP)": "#17becf",
    "CS (TSP)":  "#ff7f0e",
    "ABC (TSP)": "#bcbd22",
    "FA (TSP)":  "#2ca02c",
    "A* (TSP)":  "#e377c2",
    "BFS (TSP)": "#1f77b4",
    "DFS (TSP)": "#d62728",
    # Knapsack solver variants
    "SA (Knapsack)":  "#9467bd",
    "GA (Knapsack)":  "#7f7f7f",
    "ACO (Knapsack)": "#17becf",
    "CS (Knapsack)":  "#ff7f0e",
    "ABC (Knapsack)": "#bcbd22",
    "FA (Knapsack)":  "#2ca02c",
    "A* (Knapsack)":  "#e377c2",
    "BFS (Knapsack)": "#1f77b4",
    "DFS (Knapsack)": "#d62728",
    # Graph Coloring solver variants
    "SA (GC)":  "#9467bd",
    "GA (GC)":  "#7f7f7f",
    "ACO (GC)": "#17becf",
    "CS (GC)":  "#ff7f0e",
    "ABC (GC)": "#bcbd22",
    "FA (GC)":  "#2ca02c",
    "A* (GC)":  "#e377c2",
    "BFS (GC)": "#1f77b4",
    "DFS (GC)": "#d62728",
}

GRAPH_COLORS = {
    "BFS":             "#4C72B0",
    "DFS":             "#DD8452",
    "UCS":             "#55A868",
    "GBFS":            "#C44E52",
    "A* (Manhattan)":  "#8172B2",
    "A* (Euclidean)":  "#937860",
}

def _color(name):
    return NATURE_COLORS.get(name, GRAPH_COLORS.get(name, "#333333"))

def _get(obj, attr, default=None):
    """Retrieve an attribute from either a plain dict or a live object."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _coords_from_distance_matrix(dist_matrix):
    """Build a 2D embedding from a pairwise distance matrix using classical MDS."""
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_matrix must be a square 2D array.")

    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if n == 1:
        return np.array([[0.0, 0.0]], dtype=float)

    # Classical MDS: B = -0.5 * J * (D^2) * J
    J = np.eye(n) - np.ones((n, n), dtype=float) / n
    B = -0.5 * J @ (D ** 2) @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Keep the top two non-negative components.
    pos = np.maximum(eigvals[:2], 0.0)
    coords = eigvecs[:, :2] * np.sqrt(pos)

    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n, dtype=float)])
    elif coords.shape[1] == 0:
        coords = np.zeros((n, 2), dtype=float)

    return coords

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  1. Continuous Optimisation Plots                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def plot_convergence(results, save_dir="output"):
    """
    One convergence plot per benchmark function (mean across trials).
    """
    for fname, algos in results.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        
        has_data = False
        for aname, trials in algos.items():
            all_hist = [tr["history"] for tr in trials if tr["history"]]
            if not all_hist:
                continue
            
            has_data = True
            
            # Chiều dài tối đa của lịch sử trong thuật toán hiện tại
            max_len = max(len(h) for h in all_hist)
            
            # Khởi tạo mảng NaN để gióng hàng dữ liệu tính mean
            arr = np.full((len(all_hist), max_len), np.nan)
            for i, h in enumerate(all_hist):
                arr[i, :len(h)] = h
            
            # Chỉ tính trung bình
            mean = np.nanmean(arr, axis=0)
            
            iters = np.arange(1, max_len + 1)
            
            # Vẽ đường line cho giá trị trung bình
            ax.plot(iters, mean, label=aname, linewidth=2, alpha=0.85)

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title(f"Convergence - {fname}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        
        try:
            ax.set_yscale("log")
        except Exception:
            pass
            
        fig.tight_layout()
        
        plt.show()
        plt.close(fig)


def plot_exploration_exploitation(results, save_dir="output"):
    """
    Plot population diversity across iterations for continuous algorithms.

    Input format
    ------------
    results[function_name][algorithm_name] = [
        {"diversity": [d1, d2, ...], "time": ...}, ...
    ]
    """
    _ensure_dir(save_dir)

    for fname, algos in results.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        for aname, trials in algos.items():
            all_div = [tr.get("diversity", []) for tr in trials if tr.get("diversity")]
            if not all_div:
                continue

            has_data = True
            max_len = max(len(d) for d in all_div)
            arr = np.full((len(all_div), max_len), np.nan)
            for i, d in enumerate(all_div):
                arr[i, :len(d)] = d

            mean_div = np.nanmean(arr, axis=0)
            iters = np.arange(1, max_len + 1)
            ax.plot(iters, mean_div, label=aname, color=_color(aname), linewidth=2, alpha=0.9)

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Population Diversity", fontsize=12)
        ax.set_title(f"Exploration vs Exploitation - {fname}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

def plot_box_scores(results, save_dir="output"):
    """
    Box-plot comparison of final scores per function (one PNG per function).
    """
    _ensure_dir(save_dir)

    for fname, algos in results.items():
        names, data = [], []
        for aname, trials in algos.items():
            scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
            if scores:
                names.append(aname)
                data.append(scores)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
        bp = ax.boxplot(data, labels=names, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markeredgecolor="black",
                                       markerfacecolor="gold", markersize=7))
        colors = [_color(n) for n in names]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title(f"Score Distribution - {fname}", fontsize=14, fontweight="bold")
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_time_comparison(results, save_dir="output"):
    """
    Grouped bar chart: mean runtime per algorithm across all functions.
    """
    _ensure_dir(save_dir)

    func_names = list(results.keys())
    algo_names = list(dict.fromkeys(
        aname for algos in results.values() for aname in algos
    ))

    x = np.arange(len(func_names))
    width = 0.8 / max(len(algo_names), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(func_names) * 2), 6))
    for i, aname in enumerate(algo_names):
        means = []
        for fname in func_names:
            trials = results[fname].get(aname, [])
            times = [tr["time"] for tr in trials]
            means.append(np.mean(times) if times else 0)
        ax.bar(x + i * width, means, width, label=aname, color=_color(aname), alpha=0.8)

    ax.set_xticks(x + width * len(algo_names) / 2)
    ax.set_xticklabels(func_names, fontsize=11)
    ax.set_ylabel("Mean Time (s)", fontsize=12)
    ax.set_title("Runtime Comparison - Continuous Functions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_heatmap_ranking(results, save_dir="output"):
    """
    Heatmap: rank of each algorithm on every function (1 = best).
    """
    _ensure_dir(save_dir)

    func_names = list(results.keys())
    algo_names = list(dict.fromkeys(
        aname for algos in results.values() for aname in algos
    ))

    score_matrix = np.full((len(algo_names), len(func_names)), np.nan)
    for j, fname in enumerate(func_names):
        for i, aname in enumerate(algo_names):
            trials = results[fname].get(aname, [])
            scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
            if scores:
                score_matrix[i, j] = np.mean(scores)

    # rank per column (function) — lower score = rank 1
    rank_matrix = np.full_like(score_matrix, np.nan)
    for j in range(len(func_names)):
        col = score_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.any():
            order = col[valid].argsort().argsort() + 1
            rank_matrix[valid, j] = order

    fig, ax = plt.subplots(figsize=(max(8, len(func_names) * 1.8),
                                     max(5, len(algo_names) * 0.6)))
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1,
                   vmax=np.nanmax(rank_matrix))
    ax.set_xticks(range(len(func_names)))
    ax.set_xticklabels(func_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(algo_names)))
    ax.set_yticklabels(algo_names, fontsize=10)
    for i in range(len(algo_names)):
        for j in range(len(func_names)):
            val = rank_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > len(algo_names) / 2 else "black")
    plt.colorbar(im, ax=ax, label="Rank (1 = best)")
    ax.set_title("Algorithm Ranking Heatmap - Continuous Functions",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_summary_table(results, save_dir="output"):
    """
    Render a summary statistics table as a PNG image.
    Columns: Algorithm | Function | Mean | Std | Best | Worst | Mean Time
    """
    _ensure_dir(save_dir)

    rows = []
    for fname, algos in results.items():
        for aname, trials in algos.items():
            scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
            times  = [tr["time"] for tr in trials]
            if not scores:
                continue
            rows.append([
                aname, fname,
                f"{np.mean(scores):.4e}",
                f"{np.std(scores):.4e}",
                f"{np.min(scores):.4e}",
                f"{np.max(scores):.4e}",
                f"{np.mean(times):.3f}s",
            ])

    if not rows:
        return

    col_labels = ["Algorithm", "Function", "Mean", "Std", "Best", "Worst", "Time"]
    fig, ax = plt.subplots(figsize=(16, max(4, 0.4 * len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    # colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # alternating row colours
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#D9E2F3" if i % 2 == 0 else "white")

    ax.set_title("Continuous Benchmark - Summary Statistics",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_scalability(scalability_results, save_dir="output"):
    """
    Line chart: Mean Best Fitness ± Std vs Problem Size (dimension).

    One figure per benchmark function.  Each algorithm is drawn as a solid
    line (mean) with a translucent ±1 std band.

    Parameters
    ----------
    scalability_results : dict
        Nested dict returned by ``BenchmarkRunner.run_scalability_benchmarks``::

            {
              10: {fname: {aname: [{"score": ..., "time": ..., "history": [...]}]}},
              30: {...},
              50: {...},
            }

    save_dir : str
        Directory used for any future file saving (created automatically).
    """
    _ensure_dir(save_dir)

    dims = sorted(scalability_results.keys())
    if not dims:
        return

    # Collect all function names (preserve insertion order)
    func_names = list(dict.fromkeys(
        fname
        for dim_data in scalability_results.values()
        for fname in dim_data
    ))
    # Collect all algorithm names
    algo_names = list(dict.fromkeys(
        aname
        for dim_data in scalability_results.values()
        for fname_data in dim_data.values()
        for aname in fname_data
    ))

    for fname in func_names:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_data = False

        for aname in algo_names:
            means, stds = [], []
            valid_dims = []

            for dim in dims:
                trials = (
                    scalability_results
                    .get(dim, {})
                    .get(fname, {})
                    .get(aname, [])
                )
                scores = [
                    tr["score"] for tr in trials
                    if tr.get("score") is not None and np.isfinite(tr["score"])
                ]
                if scores:
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                    valid_dims.append(dim)

            if len(valid_dims) < 1:
                continue

            has_data = True
            means  = np.array(means)
            stds   = np.array(stds)
            xs     = np.array(valid_dims)
            color  = _color(aname)

            ax.plot(xs, means, marker="o", linewidth=2, markersize=6,
                    label=aname, color=color, alpha=0.9)
            ax.fill_between(xs, means - stds, means + stds,
                            color=color, alpha=0.15)

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Problem Dimension", fontsize=12)
        ax.set_ylabel("Mean Best Fitness ± Std", fontsize=12)
        ax.set_title(f"Scalability - {fname}", fontsize=14, fontweight="bold")
        ax.set_xticks(dims)
        ax.legend(fontsize=9, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  2. Discrete Problem Plots                                            ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def plot_discrete_convergence(results, save_dir="output"):
    """Convergence curves for each discrete problem."""
    _ensure_dir(save_dir)
    
    for pname, algos in results.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        has_data = False
        for aname, trials in algos.items():
            if aname == "problem":
                continue
            
            all_hist = [tr["history"] for tr in trials if tr["history"]]
            if not all_hist:
                continue
            
            has_data = True
            
            # Lấy chiều dài tối đa của lịch sử thực tế
            max_len = max(len(h) for h in all_hist)
            
            # Khởi tạo mảng NaN để chứa dữ liệu không padding
            arr = np.full((len(all_hist), max_len), np.nan)
            for i, h in enumerate(all_hist):
                arr[i, :len(h)] = h
                
            # Chỉ tính giá trị trung bình (bỏ qua NaN)
            mean = np.nanmean(arr, axis=0)
            iters = np.arange(1, max_len + 1)
            
            # Chỉ vẽ đường trung bình
            ax.plot(iters, mean, label=aname, color=_color(aname), lw=2, alpha=0.85)
            
        # Nếu không có dữ liệu hợp lệ nào, đóng figure và bỏ qua
        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title(f"Convergence - {pname}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_tsp_tour(tsp_results, save_dir="output", title="TSP Tour"):
    """
    Draw the TSP city graph and highlight the best tour per algorithm.

    Accepts the sub-dict returned by BenchmarkRunner for the TSP problem
    (both live objects and JSON-loaded plain dicts):
        disc_results["TSP"]  ->  {"problem": tsp_or_dict, "SA (TSP)": [...], ...}
    """
    if "problem" not in tsp_results and "TSP" in tsp_results:
        tsp_results = tsp_results["TSP"]
    problem = tsp_results["problem"]

    dist_matrix = _get(problem, "dist_matrix")
    if dist_matrix is not None:
        dist_matrix = np.asarray(dist_matrix)

    coords = _get(problem, "coords")
    if coords is not None:
        coord = np.asarray(coords, dtype=float)
    elif dist_matrix is not None:
        coord = _coords_from_distance_matrix(dist_matrix)
    else:
        raise ValueError("tsp visualization requires either coords or dist_matrix.")

    for aname, trials in tsp_results.items():
        if aname == "problem":
            continue

        best_tour = None
        best_dist = float("inf")
        for tr in trials:
            if tr["solution"] is not None and tr["score"] < best_dist:
                best_dist = tr["score"]
                best_tour = tr["solution"]

        if best_tour is None:
            continue

        tour = np.asarray(best_tour, dtype=int)

        # Compute distance inline (works for both live object and dict)
        if dist_matrix is not None:
            dist = sum(
                dist_matrix[tour[k], tour[(k + 1) % len(tour)]]
                for k in range(len(tour))
            )
        elif hasattr(problem, "total_distance"):
            dist = problem.total_distance(tour)
        else:
            dist = best_dist

        fig, ax = plt.subplots(figsize=(8, 8))
        for k in range(len(tour)):
            a = tour[k]; b = tour[(k + 1) % len(tour)]
            ax.annotate(
                "", xy=coord[b], xytext=coord[a],
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
            )
        ax.scatter(coord[:, 0], coord[:, 1], s=120, zorder=5, color="#ff7f0e", edgecolors="black", linewidths=0.8)
        for i, (x, y) in enumerate(coord):
            ax.text(x + 0.8, y + 0.8, str(i), fontsize=9, fontweight="bold", color="#333333")
        ax.set_title("%s – %s  (distance = %.2f)" % (title, aname, dist), fontsize=14, fontweight="bold")
        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_knapsack_items(knapsack_results, save_dir="output", title="Knapsack Items"):
    """
    Bar chart: item index (x-axis) vs weight (y-axis) for each algorithm's best solution.

    Accepts the sub-dict for the Knapsack problem (live object or JSON-loaded dict):
        disc_results["Knapsack"]  ->  {"problem": knapsack_or_dict, ...}
    """
    if "problem" not in knapsack_results and "Knapsack" in knapsack_results:
        knapsack_results = knapsack_results["Knapsack"]
    problem = knapsack_results["problem"]

    n        = int(_get(problem, "n_items", 0))
    weights  = np.asarray(_get(problem, "weights", []))
    values   = np.asarray(_get(problem, "values",  []))
    capacity = float(_get(problem, "capacity", 0))
    x        = np.arange(n)

    for aname, trials in knapsack_results.items():
        if aname == "problem":
            continue

        best_sol = None
        best_val = -1.0
        for tr in trials:
            if tr["solution"] is not None and tr["score"] > best_val:
                best_val = tr["score"]
                best_sol = tr["solution"]

        if best_sol is not None:
            sol     = np.asarray(best_sol, dtype=int)
            colors  = ["#2ca02c" if sol[i] == 1 else "#aec7e8" for i in range(n)]
            total_w = float(sol @ weights)
            subtitle = "  weight: %.1f / %.1f   value: %.1f" % (total_w, capacity, best_val)
        else:
            colors   = ["#aec7e8"] * n
            subtitle = ""

        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), 5))
        bars = ax.bar(x, weights, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                "v=%.0f" % v,
                ha="center", va="bottom", fontsize=8, color="#444444",
            )
        ax.axhline(capacity, color="#d62728", linestyle="--", linewidth=1.5,
                   label="Capacity (%.1f)" % capacity)
        ax.set_xlabel("Item Index", fontsize=12)
        ax.set_ylabel("Weight", fontsize=12)
        ax.set_title("%s – %s%s" % (title, aname, subtitle), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_graph_coloring(gc_results, save_dir="output", title="Graph Coloring"):
    """
    Draw the graph with vertices coloured by each algorithm's best coloring.

    Accepts the sub-dict for Graph Coloring (live object or JSON-loaded dict):
        disc_results["Graph Coloring"]  ->  {"problem": gc_or_dict, "SA (GC)": [...], ...}
    """
    if "problem" not in gc_results and "Graph Coloring" in gc_results:
        gc_results = gc_results["Graph Coloring"]
    problem = gc_results["problem"]

    n_v   = int(_get(problem, "n_vertices", 0))
    edges = _get(problem, "edges", [])

    angles = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    pos    = np.column_stack([np.cos(angles), np.sin(angles)])

    from matplotlib.patches import Patch

    for aname, trials in gc_results.items():
        if aname == "problem":
            continue

        best_coloring = None
        best_score    = float("inf")
        for tr in trials:
            if tr["solution"] is not None and tr["score"] < best_score:
                best_score    = tr["score"]
                best_coloring = tr["solution"]

        if best_coloring is None:
            continue

        coloring = np.asarray(best_coloring, dtype=int)
        n_used   = int(len(set(coloring.tolist())))

        # Compute conflicts inline (works for dict and live object)
        conflicts = 0
        for i, j in edges:
            if coloring[int(i)] == coloring[int(j)]:
                conflicts += 1

        cmap        = plt.cm.get_cmap("tab20", max(n_used, 1))
        vertex_rgba = [cmap(int(coloring[i]) % cmap.N) for i in range(n_v)]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        for i, j in edges:
            ax.plot([pos[int(i), 0], pos[int(j), 0]], [pos[int(i), 1], pos[int(j), 1]],
                    color="#cccccc", linewidth=1.0, zorder=1)
        radius = 0.08
        for i in range(n_v):
            circle = plt.Circle(pos[i], radius, color=vertex_rgba[i],
                                ec="black", linewidth=1.2, zorder=3)
            ax.add_patch(circle)
            ax.text(pos[i, 0], pos[i, 1], str(i),
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if sum(vertex_rgba[i][:3]) < 1.5 else "black", zorder=4)
        legend_patches = [Patch(facecolor=cmap(c), edgecolor="black", label="Color %d" % c)
                          for c in range(n_used)]
        ax.legend(handles=legend_patches, fontsize=9, loc="upper right", title="Colors used")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")
        ax.set_title("%s – %s  (%d colors, %d conflicts)" % (title, aname, n_used, conflicts),
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_discrete_bar(results, save_dir="output"):
    """Bar chart of mean scores for each discrete problem."""
    _ensure_dir(save_dir)
    for pname, algos in results.items():
        names, means, stds = [], [], []
        for aname, trials in algos.items():
            if aname == "problem":
                continue
            scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
            if scores:
                names.append(aname)
                means.append(np.mean(scores))
                stds.append(np.std(scores))
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
        bars = ax.bar(names, means, yerr=stds, capsize=5,
                      color=[_color(n) for n in names], alpha=0.8)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Algorithm Comparison - {pname}", fontsize=14, fontweight="bold")
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_discrete_box(results, save_dir="output"):
    """
    Box-plot comparison of final scores for each discrete problem.

    One figure per problem (TSP / Knapsack / Graph Coloring), one box per
    algorithm.  Works with both live BenchmarkRunner output and JSON-loaded
    plain dicts.

    Parameters
    ----------
    results : dict
        Top-level discrete results dict, e.g.::

            {
              "TSP":           {"problem": ..., "SA (TSP)": [...], ...},
              "Knapsack":      {"problem": ..., "SA (Knapsack)": [...], ...},
              "Graph Coloring": {"problem": ..., "SA (GC)": [...], ...},
            }

        Each trial entry must contain at least ``{"score": float, ...}``.

    save_dir : str
        Directory for saved PNG files (created automatically).
    """
    _ensure_dir(save_dir)

    for pname, algos in results.items():
        names, data = [], []

        for aname, trials in algos.items():
            if aname == "problem":          # skip the stored problem object
                continue
            if not isinstance(trials, list):
                continue
            scores = [
                tr["score"] for tr in trials
                if tr.get("score") is not None and np.isfinite(tr["score"])
            ]
            if scores:
                names.append(aname)
                data.append(scores)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))

        bp = ax.boxplot(
            data,
            labels=names,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markeredgecolor="black",
                markerfacecolor="gold",
                markersize=7,
            ),
        )

        colors = [_color(n) for n in names]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            f"Score Distribution - {pname}", fontsize=14, fontweight="bold"
        )
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  3. Graph-Search Pathfinding Plots                                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def plot_pathfinding_grids(graph_results, save_dir="output"):
    """
    For each grid, draw a 2×3 subplot showing exploration + path per algorithm.
    """
    _ensure_dir(save_dir)
    for label, data in graph_results.items():
        grid  = data["grid"]
        algos = data["algos"]
        names = list(algos.keys())
        n = len(names)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
        axes = np.array(axes).flatten()
        for ax, name in zip(axes, names):
            r = algos[name]
            grid.plot(path=r["path"], visited=r["visited"], title=name, ax=ax)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"Pathfinding - {label}", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close(fig)


def plot_pathfinding_metrics(graph_results, save_dir="output"):
    """
    Bar charts comparing path_length, nodes_expanded, elapsed_ms across algorithms
    for all grid sizes on the same figure.
    """
    _ensure_dir(save_dir)

    labels_all = list(graph_results.keys())
    algo_names = list(next(iter(graph_results.values()))["algos"].keys())
    metrics = [
        ("path_length",    "Path Length (cells)"),
        ("nodes_expanded", "Nodes Expanded"),
        ("elapsed_ms",     "Runtime (ms)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(labels_all))
    width = 0.8 / max(len(algo_names), 1)

    for ax, (key, ylabel) in zip(axes, metrics):
        for i, aname in enumerate(algo_names):
            vals = [graph_results[lab]["algos"][aname][key] for lab in labels_all]
            ax.bar(x + i * width, vals, width, label=aname,
                   color=_color(aname), alpha=0.8)
        ax.set_xticks(x + width * len(algo_names) / 2)
        ax.set_xticklabels(labels_all, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(key.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Graph-Search Algorithm Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close(fig)


def plot_pathfinding_summary_table(graph_results, save_dir="output"):
    """Table image: algo × grid × {path_length, nodes_expanded, time}."""
    _ensure_dir(save_dir)
    rows = []
    for label, data in graph_results.items():
        for aname, m in data["algos"].items():
            rows.append([
                aname, label,
                str(m["path_length"]),
                str(m["nodes_expanded"]),
                f"{m['elapsed_ms']:.3f}",
            ])
    col_labels = ["Algorithm", "Grid", "Path Len", "Expanded", "Time (ms)"]
    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#D9E2F3" if i % 2 == 0 else "white")
    ax.set_title("Pathfinding - Summary Table", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  4.  Overall comparison chart (nature-inspired vs traditional)        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def plot_overall_comparison(continuous_results, save_dir="output"):
    """
    Radar chart: normalised (0-1) mean score per function for each algorithm.
    Lower = better (closer to centre for minimisation).
    """
    _ensure_dir(save_dir)

    func_names = list(continuous_results.keys())
    algo_names = list(dict.fromkeys(
        aname for algos in continuous_results.values() for aname in algos
    ))

    # build raw score matrix (algo × func)
    raw = np.full((len(algo_names), len(func_names)), np.nan)
    for j, fname in enumerate(func_names):
        for i, aname in enumerate(algo_names):
            trials = continuous_results[fname].get(aname, [])
            scores = [tr["score"] for tr in trials if np.isfinite(tr["score"])]
            if scores:
                raw[i, j] = np.mean(scores)

    # normalise per function (min-max → 0 = best)
    norm = np.full_like(raw, np.nan)
    for j in range(len(func_names)):
        col = raw[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            norm[valid, j] = 0.5
        else:
            lo, hi = col[valid].min(), col[valid].max()
            if hi - lo < 1e-15:
                norm[valid, j] = 0.5
            else:
                norm[valid, j] = (col[valid] - lo) / (hi - lo)

    n_vars = len(func_names)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, aname in enumerate(algo_names):
        vals = norm[i].tolist()
        if any(np.isnan(v) for v in vals):
            continue
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=aname, color=_color(aname), alpha=0.8)
        ax.fill(angles, vals, color=_color(aname), alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), func_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Normalised Score (lower=better) - Radar Chart",
                 fontsize=14, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

"""
Visualization module for benchmark results
==========================================
Generates publication-quality PNG charts that compare Nature-Inspired
algorithms (PSO, CS, FA, DE, SA, TLBO, GA, ABC, ACO) with Traditional
search algorithms (BFS, DFS, UCS, GBFS, A*, Hill Climbing).

All public ``plot_*`` helpers accept an optional ``save_dir`` and write
PNG files at 300 dpi.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                         # non-interactive backend → PNG only
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
    "SA (TSP)": "#9467bd",
    "GA (TSP)": "#7f7f7f",
    "GA (Knapsack)": "#7f7f7f",
    "ABC (Knapsack)": "#bcbd22",
    "SA (GC)": "#9467bd",
    "GA (GC)": "#7f7f7f",
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

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  1. Continuous Optimisation Plots                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def plot_convergence(results, save_dir="output"):
    """
    One convergence plot per benchmark function (mean ± std across trials).

    Parameters
    ----------
    results : dict   results[func_name][algo_name] = list of trial dicts
    save_dir : str   Directory to write PNGs into.
    """
    _ensure_dir(save_dir)

    for fname, algos in results.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        for aname, trials in algos.items():
            all_hist = [tr["history"] for tr in trials if tr["history"]]
            if not all_hist:
                continue
            max_len = max(len(h) for h in all_hist)
            padded = np.full((len(all_hist), max_len), np.nan)
            for i, h in enumerate(all_hist):
                padded[i, :len(h)] = h
                padded[i, len(h):] = h[-1]       # fill with final value
            mean = np.nanmean(padded, axis=0)
            std  = np.nanstd(padded, axis=0)
            iters = np.arange(1, max_len + 1)
            ax.plot(iters, mean, label=aname, color=_color(aname), linewidth=2, alpha=0.85)
            lower = np.clip(mean - std, 1e-30, None)
            ax.fill_between(iters, lower, mean + std, color=_color(aname), alpha=0.15)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title(f"Convergence - {fname}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        # safe log scale: only if all plotted values are positive
        try:
            ax.set_yscale("log")
        except Exception:
            pass
        fig.tight_layout()
        path = os.path.join(save_dir, f"convergence_{fname.lower()}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {path}")


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
        path = os.path.join(save_dir, f"boxplot_{fname.lower()}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK]  {path}")


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
    path = os.path.join(save_dir, "runtime_continuous.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {path}")


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
    path = os.path.join(save_dir, "ranking_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {path}")


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
    path = os.path.join(save_dir, "summary_table_continuous.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {path}")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  2. Discrete Problem Plots                                            ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def plot_discrete_convergence(results, save_dir="output"):
    """Convergence curves for each discrete problem."""
    _ensure_dir(save_dir)
    for pname, algos in results.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        for aname, trials in algos.items():
            all_hist = [tr["history"] for tr in trials if tr["history"]]
            if not all_hist:
                continue
            max_len = max(len(h) for h in all_hist)
            padded = np.full((len(all_hist), max_len), np.nan)
            for i, h in enumerate(all_hist):
                padded[i, :len(h)] = h
                padded[i, len(h):] = h[-1]
            mean = np.nanmean(padded, axis=0)
            std  = np.nanstd(padded, axis=0)
            iters = np.arange(1, max_len + 1)
            ax.plot(iters, mean, label=aname, color=_color(aname), lw=2, alpha=0.85)
            ax.fill_between(iters, mean - std, mean + std, color=_color(aname), alpha=0.15)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Score", fontsize=12)
        ax.set_title(f"Convergence - {pname}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, f"convergence_{pname.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK]  {path}")


def plot_discrete_bar(results, save_dir="output"):
    """Bar chart of mean scores for each discrete problem."""
    _ensure_dir(save_dir)
    for pname, algos in results.items():
        names, means, stds = [], [], []
        for aname, trials in algos.items():
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
        path = os.path.join(save_dir, f"bar_{pname.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK]  {path}")


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
        path = os.path.join(save_dir, f"pathfinding_grid_{label.lower().replace(' ', '_').replace('×', 'x')}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK]  {path}")


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
    path = os.path.join(save_dir, "pathfinding_metrics.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {path}")


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
    path = os.path.join(save_dir, "pathfinding_summary_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {path}")


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
    path = os.path.join(save_dir, "radar_overall.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {path}")

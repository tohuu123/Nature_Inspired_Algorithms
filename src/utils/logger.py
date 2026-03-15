import os
import csv
import numpy as np
import json

def _to_list(obj):
    """Recursively convert numpy arrays / scalars to plain Python types."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, list):
        return [_to_list(v) for v in obj]
    return obj


def _problem_to_dict(pname, problem):
    """
    Serialise a discrete problem object to a JSON-safe plain dict.
    Supported types: TSP, KnapsackProblem, GraphColoring.
    """
    if pname == "TSP":
        d = {
            "type":        "TSP",
            "n_cities":    int(problem.n_cities),
            "dist_matrix": _to_list(problem.dist_matrix),
        }
        if problem.coords is not None:
            d["coords"] = _to_list(problem.coords)
        return d
    elif pname == "Knapsack":
        return {
            "type":     "Knapsack",
            "n_items":  int(problem.n_items),
            "weights":  _to_list(problem.weights),
            "values":   _to_list(problem.values),
            "capacity": float(problem.capacity),
        }
    elif pname == "Graph Coloring":
        return {
            "type":       "Graph Coloring",
            "n_vertices": int(problem.n_vertices),
            "edges":      [list(e) for e in problem.edges],
        }
    # Fallback: try __dict__
    try:
        return {k: _to_list(v) for k, v in vars(problem).items()}
    except Exception:
        return {"type": str(type(problem).__name__)}


def _serialise_discrete(results):
    """
    Convert the dict returned by run_discrete_benchmarks() into a fully
    JSON-serialisable structure.

    Resulting JSON shape
    --------------------
    {
      "TSP": {
        "problem": { "type": "TSP", "n_cities": 15, "dist_matrix": [...], "coords": [...] },
        "SA (TSP)": [ {"score": ..., "time": ..., "history": [...], "solution": [...]}, ... ],
        ...
      },
      "Knapsack":      { ... },
      "Graph Coloring": { ... }
    }
    """
    out = {}
    for pname, algos in results.items():
        out[pname] = {}
        for key, value in algos.items():
            if key == "problem":
                out[pname]["problem"] = _problem_to_dict(pname, value)
            else:
                out[pname][key] = [
                    {
                        "score":    _to_list(tr.get("score")),
                        "time":     _to_list(tr.get("time")),
                        "history":  _to_list(tr.get("history", [])),
                        "solution": _to_list(tr.get("solution")),
                    }
                    for tr in value
                ]
    return out

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

# ──────────────────────────────────────────────────────────────────────
#  Save / Load continuous results as JSON
# ──────────────────────────────────────────────────────────────────────
def save_continuous_benchmarks(results, save_dir="output"):
    """
    Serialise the dict returned by run_continuous_benchmarks() to JSON.

    Parameters
    ----------
    results  : dict   Returned by run_continuous_benchmarks().
    save_dir : str    Target directory; file is <save_dir>/continuous_results.json.

    Returns
    -------
    str : Absolute path of the written JSON file.
    """
    from src.utils.logger import _to_list
    os.makedirs(save_dir, exist_ok=True)
    payload = {}
    for fname, algos in results.items():
        payload[fname] = {}
        for aname, trials in algos.items():
            payload[fname][aname] = [
                {
                    "score":   _to_list(tr.get("score")),
                    "time":    _to_list(tr.get("time")),
                    "history": _to_list(tr.get("history", [])),
                }
                for tr in trials
            ]
    path = os.path.join(save_dir, "continuous_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(path)

def load_continuous_benchmarks(save_dir="output"):
    """
    Load continuous benchmark results previously saved by save_continuous_benchmarks().

    Parameters
    ----------
    save_dir : str   Directory containing ``continuous_results.json``.

    Returns
    -------
    dict : Same structure as run_continuous_benchmarks().
    """
    path = os.path.join(save_dir, "continuous_results.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No continuous_results.json found in '{save_dir}'.")
    with open(path) as f:
        return json.load(f)


def save_exploration_exploitation_benchmarks(results, save_dir="output"):
    """
    Serialise exploration-exploitation benchmark results to JSON.

    Parameters
    ----------
    results : dict
        Returned by ``bench_exploration_exploitation()``.
    save_dir : str
        Target directory; file is ``<save_dir>/exploration_exploitation_results.json``.

    Returns
    -------
    str
        Absolute path of the written JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = {}
    for fname, algos in results.items():
        payload[fname] = {}
        for aname, trials in algos.items():
            payload[fname][aname] = [
                {
                    "diversity": _to_list(tr.get("diversity", [])),
                    "time": _to_list(tr.get("time", 0)),
                }
                for tr in trials
            ]
    path = os.path.join(save_dir, "exploration_exploitation_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(path)


def load_exploration_exploitation_benchmarks(save_dir="output"):
    """
    Load exploration-exploitation benchmark results from JSON.

    Parameters
    ----------
    save_dir : str
        Directory containing ``exploration_exploitation_results.json``.

    Returns
    -------
    dict
        Same structure as ``bench_exploration_exploitation()`` output.
    """
    path = os.path.join(save_dir, "exploration_exploitation_results.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No exploration_exploitation_results.json found in '{save_dir}'.")
    with open(path) as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────────────
#  Save / Load discrete results as JSON
# ──────────────────────────────────────────────────────────────────────
def save_discrete_benchmarks(results, save_dir="output"):
    """
    Serialise the dict returned by run_discrete_benchmarks() to JSON.

    Parameters
    ----------
    results  : dict   Returned by run_discrete_benchmarks().
    save_dir : str    Target directory; file is <save_dir>/discrete_results.json.

    Returns
    -------
    str : Absolute path of the written JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = _serialise_discrete(results)
    path = os.path.join(save_dir, "discrete_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(path)

def load_discrete_benchmarks(save_dir="output"):
    """
    Load discrete benchmark results previously saved by save_discrete_benchmarks().

    Parameters
    ----------
    save_dir : str   Directory containing ``discrete_results.json``.
    
    Returns
    -------
    dict : Same structure as run_discrete_benchmarks().
    """
    path = os.path.join(save_dir, "discrete_results.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No discrete_results.json found in '{save_dir}'.")
    with open(path) as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────────────
#  Save / Load scalability results as JSON
# ──────────────────────────────────────────────────────────────────────
def save_scalability_benchmarks(results, save_dir="output"):
    """
    Serialise the dict returned by run_scalability_benchmarks() to JSON.

    Parameters
    ----------
    results  : dict
        Returned by ``BenchmarkRunner.run_scalability_benchmarks()``.
        Shape: ``{dim: {func_name: {algo_name: [trial_dict, ...]}}}``.
    save_dir : str
        Target directory; the file is written as
        ``<save_dir>/scalability_results.json``.

    Returns
    -------
    str : Absolute path of the written JSON file.

    Notes
    -----
    JSON only supports string keys, so integer dimension keys are stored as
    strings (e.g. ``"10"``, ``"30"``).  ``load_scalability_benchmarks``
    converts them back to ``int`` automatically.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = {}
    for dim, dim_data in results.items():
        dim_key = str(dim)          # JSON keys must be strings
        payload[dim_key] = {}
        for fname, algos in dim_data.items():
            payload[dim_key][fname] = {}
            for aname, trials in algos.items():
                payload[dim_key][fname][aname] = [
                    {
                        "score":   _to_list(tr.get("score")),
                        "time":    _to_list(tr.get("time")),
                        "history": _to_list(tr.get("history", [])),
                    }
                    for tr in trials
                ]
    path = os.path.join(save_dir, "scalability_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK]  {os.path.abspath(path)}")
    return os.path.abspath(path)


def load_scalability_benchmarks(save_dir="output"):
    """
    Load scalability benchmark results previously saved by
    ``save_scalability_benchmarks()``.

    Parameters
    ----------
    save_dir : str
        Directory containing ``scalability_results.json``.

    Returns
    -------
    dict
        Same structure as ``run_scalability_benchmarks()``:
        ``{dim (int): {func_name: {algo_name: [trial_dict, ...]}}}``.
        Ready to pass directly to ``plot_scalability()``.
    """
    path = os.path.join(save_dir, "scalability_results.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No scalability_results.json found in '{save_dir}'.")
    with open(path) as f:
        raw = json.load(f)
    # Convert string dimension keys back to int
    return {int(dim): dim_data for dim, dim_data in raw.items()}


def save_parameter_sensitivity_benchmarks(results, save_dir="output"):
    """
    Serialise parameter sensitivity benchmark results to JSON.

    Parameters
    ----------
    results : dict
        Returned by ``BenchmarkRunner.run_parameters_sensitivity()``.
    save_dir : str
        Target directory; file is ``<save_dir>/parameter_sensitivity_results.json``.

    Returns
    -------
    str
        Absolute path of the written JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = _to_list(results)
    path = os.path.join(save_dir, "parameter_sensitivity_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(path)


def load_parameter_sensitivity_benchmarks(save_dir="output"):
    """
    Load parameter sensitivity benchmark results from JSON.

    Parameters
    ----------
    save_dir : str
        Directory containing ``parameter_sensitivity_results.json``.

    Returns
    -------
    dict
        Same structure as ``run_parameters_sensitivity()`` output.
    """
    path = os.path.join(save_dir, "parameter_sensitivity_results.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No parameter_sensitivity_results.json found in '{save_dir}'.")
    with open(path) as f:
        return json.load(f)


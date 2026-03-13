import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import time


class Grid:
    """
    2-D grid environment for pathfinding algorithms.

    Parameters
    ----------
    rows    : int   Number of rows in the grid.
    cols    : int   Number of columns in the grid.
    start   : tuple (row, col) coordinates of the start cell.
    goal    : tuple (row, col) coordinates of the goal cell.
    obstacle_ratio : float  Fraction of cells randomly blocked [0, 1).
    seed    : int or None   Random seed for reproducible obstacle placement.
    """

    def __init__(self, rows=20, cols=20, start=(0, 0), goal=None, obstacle_ratio=0.25, seed=None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal if goal is not None else (rows - 1, cols - 1)
        self.obstacle_ratio = obstacle_ratio
        self.seed = seed
        self.grid = self._generate()

    def _generate(self):
        """
        Create binary grid: 0 = free, 1 = obstacle.
        Retries with incremented seeds until start and goal are connected,
        guaranteeing a valid problem instance.
        """
        seed = self.seed
        attempt = 0
        while True:
            rng = np.random.default_rng(seed)
            grid = np.zeros((self.rows, self.cols), dtype=int)
            n_obstacles = int(self.rows * self.cols * self.obstacle_ratio)
            flat_indices = rng.choice(self.rows * self.cols, size=n_obstacles, replace=False)
            for idx in flat_indices:
                r, c = divmod(int(idx), self.cols)
                if (r, c) != self.start and (r, c) != self.goal:
                    grid[r, c] = 1
            if self._is_reachable(grid):
                if attempt > 0:
                    print("Grid: seed %s was unsolvable; using seed %s after %d attempt(s)." % (self.seed, seed, attempt))
                return grid
            seed = (seed + 1) if seed is not None else attempt
            attempt += 1

    def _is_reachable(self, grid):
        """BFS connectivity check: returns True if goal is reachable from start."""
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def is_valid(self, r, c):
        """Return True if (r, c) is within bounds and not an obstacle."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0

    def neighbors(self, r, c):
        """Yield valid 4-connected neighbors of (r, c)."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                yield nr, nc

    def reconstruct_path(self, came_from, node):
        """Trace back the path from goal to start using the came_from map."""
        path = []
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path

    def plot(self, path=None, visited=None, title="Grid", ax=None):
        """
        Visualise the grid, visited cells, and the found path.

        Parameters
        ----------
        path    : list of (r, c) tuples forming the solution path.
        visited : set of (r, c) tuples explored during search.
        title   : str  Title of the plot.
        ax      : matplotlib Axes object; creates a new figure if None.
        """
        display = np.ones((self.rows, self.cols, 3))  # white background
        display[self.grid == 1] = [0.1, 0.1, 0.1]    # obstacles → dark

        if visited:
            for r, c in visited:
                if (r, c) != self.start and (r, c) != self.goal:
                    display[r, c] = [0.6, 0.85, 1.0]  # light-blue explored

        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    display[r, c] = [1.0, 0.3, 0.3]   # red path

        sr, sc = self.start
        gr, gc = self.goal
        display[sr, sc] = [0.0, 0.8, 0.0]  # green start
        display[gr, gc] = [0.8, 0.0, 0.8]  # purple goal

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(display, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if standalone:
            plt.tight_layout()
            plt.show()


class BFS:
    """
    Breadth-First Search on a Grid.

    Guarantees the shortest path (fewest edges) on unweighted grids.

    Parameters
    ----------
    grid : Grid  The grid environment to search.
    """

    def __init__(self, grid):
        self.grid = grid
        self.path = []
        self.visited = set()
        self.nodes_expanded = 0
        self.elapsed = 0.0

    def run(self):
        """
        Execute BFS from grid.start to grid.goal.

        Returns
        -------
        path    : list of (r, c) tuples – empty if no path found.
        visited : set of explored cells.
        """
        t0 = time.perf_counter()
        start, goal = self.grid.start, self.grid.goal
        queue = deque([start])
        came_from = {start: None}
        self.visited = {start}
        self.nodes_expanded = 0

        while queue:
            node = queue.popleft()
            self.nodes_expanded += 1
            if node == goal:
                self.path = self.grid.reconstruct_path(came_from, goal)
                self.elapsed = time.perf_counter() - t0
                return self.path, self.visited

            for nb in self.grid.neighbors(*node):
                if nb not in came_from:
                    came_from[nb] = node
                    self.visited.add(nb)
                    queue.append(nb)

        self.path = []
        self.elapsed = time.perf_counter() - t0
        return self.path, self.visited

    def plot(self, title="BFS"):
        """Visualise the BFS result on the grid."""
        self.grid.plot(path=self.path, visited=self.visited, title=title)


class DFS:
    """
    Depth-First Search on a Grid.

    Not guaranteed to find the shortest path, but uses less memory on sparse
    grids. Explores the grid iteratively (stack-based) to avoid recursion limits.

    Parameters
    ----------
    grid : Grid  The grid environment to search.
    """

    def __init__(self, grid):
        self.grid = grid
        self.path = []
        self.visited = set()
        self.nodes_expanded = 0
        self.elapsed = 0.0

    def run(self):
        """
        Execute iterative DFS from grid.start to grid.goal.

        Returns
        -------
        path    : list of (r, c) tuples – empty if no path found.
        visited : set of explored cells.
        """
        t0 = time.perf_counter()
        start, goal = self.grid.start, self.grid.goal
        stack = [start]
        came_from = {start: None}
        self.visited = set()
        self.nodes_expanded = 0

        while stack:
            node = stack.pop()
            if node in self.visited:
                continue
            self.visited.add(node)
            self.nodes_expanded += 1

            if node == goal:
                self.path = self.grid.reconstruct_path(came_from, goal)
                self.elapsed = time.perf_counter() - t0
                return self.path, self.visited

            for nb in self.grid.neighbors(*node):
                if nb not in self.visited:
                    came_from[nb] = node
                    stack.append(nb)

        self.path = []
        self.elapsed = time.perf_counter() - t0
        return self.path, self.visited

    def plot(self, title="DFS"):
        """Visualise the DFS result on the grid."""
        self.grid.plot(path=self.path, visited=self.visited, title=title)


class Heuristic:
    """
    Collection of heuristic functions for A* Search.

    All functions accept two (row, col) tuples and return a float cost estimate.
    """

    @staticmethod
    def manhattan(a, b):
        """Sum of absolute differences in coordinates (4-connected grid)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def euclidean(a, b):
        """Straight-line distance between two cells."""
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


class AStar:
    """
    A* Search on a Grid.

    Finds the optimal (shortest) path using a heuristic to guide expansion.
    Both Manhattan and Euclidean heuristics are available via the Heuristic class.

    Parameters
    ----------
    grid      : Grid      The grid environment to search.
    heuristic : callable  Function h(a, b) -> float estimating cost from a to goal.
                          Defaults to Heuristic.manhattan.
    """

    def __init__(self, grid, heuristic=None):
        self.grid = grid
        self.heuristic = heuristic if heuristic is not None else Heuristic.manhattan
        self.path = []
        self.visited = set()
        self.nodes_expanded = 0
        self.elapsed = 0.0

    def run(self):
        """
        Execute A* from grid.start to grid.goal.

        Returns
        -------
        path    : list of (r, c) tuples – empty if no path found.
        visited : set of explored cells.
        """
        t0 = time.perf_counter()
        start, goal = self.grid.start, self.grid.goal
        # priority queue entries: (f_score, g_score, node)
        open_heap = [(self.heuristic(start, goal), 0, start)]
        came_from = {start: None}
        g_score = {start: 0}
        self.visited = set()
        self.nodes_expanded = 0

        while open_heap:
            f, g, node = heapq.heappop(open_heap)
            if node in self.visited:
                continue
            self.visited.add(node)
            self.nodes_expanded += 1

            if node == goal:
                self.path = self.grid.reconstruct_path(came_from, goal)
                self.elapsed = time.perf_counter() - t0
                return self.path, self.visited

            for nb in self.grid.neighbors(*node):
                tentative_g = g + 1  # uniform step cost for 4-connected grid
                if tentative_g < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative_g
                    came_from[nb] = node
                    f_nb = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f_nb, tentative_g, nb))

        self.path = []
        self.elapsed = time.perf_counter() - t0
        return self.path, self.visited

    def plot(self, title="A* Search"):
        """Visualise the A* result on the grid."""
        self.grid.plot(path=self.path, visited=self.visited, title=title)


class UCS:
    """
    Uniform-Cost Search on a Grid.

    Expands the node with the lowest cumulative path cost g(n).
    Equivalent to Dijkstra's algorithm when all edges have non-negative cost.
    On a unit-cost grid this behaves identically to BFS but is included
    to illustrate the priority-queue formulation.

    Parameters
    ----------
    grid : Grid  The grid environment to search.
    """

    def __init__(self, grid):
        self.grid = grid
        self.path = []
        self.visited = set()
        self.nodes_expanded = 0
        self.elapsed = 0.0

    def run(self):
        """
        Execute UCS from grid.start to grid.goal.

        Returns
        -------
        path    : list of (r, c) tuples – empty if no path found.
        visited : set of explored cells.
        """
        t0 = time.perf_counter()
        start, goal = self.grid.start, self.grid.goal
        open_heap = [(0, start)]          # (g_score, node)
        came_from = {start: None}
        g_score = {start: 0}
        self.visited = set()
        self.nodes_expanded = 0

        while open_heap:
            g, node = heapq.heappop(open_heap)
            if node in self.visited:
                continue
            self.visited.add(node)
            self.nodes_expanded += 1

            if node == goal:
                self.path = self.grid.reconstruct_path(came_from, goal)
                self.elapsed = time.perf_counter() - t0
                return self.path, self.visited

            for nb in self.grid.neighbors(*node):
                tentative_g = g + 1      # uniform step cost
                if tentative_g < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative_g
                    came_from[nb] = node
                    heapq.heappush(open_heap, (tentative_g, nb))

        self.path = []
        self.elapsed = time.perf_counter() - t0
        return self.path, self.visited

    def plot(self, title="UCS"):
        """Visualise the UCS result on the grid."""
        self.grid.plot(path=self.path, visited=self.visited, title=title)


class GBFS:
    """
    Greedy Best-First Search on a Grid.

    Expands the node with the smallest heuristic value h(n).
    Not optimal — may find a suboptimal path — but typically fast.

    Parameters
    ----------
    grid      : Grid      The grid environment to search.
    heuristic : callable  h(a, b) -> float.  Defaults to Manhattan distance.
    """

    def __init__(self, grid, heuristic=None):
        self.grid = grid
        self.heuristic = heuristic if heuristic is not None else Heuristic.manhattan
        self.path = []
        self.visited = set()
        self.nodes_expanded = 0
        self.elapsed = 0.0

    def run(self):
        """
        Execute GBFS from grid.start to grid.goal.

        Returns
        -------
        path    : list of (r, c) tuples – empty if no path found.
        visited : set of explored cells.
        """
        t0 = time.perf_counter()
        start, goal = self.grid.start, self.grid.goal
        open_heap = [(self.heuristic(start, goal), start)]
        came_from = {start: None}
        self.visited = set()
        self.nodes_expanded = 0

        while open_heap:
            _, node = heapq.heappop(open_heap)
            if node in self.visited:
                continue
            self.visited.add(node)
            self.nodes_expanded += 1

            if node == goal:
                self.path = self.grid.reconstruct_path(came_from, goal)
                self.elapsed = time.perf_counter() - t0
                return self.path, self.visited

            for nb in self.grid.neighbors(*node):
                if nb not in self.visited and nb not in came_from:
                    came_from[nb] = node
                    heapq.heappush(open_heap, (self.heuristic(nb, goal), nb))

        self.path = []
        self.elapsed = time.perf_counter() - t0
        return self.path, self.visited

    def plot(self, title="Greedy Best-First Search"):
        """Visualise the GBFS result on the grid."""
        self.grid.plot(path=self.path, visited=self.visited, title=title)


class AlgorithmComparison:
    """
    Run and compare BFS, DFS, UCS, GBFS, A* (Manhattan), and A* (Euclidean) on the same Grid.

    Parameters
    ----------
    grid : Grid  Shared grid instance used by all algorithms.
    """

    def __init__(self, grid):
        self.grid = grid
        self.results = {}

    def run_all(self):
        """
        Execute all six algorithms and collect metrics.

        Metrics collected per algorithm
        --------------------------------
        path_length     : int   Number of cells in the solution path (0 = no path).
        nodes_expanded  : int   Total nodes popped from the frontier.
        elapsed_ms      : float Wall-clock time in milliseconds.
        path            : list  Solution path (list of (r, c) tuples).
        visited         : set   Set of explored cells.
        """
        algorithms = [
            ("BFS",              BFS(self.grid)),
            ("DFS",              DFS(self.grid)),
            ("UCS",              UCS(self.grid)),
            ("GBFS",             GBFS(self.grid, Heuristic.manhattan)),
            ("A* (Manhattan)",   AStar(self.grid, Heuristic.manhattan)),
            ("A* (Euclidean)",   AStar(self.grid, Heuristic.euclidean)),
        ]

        for name, algo in algorithms:
            path, visited = algo.run()
            self.results[name] = {
                "path_length":    len(path),
                "nodes_expanded": algo.nodes_expanded,
                "elapsed_ms":     algo.elapsed * 1000,
                "path":           path,
                "visited":        visited,
            }

        return self.results

    def print_summary(self):
        """Print a formatted comparison table to stdout."""
        if not self.results:
            self.run_all()

        header = "%-20s  %12s  %16s  %12s" % (
            "Algorithm", "Path Length", "Nodes Expanded", "Time (ms)"
        )
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        for name, r in self.results.items():
            print("%-20s  %12d  %16d  %12.4f" % (
                name, r["path_length"], r["nodes_expanded"], r["elapsed_ms"]
            ))
        print("=" * len(header))

    def plot_all(self):
        """Display a 2×3 subplot grid showing each algorithm's exploration and path."""
        if not self.results:
            self.run_all()

        names = list(self.results.keys())
        n = len(names)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
        axes = axes.flatten()

        for ax, name in zip(axes, names):
            r = self.results[name]
            self.grid.plot(path=r["path"], visited=r["visited"], title=name, ax=ax)

        # Hide unused axes
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Pathfinding Algorithm Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        """Bar-chart comparison of path length, nodes expanded, and runtime."""
        if not self.results:
            self.run_all()

        names = list(self.results.keys())
        path_lens = [self.results[n]["path_length"] for n in names]
        nodes_exp = [self.results[n]["nodes_expanded"] for n in names]
        times_ms  = [self.results[n]["elapsed_ms"] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for ax, values, ylabel, title in zip(
            axes,
            [path_lens, nodes_exp, times_ms],
            ["Cells in Path", "Nodes Expanded", "Time (ms)"],
            ["Path Length", "Nodes Expanded", "Runtime (ms)"],
        ):
            bars = ax.bar(names, values, color=colors)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    str(round(val, 3)),
                    ha="center", va="bottom", fontsize=9,
                )

        fig.suptitle("Algorithm Metrics Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ROWS   = 25
    COLS   = 25
    START  = (0, 0)
    GOAL   = (ROWS - 1, COLS - 1)
    RATIO  = 0.30
    SEED   = 40

    grid = Grid(rows=ROWS, cols=COLS, start=START, goal=GOAL, obstacle_ratio=RATIO, seed=SEED)

    cmp = AlgorithmComparison(grid)
    cmp.run_all()
    cmp.print_summary()
    cmp.plot_all()
    cmp.plot_metrics()

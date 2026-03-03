import numpy as np
from matplotlib import pyplot
import os
import sys


class ACO:
    """
    Ant Colony Optimization for the Travelling Salesman Problem (TSP).

    Parameters
    ----------
    dist_matrix : ndarray, shape (n, n)
        Symmetric pairwise distance matrix.
    n_ants      : int
        Number of ants per iteration.
    n_iter      : int
        Total number of iterations.
    alpha       : float
        Pheromone importance exponent (τ^alpha).
    beta        : float
        Heuristic importance exponent (η^beta = (1/d)^beta).
    rho         : float
        Pheromone evaporation rate in [0, 1].
    Q           : float
        Pheromone deposit constant.
    tau_init    : float
        Initial pheromone level on all edges.
    """

    def __init__(self, dist_matrix, n_ants=20, n_iter=200, alpha=1.0, beta=3.0, rho=0.1, Q=100.0, tau_init=1.0):
        self.dist_matrix = np.asarray(dist_matrix, dtype=float)
        self.n_cities    = self.dist_matrix.shape[0]
        self.n_ants      = n_ants
        self.n_iter      = n_iter
        self.alpha       = alpha
        self.beta        = beta
        self.rho         = rho
        self.Q           = Q
        self.tau_init    = tau_init

        self.best_tour  = None
        self.best_dist  = np.inf
        self.history    = []

    def _heuristic(self):
        """Inverse-distance heuristic matrix η(i,j) = 1 / d(i,j)."""
        with np.errstate(divide='ignore'):
            eta = np.where(self.dist_matrix == 0, 0.0, 1.0 / self.dist_matrix)
        return eta

    def _tour_length(self, tour):
        """Compute the total closed-tour distance."""
        tour = np.asarray(tour, dtype=int)
        return float(
            sum(self.dist_matrix[tour[i], tour[(i + 1) % self.n_cities]]
                for i in range(self.n_cities))
        )

    def _build_tour(self, tau, eta):
        """
        Construct a single ant's tour using the ACO probabilistic rule.

        Returns
        -------
        tour : list[int]  Ordered city indices.
        """
        start   = np.random.randint(self.n_cities)
        tour    = [start]
        visited = {start}

        for _ in range(self.n_cities - 1):
            current   = tour[-1]
            unvisited = [c for c in range(self.n_cities) if c not in visited]

            # τ^alpha * η^beta for each unvisited city
            scores = np.array(
                [(tau[current, c] ** self.alpha) * (eta[current, c] ** self.beta)
                 for c in unvisited]
            )
            total = scores.sum()

            probs     = np.ones(len(unvisited)) / len(unvisited) if total == 0 else scores / total
            next_city = unvisited[np.random.choice(len(unvisited), p=probs)]
            tour.append(next_city)
            visited.add(next_city)

        return tour

    def _update_pheromone(self, tau, all_tours, all_lengths):
        """
        Apply evaporation and deposit pheromone based on tour quality.

        Parameters
        ----------
        tau         : ndarray  Current pheromone matrix (modified in place).
        all_tours   : list of lists  Tours built this iteration.
        all_lengths : list of float  Corresponding tour lengths.
        """
        tau *= (1.0 - self.rho)

        for tour, length in zip(all_tours, all_lengths):
            deposit = self.Q / length
            for i in range(self.n_cities):
                a = tour[i]
                b = tour[(i + 1) % self.n_cities]
                tau[a, b] += deposit
                tau[b, a] += deposit

    def run(self, verbose=True):
        """
        Execute the ACO algorithm.

        Parameters
        ----------
        verbose : bool
            If True, print a line whenever a new global best is found.

        Returns
        -------
        best_tour : list[int]  Best sequence of city indices (closed loop).
        best_dist : float      Total tour distance.
        history   : list[float] Best distance recorded after each iteration.
        """
        eta = self._heuristic()
        tau = np.full((self.n_cities, self.n_cities), self.tau_init)
        np.fill_diagonal(tau, 0.0)

        self.best_tour = None
        self.best_dist = np.inf
        self.history   = []

        for iteration in range(1, self.n_iter + 1):
            all_tours   = [self._build_tour(tau, eta) for _ in range(self.n_ants)]
            all_lengths = [self._tour_length(t) for t in all_tours]

            self._update_pheromone(tau, all_tours, all_lengths)

            iter_best_idx = int(np.argmin(all_lengths))
            iter_best     = all_lengths[iter_best_idx]

            if iter_best < self.best_dist:
                self.best_dist = iter_best
                self.best_tour = all_tours[iter_best_idx][:]
                if verbose:
                    print(
                        "Iteration: %d  Best distance: %.4f  Tour: %s"
                        % (iteration, self.best_dist, self.best_tour)
                    )

            self.history.append(self.best_dist)

        return self.best_tour, self.best_dist, self.history

    def plot(self):
        """Plot the convergence curve (best distance per iteration)."""
        if not self.history:
            print("No history to plot. Call run() first.")
            return
        pyplot.plot(self.history, ".-", markersize=3)
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Best Tour Distance")
        pyplot.title("Ant Colony Optimization – Convergence Curve")
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    # testing on TSP
    from testing.discrete_problems import TSP

    N_CITIES = 20
    SEED     = 42
    N_ANTS   = 30
    N_ITER   = 300
    ALPHA    = 1.0
    BETA     = 4.0
    RHO      = 0.1
    Q        = 100.0

    tsp = TSP.generate(n_cities=N_CITIES, seed=SEED)

    print("=" * 56)
    print("  ACO – Ant Colony Optimization")
    print("  Problem : TSP (%d cities)" % N_CITIES)
    print("=" * 56)

    aco = ACO(
        dist_matrix=tsp.dist_matrix,
        n_ants=N_ANTS,
        n_iter=N_ITER,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        Q=Q,
    )

    best_tour, best_dist, history = aco.run(verbose=True)

    print("\nBest solution (tour) :", best_tour)
    print("Best score (distance): %.5f" % best_dist)

    aco.plot()

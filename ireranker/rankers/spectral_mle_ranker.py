from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import brentq

from ireranker.oracles import (
    BidirectionalMatrixOracle,
    BudgetExceeded,
    CachedSamplingMatrixOracle,
    Oracle,
    SamplingMatrixOracle,
    WeirdSamplingMatrixOracle,
)

from .ranker import Ranker
from .registry import register_ranker


@register_ranker(
    "Spectral MLE (Refinement)",
    oracle_factories=[
        ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
        ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        (
            "weird $(1.5)$",
            lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
        ),
    ],
)
class SpectralMLERanker(Ranker):
    """
    Implements the Refinement stage of Spectral MLE.

    It assumes the initial list order (BM25) is a 'reasonably faithful'
    initialization and refines it using Coordinate-wise MLE.

    Paper: "Spectral MLE: Top-K Rank Aggregation from Pairwise Comparisons" (Chen et al., 2015).
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        *,
        p_obs: float = 0.5,  # Probability of observing an edge in the graph
        L: int = 1,  # Number of comparisons per observed pair
        mle_iterations: int = 10,  # Number of refinement cycles (T)
        w_min: float = 0.5,  # Minimum score for initialization
        w_max: float = 2.0,  # Maximum score for initialization
    ):
        super().__init__(oracle, seed)
        self.p_obs = p_obs
        self.L = L
        self.mle_iterations = mle_iterations
        self.w_min = w_min
        self.w_max = w_max

    def _rank(self) -> List[int]:
        n = self.n
        if n <= 1:
            return list(range(n))

        # --- 1. Data Collection (Building the Comparison Graph) ---
        # The paper assumes a random Erdos-Renyi graph G(n, p_obs).
        # We simulate this by iterating pairs and querying the oracle with prob p_obs.

        # wins[i] = total wins for item i
        # total_comparisons[i][j] = number of times i and j were compared
        wins = np.zeros(n)
        total_comparisons = np.zeros((n, n))

        # Adjacency list for fast neighbor lookup during MLE
        # neighbors[i] = list of j where (i, j) were compared
        neighbors = [[] for _ in range(n)]

        rng = np.random.default_rng(self.seed)

        try:
            # We iterate upper triangle to avoid double counting, but populate symmetric structures
            for i in range(n):
                for j in range(i + 1, n):
                    # Randomly decide if we observe this pair (Sparsity control)
                    if rng.random() < self.p_obs:
                        # Query the Oracle L times
                        for _ in range(self.L):
                            if self.lt(i, j):
                                # lt(i, j) -> True means i < j (j is better/winner)
                                wins[j] += 1
                            else:
                                # i is better/winner
                                wins[i] += 1

                        # Record that a comparison happened
                        total_comparisons[i, j] += self.L
                        total_comparisons[j, i] += self.L
                        neighbors[i].append(j)
                        neighbors[j].append(i)

        except BudgetExceeded:
            # If budget explodes, we proceed with whatever data we collected.
            pass

        # --- 2. Initialization (Using Provided Order) ---
        # We assume the input indices 0..n-1 are sorted by BM25 (0 is best).
        # We assign scores linearly decaying from w_max to w_min.
        # This creates the vector w^(0).
        current_w = np.linspace(self.w_max, self.w_min, n)

        # --- 3. Refinement (Coordinate-wise MLE) ---
        # We iteratively update each w_i to maximize the likelihood given its neighbors.

        for _ in range(self.mle_iterations):
            current_w = self._perform_coordinate_descent(
                n, current_w, wins, total_comparisons, neighbors
            )

        # --- 4. Output ---
        # Return indices sorted by the final refined scores (Descending).
        # argsort is ascending, so we reverse it.
        final_ranking = np.argsort(current_w)[::-1].tolist()

        return final_ranking

    def _perform_coordinate_descent(
        self, n: int, w: np.ndarray, wins: np.ndarray, N: np.ndarray, neighbors: List[List[int]]
    ) -> np.ndarray:
        """
        Performs one full pass of Coordinate-wise MLE updates.
        Updates each w_i to satisfy the stationarity condition of the BTL log-likelihood.
        """
        new_w = w.copy()

        for i in range(n):
            i_neighbors = neighbors[i]
            if not i_neighbors:
                continue

            W_i = wins[i]  # Total wins for item i

            # The stationarity equation for BTL MLE (derivative = 0):
            # (Total Wins for i) / w_i  -  Sum_{j in neighbors} ( N_ij / (w_i + w_j) ) = 0

            # We solve for w_i (denoted as tau here).
            # We use a root-finding algorithm (Brent's method).

            def objective_function(tau):
                # Avoid division by zero
                if tau <= 1e-9:
                    return 1e10

                sum_term = 0.0
                for j in i_neighbors:
                    sum_term += N[i, j] / (tau + new_w[j])

                return (W_i / tau) - sum_term

            # Root finding requires a bracket [a, b] where signs are opposite.
            # BTL scores are relative, but we bound them to keep numerical stability.
            low, high = 1e-4, 100.0

            try:
                f_low = objective_function(low)
                f_high = objective_function(high)

                if f_low * f_high < 0:
                    # Signs differ, root exists in bracket
                    res = brentq(objective_function, low, high)
                    new_w[i] = res
                else:
                    # Monotonic function didn't cross 0 in range.
                    # Push towards the direction of the gradient.
                    if abs(f_low) < abs(f_high):
                        new_w[i] = low
                    else:
                        new_w[i] = high
            except ValueError:
                # If optimization fails, retain previous estimate
                pass

        return new_w

from __future__ import annotations

import random
from typing import List

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
    "PAC Top-k (Optimized)",
    oracle_factories=[
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
        ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
        (
            "weird $(1.5)$",
            lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
        ),
    ],
)
class PACOptimizedRanker(Ranker):
    """Heavily optimized PAC Top-k that minimizes comparisons.

    Key optimizations:
    1. Use BM25 prior to select anchors directly (no Round 1 comparisons)
    2. Only compare top candidates against anchors (not all n items)
    3. Early stopping when we have confident top-k
    4. Minimal refinement in Round 3

    Expected comparisons: O(K * sqrt(n)) instead of O(n * K)
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        top_k: int = 10,
        candidate_pool_multiplier: int = 3,
        bubble_refine: bool = False,
        num_child: int = 1,
    ):
        """
        Args:
            oracle: Comparison oracle
            seed: Random seed
            top_k: Number of items to select
            candidate_pool_multiplier: How many candidates to consider per top-k slot
                                      (default 3 means we look at top 30 items for top-10)
            bubble_refine: If True, run bubble sort on the selected top-k set to refine ordering
            num_child: Window size for bubble sort (only used if bubble_refine=True)
        """
        self.k = top_k
        self.pool_mult = candidate_pool_multiplier
        self.bubble_refine = bubble_refine
        self.num_child = num_child
        super().__init__(oracle, seed)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(self.seed)

    def _rank(self) -> List[int]:
        """Execute optimized PAC top-k."""
        n = self.n
        K = min(self.k, n)

        if n <= K:
            return list(range(n))

        try:
            # OPTIMIZATION 1: Select anchors from BM25 order without comparisons
            # Use K/2 anchors evenly spaced
            num_anchors = max(3, K // 2)
            step = max(1, n // num_anchors)
            anchors = [i * step for i in range(num_anchors) if i * step < n]

            # OPTIMIZATION 2: Only consider top candidates, not all n items
            # Candidate pool = top (K * pool_mult) items by BM25
            pool_size = min(n, K * self.pool_mult)
            candidates = list(range(pool_size))

            # Build winner sets: for each anchor, find candidates that beat it
            winner_sets = {}
            for anchor in anchors:
                winners = set()
                for candidate in candidates:
                    if candidate == anchor:
                        winners.add(candidate)  # anchor is its own winner
                        continue
                    if self.gt(candidate, anchor):
                        winners.add(candidate)
                winner_sets[anchor] = winners

            # Sort anchors by quality (smaller winner set = better)
            sorted_anchors = sorted(anchors, key=lambda a: (len(winner_sets[a]), a))

            # Greedy: collect winners from best anchors
            top_k_set = set()
            for anchor in sorted_anchors:
                top_k_set.update(winner_sets[anchor])
                if len(top_k_set) >= K:
                    break

            # Take top K by BM25 order
            result = sorted(list(top_k_set))[:K]

            # If we don't have enough, just take top K from BM25
            if len(result) < K:
                result = list(range(K))

            # Optional: Refine the top-K ordering with bubble sort
            if self.bubble_refine and len(result) > 1:
                self._bubble_sort_refine(result)

            # Append remaining items
            remaining = [i for i in range(n) if i not in result]
            return result + remaining

        except BudgetExceeded:
            if "result" in locals():
                # If we have selected a top-k set (even if refinement failed), return it
                remaining = [i for i in range(n) if i not in result]
                return result + remaining
            return list(range(n))

    def _best_in_window(self, order: List[int], start: int, end: int) -> int:
        """Return offset of the best doc in [start, end)."""
        if end - start <= 1:
            return 0
        best = start
        for idx in range(start + 1, end):
            if self.lt(order[best], order[idx]):
                best = idx
        return best - start

    def _bubble_sort_refine(self, ranking: List[int]) -> None:
        """Run bubble sort in-place on the top-K ranking to refine BM25 ordering.

        This uses comparisons to fix any errors in the BM25 ordering within
        the PAC-selected set.
        """
        n = len(ranking)
        if n <= 1:
            return

        window = self.num_child + 1
        last_start = len(ranking) - window

        for i in range(n):
            start_ind = last_start
            end_ind = last_start + window
            is_change = False

            while True:
                if start_ind < i:
                    start_ind = i
                if end_ind > len(ranking):
                    end_ind = len(ranking)
                if end_ind <= start_ind:
                    end_ind = start_ind + 1

                best_offset = self._best_in_window(ranking, start_ind, end_ind)
                if best_offset != 0:
                    best_idx = start_ind + best_offset
                    ranking[start_ind], ranking[best_idx] = (
                        ranking[best_idx],
                        ranking[start_ind],
                    )
                    if not is_change:
                        is_change = True
                        if (
                            last_start != len(ranking) - window
                            and best_offset == len(ranking[start_ind:end_ind]) - 1
                        ):
                            last_start += len(ranking[start_ind:end_ind]) - 1

                if start_ind == i:
                    break

                if not is_change:
                    last_start -= self.num_child

                start_ind -= self.num_child
                end_ind -= self.num_child

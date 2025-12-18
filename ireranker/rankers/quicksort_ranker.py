from __future__ import annotations

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
    "Quick Sort (Classic)",
    oracle_factories=[
        ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
        # ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
        # ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        # (
        #     "weird $(1.5)$",
        #     lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
        # ),
    ],
)
class QuicksortTopKRanker(Ranker):
    """PRP-QuickSort with Partial QuickSort (top-k stopping)."""

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        top_k: int | None = None,
    ):
        super().__init__(oracle, seed)
        self.top_k = top_k

    def _rank(self) -> List[int]:
        n = self.n
        order = list(range(n))
        if n <= 1:
            return order

        k = self._effective_k(n)
        try:
            self._partial_quicksort(order, 0, n - 1, k)
        except BudgetExceeded:
            return order
        return order

    def _effective_k(self, n: int) -> int:
        k = self.top_k
        if k is None or k <= 0 or k >= n:
            # For k <= 0 or k >= n, behave like full quicksort.
            return n
        return k

    def _partition(self, order: List[int], lo: int, hi: int) -> int:
        """Partition using the middle element as pivot (descending order: best first)."""
        mid = (lo + hi) // 2
        pivot_id = order[mid]
        # Move pivot to the end
        order[mid], order[hi] = order[hi], order[mid]

        store = lo
        for j in range(lo, hi):
            # Put items better than the pivot on the left:
            # ComparableDoc version would be: if order[j] > pivot
            if self.gt(order[j], pivot_id):
                order[store], order[j] = order[j], order[store]
                store += 1

        # Place pivot after all "better-than-pivot" items
        order[store], order[hi] = order[hi], order[store]
        return store

    def _partial_quicksort(self, order: List[int], lo: int, hi: int, k: int) -> None:
        """
        Partial QuickSort (Martínez 2004): only recurse into segments
        intersecting the [0, k) prefix (top-k region).
        Converted to an explicit stack to avoid recursion-depth blowups.
        """
        if lo >= hi or lo >= k:
            return

        stack: list[tuple[int, int]] = [(lo, hi)]
        while stack:
            left, h = stack.pop()
            if left >= h or left >= k:
                continue

            p = self._partition(order, left, h)

            # Always process the left side (it may contain top-k items)
            stack.append((left, p - 1))
            # Only process the right side if the pivot lies inside the top-k prefix
            if p < k - 1:
                stack.append((p + 1, h))

from __future__ import annotations

from typing import List

from ireranker.oracles import Oracle, SamplingMatrixOracle

from .ranker import CacheRanker
from .registry import register_ranker


@register_ranker(
    "Quick Sort (Classic)",
    default_oracle_factory=lambda seed: SamplingMatrixOracle(seed=seed),
)
class QuicksortTopKRanker(CacheRanker):
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
        self._partial_quicksort(order, 0, n - 1, k)
        return order

    def _effective_k(self, n: int) -> int:
        k = self.top_k
        if k is None or k <= 0 or k >= n:
            return n
        return k

    def _better(self, a: int, b: int) -> bool:
        """Return True if a should be ranked before b."""
        return self.lt(b, a)

    def _partition(self, order: List[int], lo: int, hi: int) -> int:
        """Partition using the middle element as pivot."""
        mid = (lo + hi) // 2
        pivot_id = order[mid]
        order[mid], order[hi] = order[hi], order[mid]

        store = lo
        for j in range(lo, hi):
            if self._better(order[j], pivot_id):
                order[store], order[j] = order[j], order[store]
                store += 1

        order[store], order[hi] = order[hi], order[store]
        return store

    def _partial_quicksort(
        self,
        order: List[int],
        lo: int,
        hi: int,
        k: int,
    ) -> None:
        """
        Partial QuickSort (Martínez 2004): only recurse into segments
        intersecting the [0, k) prefix (top-k region).
        """

        if lo >= hi or lo >= k:
            return

        p = self._partition(order, lo, hi)
        self._partial_quicksort(order, lo, p - 1, k)

        if p < k - 1:
            self._partial_quicksort(order, p + 1, hi, k)


# Legacy class alias for backward compatibility (does not register an alias name)
QuickSortRanker = QuicksortTopKRanker

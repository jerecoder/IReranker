from __future__ import annotations

import os
from typing import List

from ireranker.oracles import (
    BidirectionalFlanSeq2SeqOracle,
    BudgetExceeded,
    Oracle,
    SamplingMatrixOracle,
)

from .ranker import Ranker
from .registry import register_ranker


@register_ranker(
    "prp sort (classic)",
    oracle_factories=[
        (
            "flan-live",
            lambda seed: BidirectionalFlanSeq2SeqOracle(
                model_name=os.environ.get("FLAN_MODEL_NAME", "google/flan-t5-xl"),
                quantization=os.environ.get("FLAN_QUANT", "8bit"),
                device=os.environ.get("FLAN_DEVICE", "cuda"),
            ),
        ),
        (
            "sampling",
            lambda seed: SamplingMatrixOracle(
                seed=seed,
                comparison_limit_per_task=True,
            ),
        ),
    ],
)
class PRPSortingRanker(Ranker):
    """PRP-HeapSort with partial HeapSort (top-k stopping).

    This is a faithful translation of the `heapSort` / `heapify` logic
    from `PairwiseLlmRanker` into the CacheRanker/Oracle setting.
    All comparisons are routed through `self.gt(i, j)` in the same
    relative argument order as `ComparableDoc.__gt__` was used in:
        if l < n and arr[l] > arr[i]
        if r < n and arr[r] > arr[largest]
    """

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
            self._partial_heapsort(order, k)
        except BudgetExceeded:
            pass

        order.reverse()
        return order

    def _effective_k(self, n: int) -> int:
        """Normalize top_k to a usable k (identical behavior to QuicksortTopKRanker)."""
        k = self.top_k
        if k is None or k <= 0 or k >= n:
            return n
        return k

    def _heapify(self, order: List[int], n: int, i: int) -> None:
        """Heapify subtree rooted at index i (max-heap), identical control flow
        to PairwiseLlmRanker.heapify, but using `self.gt` instead of `__gt__`."""

        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.gt(order[left], order[i]):
            largest = left

        if right < n and self.gt(order[right], order[largest]):
            largest = right

        if largest != i:
            order[i], order[largest] = order[largest], order[i]
            self._heapify(order, n, largest)

    def _partial_heapsort(self, order: List[int], k: int) -> None:
        """Partial HeapSort: same as PairwiseLlmRanker.heapSort(arr, k).
        Builds a max-heap, then extracts the max element k times (or until
        exhausted), performing heapify after each extraction.
        """

        n = len(order)
        ranked = 0

        for i in range(n // 2, -1, -1):
            self._heapify(order, n, i)

        for i in range(n - 1, 0, -1):
            order[i], order[0] = order[0], order[i]
            ranked += 1
            if ranked == k:
                break

            self._heapify(order, i, 0)

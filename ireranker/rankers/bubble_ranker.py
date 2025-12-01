from __future__ import annotations

from typing import List

from ireranker.oracles import BidirectionalMatrixOracle, Oracle, SamplingMatrixOracle

from .ranker import Ranker
from .registry import register_ranker


@register_ranker(
    "Bubble Sort (Classic)",
    oracle_factories=[
        ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
    ],
)
class BubbleRanker(Ranker):
    """Performs Bubble Sort based on the comparison matrices."""

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        *,
        num_child: int = 1,
        top_k: int | None = None,
    ):
        super().__init__(oracle, seed)
        self.num_child = max(1, num_child)
        self.top_k = top_k

    def _effective_k(self, n: int) -> int:
        """Normalize top_k to a usable k (behaves like full bubble sort when unset)."""
        if self.top_k is None or self.top_k <= 0 or self.top_k > n:
            return n
        return self.top_k

    def _best_in_window(self, order: List[int], start: int, end: int) -> int:
        """Return offset of the best doc in [start, end)."""
        if end - start <= 1:
            return 0
        best = start
        for idx in range(start + 1, end):
            if self.lt(order[best], order[idx]):
                best = idx
        return best - start

    def _rank(self) -> List[int]:
        n = self.n
        ranking = list(range(n))

        if n <= 1:
            return ranking

        window = self.num_child + 1
        k = self._effective_k(n)
        last_start = len(ranking) - window

        for i in range(k):
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
                    ranking[start_ind], ranking[best_idx] = ranking[best_idx], ranking[start_ind]
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

        return ranking

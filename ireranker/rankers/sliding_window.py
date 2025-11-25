from __future__ import annotations

from typing import List

from ireranker.oracles import Oracle
from ireranker.types import RankingTask

from .ranker import CacheRanker


# @register_ranker("sliding")
class SlidingWindowRanker(CacheRanker):
    """PRP-Sliding-K: sliding-window passes starting from the bottom of the list.

    One pass is a single bottom-up sweep over adjacent pairs. We perform a
    small, fixed number of passes (K), which is typically enough to get a
    good Top-K ranking while keeping complexity close to O(K · N) with
    K ≪ N.
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        passes: int = 10,
    ):
        super().__init__(oracle, seed)
        self.passes = passes

    def rank(self, task: RankingTask) -> List[int]:
        self.task = task
        order = list(range(len(task.candidate_ids)))
        n = len(order)

        if n <= 1:
            return order

        num_passes = min(self.passes, n - 1)
        for _ in range(num_passes):
            swapped = False

            for i in range(n - 2, -1, -1):
                if self.lt(order[i], order[i + 1]):
                    order[i], order[i + 1] = order[i + 1], order[i]
                    swapped = True

            if not swapped:
                break

        return order

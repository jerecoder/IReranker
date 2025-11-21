from __future__ import annotations

from typing import List

from ireranker.rankers.registry import register_ranker
from ireranker.types import Oracle, RankingTask

from .ranker import CacheRanker


@register_ranker("bubbly")
class BubbleRanker(CacheRanker):
    """Performs Bubble Sort based on the comparison matrices."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def rank(self, task: RankingTask) -> List[int]:
        self.task = task
        order = list(range(len(task.candidate_ids)))
        swapped = True
        while swapped:
            swapped = False
            for i in range(len(order) - 1):
                if self.lt(order[i], order[i + 1]):
                    order[i], order[i + 1] = order[i + 1], order[i]
                    swapped = True
        return order

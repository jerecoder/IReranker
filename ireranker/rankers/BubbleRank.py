from __future__ import annotations

from typing import List

from ireranker.types import Oracle, RankingTask

from .base import Ranker
from .registry import register_ranker


@register_ranker("bubbly")
class BubbleRanker(Ranker):
    """Performs Bubble Sort based on the comparisson matrices"""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def rank(self, task: RankingTask) -> List[int]:
        order = list(range(len(task.candidate_ids)))
        if not order:
            return order

        swapped = True
        while swapped:
            swapped = False
            for i in range(len(order) - 1):
                left_idx = order[i]
                right_idx = order[i + 1]
                left_doc = task.candidate_ids[left_idx]
                right_doc = task.candidate_ids[right_idx]

                if self.oracle.sample_lt(task.query_id, left_doc, right_doc):
                    order[i], order[i + 1] = order[i + 1], order[i]
                    swapped = True
        return order

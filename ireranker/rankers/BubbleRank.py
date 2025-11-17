from __future__ import annotations

import random
from typing import List

from ireranker.types import RankingTask

from .base import Ranker
from .registry import register_ranker


@register_ranker("bubbly")
class BubbleRanker(Ranker):
    """Performs Bubble Sort based on the comparisson matrices"""

    def rank(self, task: RankingTask) -> List[int]:
        return list(range(len(task.candidate_ids)))

from __future__ import annotations

import random
from typing import List

from ireranker.oracles import Oracle
from ireranker.types import RankingTask

from .ranker import CacheRanker
from .registry import register_ranker


@register_ranker("random")
class RandomRanker(CacheRanker):
    """Returns a deterministic pseudo-random permutation given a seed."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def _rank(self) -> List[int]:
        rng = random.Random(self.seed)
        idx = list(range(len(self.task.candidate_ids)))
        rng.shuffle(idx)
        return idx

from __future__ import annotations

import random
from typing import List

from ireranker.types import Oracle, RankingTask

from .base import Ranker
from .registry import register_ranker


@register_ranker("identity")
class IdentityRanker(Ranker):
    """Returns candidates as-is."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def rank(self, task: RankingTask) -> List[int]:
        return list(range(len(task.candidate_ids)))


@register_ranker("reverse")
class ReverseRanker(Ranker):
    """Returns reversed order of candidates."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def rank(self, task: RankingTask) -> List[int]:
        return list(reversed(range(len(task.candidate_ids))))


@register_ranker("random")
class RandomRanker(Ranker):
    """Returns a deterministic pseudo-random permutation given a seed."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def rank(self, task: RankingTask) -> List[int]:
        rng = random.Random(self.seed)
        idx = list(range(len(task.candidate_ids)))
        rng.shuffle(idx)
        return idx

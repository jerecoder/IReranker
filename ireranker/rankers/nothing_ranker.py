from __future__ import annotations

from typing import List

from ireranker.oracles import Oracle

from .ranker import CacheRanker
from .registry import register_ranker

@register_ranker("nothing")
class NothingRanker(CacheRanker):

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def _rank(self) -> List[int]:
        self.task = self.task
        order = list(range(self.n))
        return order

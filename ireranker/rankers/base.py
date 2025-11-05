from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ireranker.types import RankingTask


class Ranker(ABC):
    """Abstract Ranker interface.

    Implementations must be deterministic when given the same parameters/seed.
    The `rank` method returns indices into `task.candidate_ids` in the predicted order.
    """

    name: str = "base"

    def __init__(self, seed: int | None = None):
        self.seed = seed

    @abstractmethod
    def rank(self, task: RankingTask) -> List[int]:
        """Return a permutation of indices for the candidate list."""
        raise NotImplementedError

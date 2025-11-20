from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ireranker.types import Oracle, RankingTask


class Ranker(ABC):
    """Abstract Ranker interface.

    Implementations must be deterministic when given the same parameters/seed.
    The `rank` method returns indices into `task.candidate_ids` in the predicted order.
    """

    name: str = "base"

    def __init__(self, oracle: Oracle, seed: int | None = None):
        self.seed = seed
        self.oracle = oracle
        self.comparissons = 0
        self.task: RankingTask

    def set_dataset(self, dataset: str, *, split: str = "test") -> None:
        """Update the oracle with a new dataset before ranking."""
        self.oracle.load_dataset(dataset, split=split)

    @abstractmethod
    def rank(self, task: RankingTask) -> List[int]:
        """Return a permutation of indices for the candidate list."""
        raise NotImplementedError

    def lt(self, i: int, j: int) -> bool:
        """Return True when item i should be ranked after item j."""
        self.comparissons += 1
        return self.oracle.sample_lt(self.task, i, j)

    def reset_comparisons(self) -> None:
        """Reset comparison counter before a new evaluation."""
        self.comparissons = 0

    @property
    def comparisons(self) -> int:
        """Return the number of comparisons made by this ranker."""
        return self.comparissons

from __future__ import annotations

from abc import ABC, abstractmethod

from ireranker.types import RankingTask


class Oracle(ABC):
    """Abstract oracle that answers pairwise comparison queries."""

    @abstractmethod
    def load_dataset(self, dataset: str, *, split: str = "test") -> None:
        """Load comparison data for the given dataset, replacing any previous state."""

    @abstractmethod
    def sample_lt(self, task: RankingTask, i: int, j: int) -> bool:
        """Return True when doc at index i should be ranked after doc at index j."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ireranker.oracles import Oracle
from ireranker.types import RankingTask


class Ranker(ABC):
    """Abstract Ranker interface.

    Implementations must be deterministic when given the same parameters/seed.
    The `rank` method returns indices into `task.candidate_ids` in the predicted order.
    """

    name: str = "base"

    def __init__(self, oracle: Oracle, seed: int | None = None):
        self.oracle = oracle
        self.oracle_label = getattr(oracle, "name", None)
        self.display_name = (
            f"{self.name} [{self.oracle_label}]" if self.oracle_label else self.name
        )
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the deterministic seed for this ranker and its oracle (if supported)."""
        self.seed = seed if seed is not None else 0
        try:
            self.oracle.set_seed(self.seed)
        except AttributeError:
            pass

    def set_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: list[str] | None = None,
        matrix_model: str | None = None,
    ) -> None:
        """Update the oracle with a new dataset before ranking."""
        self.oracle.load_dataset(
            dataset, split=split, query_ids=query_ids, matrix_model=matrix_model
        )

    def rank(self, task: RankingTask) -> List[int]:
        """Return a permutation of indices for the candidate list."""
        self.oracle.set_task(task)
        self.task = task
        self.n = len(task.candidate_ids)
        return self._rank()

    @abstractmethod
    def _rank(self) -> List[int]:
        """Return a permutation of indices for the candidate list."""
        raise NotImplementedError

    def lt(self, i: int, j: int) -> bool:
        """Return True when item i is less than item j"""
        self.oracle.set_task(self.task)
        return self.oracle.lt(i, j)

    def gt(self, i: int, j: int) -> bool:
        """Return True when item j is less than item i"""
        return self.lt(j, i)

    def reset_comparisons(self) -> None:
        """Reset comparison counter before a new evaluation."""
        try:
            self.oracle.reset_comparisons()
        except AttributeError:
            pass

    @property
    def comparisons(self) -> int:
        """Return the number of comparisons made by this ranker."""
        try:
            return int(getattr(self.oracle, "comparisons"))
        except Exception:
            return 0

    @property
    def comparissons(self) -> int:  # pragma: no cover - backward compatibility typo
        """Legacy alias for comparisons."""
        return self.comparisons

    @property
    def cache_hits(self) -> int:
        try:
            return int(getattr(self.oracle, "cache_hits"))
        except Exception:
            return 0

    @property
    def comparison_calls(self) -> int:
        try:
            return int(getattr(self.oracle, "comparison_calls"))
        except Exception:
            return 0

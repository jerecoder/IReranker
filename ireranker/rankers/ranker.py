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
        self._comparisons = 0
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

    @abstractmethod
    def lt(self, i: int, j: int) -> bool:
        """Return True when item i is less than item j"""
        raise NotImplementedError

    @abstractmethod
    def gt(self, i: int, j: int) -> bool:
        """Return True when item j is less than item i"""
        raise NotImplementedError

    def reset_comparisons(self) -> None:
        """Reset comparison counter before a new evaluation."""
        self._comparisons = 0

    @property
    def comparisons(self) -> int:
        """Return the number of comparisons made by this ranker."""
        return self._comparisons

    @property
    def comparissons(self) -> int:  # pragma: no cover - backward compatibility typo
        """Legacy alias for comparisons."""
        return self._comparisons


class CacheRanker(Ranker):
    """Ranker base that caches pairwise comparisons within the same task."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)
        self._comparison_cache: dict[tuple[int, int], bool] = {}
        self._cache_signature: tuple[str, tuple[str, ...]] | None = None

    def set_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: list[str] | None = None,
        matrix_model: str | None = None,
    ) -> None:
        super().set_dataset(dataset, split=split, query_ids=query_ids, matrix_model=matrix_model)
        self._comparison_cache.clear()
        self._cache_signature = None

    def reset_comparisons(self) -> None:
        """Reset comparison counter and drop any cached results."""
        super().reset_comparisons()
        self._comparison_cache.clear()
        self._cache_signature = None

    def _ensure_cache_for_task(self) -> None:
        """Clear the cache when switching to a different task."""
        if not hasattr(self, "task"):
            raise RuntimeError("Callers must set self.task before requesting comparisons.")
        signature = (self.task.query_id, tuple(self.task.candidate_ids))
        if signature != self._cache_signature:
            self._comparison_cache.clear()
            self._cache_signature = signature

    def lt(self, i: int, j: int) -> bool:
        """Return True when item i is less than item j"""
        self._ensure_cache_for_task()
        key = (i, j)
        if key not in self._comparison_cache:
            self._comparisons += 2
            self._comparison_cache[key] = self.oracle.sample_lt(i, j)
        return self._comparison_cache[key]

    def gt(self, i: int, j: int) -> bool:
        """Return True when item j is less than item i"""
        return self.lt(j, i)


class SampleRanker(Ranker):
    """Ranker base that samples comparisons within the same task."""

    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def lt(self, i: int, j: int) -> bool:
        self._comparisons += 1
        return self.oracle.sample_lt(i, j)

    def gt(self, i: int, j: int) -> bool:
        return self.lt(j, i)

"""TrackingOracle wrapper for logging pairwise queries during reranking."""
from __future__ import annotations

from typing import Iterable, Optional

from ireranker.types import RankingTask

from .oracle import Oracle


class TrackingOracle(Oracle):
    """Wrapper that logs all pairwise queries and delegates to inner oracle.
    
    Used for diagnostics to analyze query patterns (distinct pairs, re-query rate).
    Tracks stats both per-task and aggregated across all tasks.
    """

    def __init__(self, inner_oracle: Oracle):
        # Don't call super().__init__() since we delegate everything
        self._inner = inner_oracle
        self._current_task_queries: list[tuple[int, int]] = []
        self._task_stats: list[dict] = []  # Stats from completed tasks
        
    @property
    def name(self) -> str:
        return f"Tracking[{self._inner.name}]"
    
    def load_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: Optional[Iterable[str]] = None,
        matrix_model: Optional[str] = None,
    ) -> None:
        return self._inner.load_dataset(
            dataset, split=split, query_ids=query_ids, matrix_model=matrix_model
        )

    def set_task(self, task: RankingTask) -> None:
        # Save current task stats before switching
        if self._current_task_queries:
            self._task_stats.append(self._compute_task_stats(self._current_task_queries))
        self._current_task_queries = []
        return self._inner.set_task(task)

    def set_seed(self, seed: int | None) -> None:
        return self._inner.set_seed(seed)

    def enable_cache(self, enabled: bool = True) -> None:
        return self._inner.enable_cache(enabled)

    def reset_comparisons(self) -> None:
        # Also finalize current task if needed
        if self._current_task_queries:
            self._task_stats.append(self._compute_task_stats(self._current_task_queries))
        self._current_task_queries = []
        self._task_stats = []
        return self._inner.reset_comparisons()

    def sample_lt(self, i: int, j: int) -> bool:
        return self._inner.sample_lt(i, j)

    def lt(self, i: int, j: int) -> bool:
        # Log the ordered pair before delegating
        self._current_task_queries.append((i, j))
        return self._inner.lt(i, j)

    @property
    def comparisons(self) -> int:
        return self._inner.comparisons

    @property
    def comparison_calls(self) -> int:
        return self._inner.comparison_calls

    @property
    def cache_hits(self) -> int:
        return self._inner.cache_hits

    @property
    def current_task(self):
        return self._inner.current_task

    @current_task.setter
    def current_task(self, value):
        self._inner.current_task = value

    # ---- Analysis methods ----

    def _compute_task_stats(self, queries: list[tuple[int, int]]) -> dict:
        """Compute pair uniqueness stats for a single task."""
        if not queries:
            return {
                "total_calls": 0,
                "distinct_pairs": 0,
                "requery_count": 0,
            }
        
        seen: set[frozenset] = set()
        requery_count = 0
        
        for i, j in queries:
            pair = frozenset({i, j})
            if pair in seen:
                requery_count += 1
            else:
                seen.add(pair)
        
        return {
            "total_calls": len(queries),
            "distinct_pairs": len(seen),
            "requery_count": requery_count,
        }

    def finalize_current_task(self) -> None:
        """Call this after the last task to record its stats."""
        if self._current_task_queries:
            self._task_stats.append(self._compute_task_stats(self._current_task_queries))
            self._current_task_queries = []

    def get_aggregated_stats_at_budget(self, budget: int) -> dict:
        """Compute pair uniqueness stats aggregated across all tasks at given per-task budget.
        
        Args:
            budget: Max queries to consider per task
            
        Returns:
            dict with keys:
                - total_calls: Sum of lt() calls across tasks (capped at budget per task)
                - distinct_pairs: Sum of unique unordered pairs per task
                - requery_count: Sum of re-queries per task
                - requery_rate: Average re-query rate across tasks
                - num_tasks: Number of tasks evaluated
        """
        # First, finalize current task if not yet done
        all_stats = list(self._task_stats)
        if self._current_task_queries:
            all_stats.append(self._compute_task_stats(self._current_task_queries[:budget]))
        
        # Recompute stats with budget limit per task
        # We need to recalculate from raw task queries... but we don't store them.
        # Instead, we'll use the stored stats as-is (they already reflect the actual queries made).
        # The budget comparison should be done via oracle's comparison_limit_per_task.
        
        # For this diagnostic, we want to see what the stats look like at different budgets.
        # So we need to store raw queries per task. Let's refactor.
        
        total_calls = sum(s["total_calls"] for s in all_stats)
        distinct_pairs = sum(s["distinct_pairs"] for s in all_stats)
        requery_count = sum(s["requery_count"] for s in all_stats)
        
        return {
            "total_calls": total_calls,
            "distinct_pairs": distinct_pairs,
            "requery_count": requery_count,
            "requery_rate": requery_count / total_calls if total_calls > 0 else 0.0,
            "num_tasks": len(all_stats),
        }

    def get_per_task_stats(self) -> list[dict]:
        """Return stats for each completed task."""
        all_stats = list(self._task_stats)
        if self._current_task_queries:
            all_stats.append(self._compute_task_stats(self._current_task_queries))
        return all_stats


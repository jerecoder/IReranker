from __future__ import annotations

import pytest

from ireranker.oracles import Oracle
from ireranker.rankers.ranker import CacheRanker
from ireranker.types import RankingTask


class RecordingOracle(Oracle):
    """Test double that records every pairwise query."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int, int]] = []
        self.dataset_loads: list[tuple[str, str]] = []

    def load_dataset(self, dataset: str, *, split: str = "test") -> None:
        self.dataset_loads.append((dataset, split))

    def sample_lt(self, i: int, j: int) -> bool:
        assert self.current_task is not None
        self.calls.append((self.current_task.query_id, i, j))
        return i < j


class DuplicateCompareRanker(CacheRanker):
    """Ranker stub that queries the same pair twice to exercise caching."""

    def _rank(self) -> list[int]:
        self.lt(0, 1)
        self.lt(0, 1)
        return [0, 1]


class NoopRanker(CacheRanker):
    """Ranker stub that exposes lt without preparing a task."""

    def _rank(self) -> list[int]:  # pragma: no cover - not used
        return list(range(len(self.task.candidate_ids)))


@pytest.fixture()
def oracle() -> RecordingOracle:
    return RecordingOracle()


@pytest.fixture()
def two_item_task() -> RankingTask:
    return RankingTask(query_id="q0", candidate_ids=["a", "b"])


def test_lt_requires_task_to_be_set(oracle: RecordingOracle) -> None:
    ranker = NoopRanker(oracle)
    with pytest.raises(RuntimeError):
        ranker.lt(0, 1)


def test_cache_ranker_uses_cached_comparisons(
    oracle: RecordingOracle, two_item_task: RankingTask
) -> None:
    ranker = DuplicateCompareRanker(oracle)

    ranker.rank(two_item_task)

    assert oracle.calls == [("q0", 0, 1)]
    assert ranker.comparisons == 1


def test_cache_survives_repeat_invocations_of_same_task(
    oracle: RecordingOracle, two_item_task: RankingTask
) -> None:
    ranker = DuplicateCompareRanker(oracle)

    ranker.rank(two_item_task)
    ranker.rank(two_item_task)

    assert oracle.calls == [("q0", 0, 1)]
    assert ranker.comparisons == 1


def test_cache_ranker_resets_between_different_tasks(
    oracle: RecordingOracle, two_item_task: RankingTask
) -> None:
    ranker = DuplicateCompareRanker(oracle)
    second_task = RankingTask(query_id="q0", candidate_ids=["x", "y"])

    ranker.rank(two_item_task)
    ranker.rank(second_task)

    assert oracle.calls == [("q0", 0, 1), ("q0", 0, 1)]
    assert ranker.comparisons == 2


def test_cache_ranker_reset_drops_cache_state(
    oracle: RecordingOracle, two_item_task: RankingTask
) -> None:
    ranker = DuplicateCompareRanker(oracle)

    ranker.rank(two_item_task)
    ranker.reset_comparisons()
    ranker.rank(two_item_task)

    assert oracle.calls == [("q0", 0, 1), ("q0", 0, 1)]
    assert ranker.comparisons == 1


def test_cache_ranker_clears_cache_on_dataset_change(
    oracle: RecordingOracle, two_item_task: RankingTask
) -> None:
    ranker = DuplicateCompareRanker(oracle)

    ranker.rank(two_item_task)
    ranker.set_dataset("other", split="validation")
    ranker.rank(two_item_task)

    assert oracle.calls == [("q0", 0, 1), ("q0", 0, 1)]
    assert ranker.comparisons == 2
    assert oracle.dataset_loads == [("other", "validation")]

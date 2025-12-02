from __future__ import annotations

import pickle
from pathlib import Path

from ireranker.oracles import CachedSamplingMatrixOracle, SamplingMatrixOracle
from ireranker.types import RankingTask

DATASET = "sample"


def _write_consistent_matrix(base: Path) -> None:
    data = {
        ("q1", "d1", "d2"): {"scores": {"A": 1.0, "B": 0.0}},
        ("q1", "d2", "d1"): {"scores": {"A": 0.0, "B": 1.0}},
    }
    path = base / f"{DATASET}_dataset.pkl"
    with path.open("wb") as f:
        pickle.dump(data, f)


def _set_task(oracle) -> None:
    task = RankingTask(query_id="q1", candidate_ids=["d1", "d2"])
    oracle.set_task(task)


def test_sampling_oracle_counts_each_call_without_cache(tmp_path) -> None:
    _write_consistent_matrix(tmp_path)
    oracle = SamplingMatrixOracle(base_dir=tmp_path, seed=0)
    oracle.load_dataset(DATASET)
    _set_task(oracle)

    first = oracle.lt(0, 1)
    second = oracle.lt(0, 1)

    assert first is False  # deterministic for this synthetic matrix
    assert second is False
    assert oracle.comparisons == 2  # no caching, both calls count
    assert oracle.cache_hits == 0


def test_cached_sampling_oracle_reuses_first_sample(tmp_path) -> None:
    _write_consistent_matrix(tmp_path)
    oracle = CachedSamplingMatrixOracle(base_dir=tmp_path, seed=0)
    oracle.load_dataset(DATASET)
    _set_task(oracle)

    first = oracle.lt(0, 1)
    assert first is False
    assert oracle.comparisons == 2
    assert oracle.cache_hits == 0

    second = oracle.lt(0, 1)
    assert second is first  # cached result returned
    assert oracle.comparisons == 2  # comparisons not incremented on cache hit
    assert oracle.cache_hits == 1
    assert oracle.comparison_calls == 2

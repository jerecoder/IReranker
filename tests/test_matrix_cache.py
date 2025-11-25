from __future__ import annotations

import pickle
from pathlib import Path

from ireranker.oracles import BidirectionalMatrixOracle
from ireranker.oracles.oracle import clear_matrix_cache, load_matrix


def _write_matrix(tmp_path: Path) -> Path:
    data = {
        ("q1", "d1", "d2"): {"scores": {"A": 1.0, "B": 0.0}},
        ("q2", "d3", "d4"): {"scores": {"A": 0.0, "B": 1.0}},
    }
    path = tmp_path / "foo_dataset.pkl"
    with path.open("wb") as f:
        pickle.dump(data, f)
    return path


def test_matrix_cache_reuses_loaded_object(tmp_path: Path) -> None:
    clear_matrix_cache()
    _write_matrix(tmp_path)

    first = load_matrix("foo", base_dir=tmp_path)
    second = load_matrix("foo", base_dir=tmp_path)

    assert first is second  # cached object reused

    clear_matrix_cache()
    third = load_matrix("foo", base_dir=tmp_path)
    assert third is not first  # cache cleared forces reload


def test_matrix_oracle_filters_query_ids(tmp_path: Path) -> None:
    clear_matrix_cache()
    _write_matrix(tmp_path)
    oracle = BidirectionalMatrixOracle(base_dir=tmp_path)

    oracle.load_dataset("foo", query_ids=["q1"])
    matrix = oracle._ensure_matrix_loaded()

    assert set(q for q, _, _ in matrix.keys()) == {"q1"}

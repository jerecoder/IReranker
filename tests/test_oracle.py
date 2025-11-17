from __future__ import annotations

import pickle
from numbers import Real
from pathlib import Path
from typing import Dict, Tuple

import pytest

from ireranker.config import EXTERNAL_DATA_DIR
from ireranker.types import BidirectionalMatrixOracle, MatrixKey


def _dataset_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[-1].lower() if parts else path.stem.lower()


def _entry_preference(entry: Dict[str, object]) -> str | None:
    scores = _extract_scores(entry)
    if "A" not in scores or "B" not in scores:
        return None
    if scores["A"] == scores["B"]:
        return None
    return "A" if scores["A"] > scores["B"] else "B"


def _extract_scores(entry: Dict[str, object]) -> Dict[str, float]:
    raw_scores = entry.get("scores")
    iterable = raw_scores.items() if isinstance(raw_scores, dict) else (raw_scores or [])
    scores: Dict[str, float] = {}
    for pair in iterable:  # type: ignore
        if (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], Real)
        ):
            scores[pair[0].strip().upper()] = float(pair[1])
    return scores


def _find_consistent_pair(
    matrix: Dict[MatrixKey, Dict[str, object]],
) -> Tuple[MatrixKey, bool] | None:
    for key, entry in matrix.items():
        if not isinstance(entry, dict):
            continue
        qid, doc_a, doc_b = key
        reverse = (qid, doc_b, doc_a)
        other = matrix.get(reverse)
        if other is None:
            continue
        forward_pref = _entry_preference(entry)
        reverse_pref = _entry_preference(other)
        if forward_pref == "A" and reverse_pref == "B":
            return key, True  # doc_a wins
        if forward_pref == "B" and reverse_pref == "A":
            return key, False  # doc_b wins
    return None


@pytest.fixture(scope="module")
def dataset_and_pair() -> Tuple[str, Tuple[MatrixKey, bool]]:
    base = EXTERNAL_DATA_DIR / "reranking-matrices"
    if not base.exists():
        pytest.skip("No reranking matrices available under data/external.")
    preferred = base / "Reranking"
    candidates = sorted(preferred.glob("*.pkl")) if preferred.exists() else []
    if not candidates:
        candidates = sorted(base.rglob("*.pkl"))
    if not candidates:
        pytest.skip("Reranking matrices folder is empty.")

    for path in candidates:
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            continue
        pair = _find_consistent_pair(data)
        if pair is not None:
            dataset = _dataset_from_path(path)
            return dataset, pair
    pytest.skip("Could not find a matrix with consistent bidirectional votes.")


@pytest.fixture()
def oracle(dataset_and_pair: Tuple[str, Tuple[MatrixKey, bool]]) -> BidirectionalMatrixOracle:
    dataset, _ = dataset_and_pair
    oracle = BidirectionalMatrixOracle()
    oracle.load_dataset(dataset)
    return oracle


def test_oracle_matches_real_votes(
    oracle: BidirectionalMatrixOracle,
    dataset_and_pair: Tuple[str, Tuple[MatrixKey, bool]],
) -> None:
    _, (key, expected) = dataset_and_pair
    qid, doc_a, doc_b = key
    assert oracle.sample_outcome(qid, doc_a, doc_b) is expected
    assert oracle.sample_outcome(qid, doc_b, doc_a) is (not expected)


def test_missing_pair_returns_false(oracle: BidirectionalMatrixOracle) -> None:
    assert oracle.sample_outcome("__missing__", "docA", "docB") is False

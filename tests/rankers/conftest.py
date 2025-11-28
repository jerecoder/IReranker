from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, TYPE_CHECKING

import pytest

DATASET_NAME = "webis-touche2020"
EXPECTED_RESULTS_PATH = Path(__file__).with_name("expected_webis_results.csv")

if TYPE_CHECKING:
    from ireranker.oracles import BidirectionalMatrixOracle
    from ireranker.types import RankingDataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_beir_artifacts_available() -> None:
    try:
        from ireranker.config import EXTERNAL_DATA_DIR
    except ModuleNotFoundError as exc:
        pytest.skip(f"ireranker dependencies missing (install extras): {exc}")

    data_dir = EXTERNAL_DATA_DIR / "beir" / DATASET_NAME
    matrix_dir = EXTERNAL_DATA_DIR / "reranking-matrices"
    has_matrix = any(matrix_dir.rglob(f"*{DATASET_NAME}*.pkl")) if matrix_dir.exists() else False
    if not data_dir.exists() or not has_matrix:
        pytest.skip(
            f"Required BEIR artifacts for {DATASET_NAME} missing "
            f"(dataset dir: {data_dir}, rerank matrices under: {matrix_dir})."
        )


def _load_expected_metrics(path: Path) -> Dict[str, Dict[str, Dict[int, Dict[str, float | int]]]]:
    if not path.exists():
        pytest.skip(f"Benchmark summary not found at {path}")
    metrics: Dict[str, Dict[str, Dict[int, Dict[str, float | int]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            ranker = row["ranker"]
            k = int(row["k"])
            metrics.setdefault(model, {}).setdefault(ranker, {})[k] = {
                "NDCG": float(row["NDCG"])
            }
    return metrics


@pytest.fixture(scope="session")
def expected_webis_results() -> Dict[str, Dict[str, Dict[int, Dict[str, float | int]]]]:
    _ensure_beir_artifacts_available()
    return _load_expected_metrics(EXPECTED_RESULTS_PATH)


@pytest.fixture(scope="session")
def webis_touche_dataset() -> RankingDataset:
    _ensure_beir_artifacts_available()
    from ireranker.data.loaders import load_beir_dataset

    return load_beir_dataset(DATASET_NAME)


@pytest.fixture(scope="session")
def webis_touche_oracle() -> BidirectionalMatrixOracle:
    _ensure_beir_artifacts_available()
    from ireranker.oracles import BidirectionalMatrixOracle

    oracle = BidirectionalMatrixOracle()
    oracle.load_dataset(DATASET_NAME)
    return oracle


@pytest.fixture(scope="session")
def expected_k_values(
    expected_webis_results: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float | int]]]]
) -> list[int]:
    ks: set[int] = set()
    for by_model in expected_webis_results.values():
        for by_k in by_model.values():
            ks.update(by_k.keys())
    return sorted(ks)

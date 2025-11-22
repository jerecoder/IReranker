from __future__ import annotations

import pytest

import csv
from pathlib import Path

import pytest

from ireranker.config import REPORTS_DIR

DATASET_NAME = "webis-touche2020"


def _load_summary_rows(dataset: str) -> list[dict[str, str]]:
    summary_path = REPORTS_DIR / "beir-metrics" / dataset / "summary.csv"
    if not summary_path.exists():
        pytest.skip(f"Summary not found for {dataset}: {summary_path}")
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_latest_summary_matches_expected(expected_webis_results) -> None:
    rows = _load_summary_rows(DATASET_NAME)
    observed = {(row["ranker"], int(row["k"])): row for row in rows}
    observed_rankers = {r for r, _ in observed}

    expected = expected_webis_results
    expected_rankers = set(expected.keys())
    common_rankers = expected_rankers & observed_rankers
    if not common_rankers:
        pytest.skip("No expected rankers present in summary; skipping consistency check.")
    missing = {(name, k) for name in common_rankers for k in expected[name]} - set(observed)
    if missing:
        pytest.fail(f"Missing expected rows in summary: {sorted(missing)}")

    for ranker_name in common_rankers:
        for k, exp in expected[ranker_name].items():
            row = observed[(ranker_name, k)]
            assert float(row["NDCG"]) == pytest.approx(exp["NDCG"], rel=1e-4)

from __future__ import annotations

import csv

import pytest

from ireranker.config import REPORTS_DIR

DATASET_NAME = "webis-touche2020"


def _load_summary_rows(model: str, dataset: str) -> list[dict[str, str]]:
    summary_path = REPORTS_DIR / "beir-metrics" / model / dataset / "summary.csv"
    if not summary_path.exists():
        pytest.skip(f"Summary not found for {dataset} (model={model}): {summary_path}")
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_latest_summary_matches_expected(expected_webis_results) -> None:
    for model, expected_by_ranker in expected_webis_results.items():
        rows = _load_summary_rows(model, DATASET_NAME)
        observed = {
            (row["ranker"], row.get("oracle", ""), int(row["k"])): row for row in rows
        }
        observed_rankers = {(r, o) for r, o, _ in observed}

        expected_rankers = set(expected_by_ranker.keys())
        common_rankers = expected_rankers & observed_rankers
        if not common_rankers:
            pytest.skip(
                f"No expected rankers present in summary for model={model}; skipping consistency check."
            )
        missing = {
            (name, oracle, k)
            for name, oracle in common_rankers
            for k in expected_by_ranker[(name, oracle)]
        } - set(observed)
        if missing:
            pytest.fail(f"Missing expected rows in summary for model={model}: {sorted(missing)}")

        for ranker_name, oracle in common_rankers:
            for k, exp in expected_by_ranker[(ranker_name, oracle)].items():
                row = observed[(ranker_name, oracle, k)]
                assert float(row["NDCG"]) == pytest.approx(exp["NDCG"], rel=1e-4)

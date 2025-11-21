from __future__ import annotations

import pytest

from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker, list_rankers
import ireranker.rankers.bubble_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.mohajer_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.random_ranker  # noqa: F401 - ensure registration side effects

DATASET_NAME = "webis-touche2020"


def test_rankers_match_expected_webis_results(
    webis_touche_dataset,
    webis_touche_oracle,
    expected_webis_results,
    expected_k_values,
) -> None:
    expected = expected_webis_results
    ranker_names = sorted(expected.keys())
    available_rankers = set(list_rankers())
    tested_ranker_names = [name for name in ranker_names if name in available_rankers]
    if not tested_ranker_names:
        pytest.skip("No expected rankers are registered; skipping benchmark assertions.")
    k_values = expected_k_values

    rankers = []
    for name in tested_ranker_names:
        params = {"oracle": webis_touche_oracle}
        if name == "random":
            params["seed"] = 0
        ranker = get_ranker(name, **params)
        ranker.set_dataset(DATASET_NAME)
        rankers.append(ranker)

    rows = evaluate_rankers_beir(rankers, webis_touche_dataset, k_values)
    observed = {(row["ranker"], row["k"]): row for row in rows}

    assert set(observed.keys()) == {(name, k) for name in tested_ranker_names for k in k_values}

    for ranker_name in tested_ranker_names:
        for k, exp in expected[ranker_name].items():
            row = observed[(ranker_name, k)]
            assert row["NDCG"] == pytest.approx(exp["NDCG"], rel=1e-4)

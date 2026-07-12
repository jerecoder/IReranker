from __future__ import annotations

import numpy as np
import pytest

from experiments.reviewer_response.analyze import (
    bootstrap_interval,
    expected_conditions,
    holm_adjust,
    query_seed_average,
    sign_flip_p_value,
)
from experiments.reviewer_response.common import (
    EXPERIMENT_1_METHODS,
    EXPERIMENT_2_METHODS,
    METHOD_VARIANTS,
    PILOT_QUERY_IDS,
    STOCHASTIC_SEEDS,
    TOKEN_BUDGETS,
    method_seeds,
)


def test_frozen_experiment_condition_counts_and_seed_policy() -> None:
    experiment_1 = expected_conditions(EXPERIMENT_1_METHODS)
    experiment_2 = expected_conditions(EXPERIMENT_2_METHODS)
    assert len(experiment_1) == 17
    assert len(experiment_2) == 23
    assert method_seeds("mohajer") == STOCHASTIC_SEEDS
    assert method_seeds("prp") == STOCHASTIC_SEEDS
    assert method_seeds("mohajer_listwise") == STOCHASTIC_SEEDS
    assert method_seeds("setwise") == [42]
    assert TOKEN_BUDGETS == [100000, 50000]
    assert METHOD_VARIANTS["prp"] == "randomized_direction_prp_heapsort_top10"
    assert PILOT_QUERY_IDS == ["INEX_XER-65", "QALD2_te-28", "QALD2_tr-3"]


def test_query_seed_average_averages_stochastic_quality_before_query_statistics() -> None:
    rows = [
        {"query_id": "q1", "ndcg10": 0.2, "total_model_tokens": 10},
        {"query_id": "q1", "ndcg10": 0.4, "total_model_tokens": 12},
        {"query_id": "q2", "ndcg10": 0.7, "total_model_tokens": 20},
        {"query_id": "q2", "ndcg10": 0.9, "total_model_tokens": 22},
    ]
    numeric = {
        "ndcg10",
        "stage_a_tokens",
        "stage_b_tokens",
        "logical_comparisons",
        "choice_events",
        "prompt_instances",
        "generation_invocations",
        "document_instances",
        "avg_documents_per_prompt",
        "encoder_nonpad_tokens",
        "encoder_padded_slots",
        "decoder_tokens",
        "total_model_tokens",
        "inference_seconds",
        "invalid_outputs",
        "inconsistent_outputs",
        "query_wall_seconds",
        "peak_gpu_memory_bytes",
    }
    for row in rows:
        for field in numeric:
            row.setdefault(field, 0)
    result = query_seed_average(rows)
    assert result["q1"]["ndcg10"] == pytest.approx(0.3)
    assert result["q2"]["ndcg10"] == pytest.approx(0.8)
    assert result["q1"]["total_model_tokens"] == pytest.approx(11)


def test_paired_statistics_are_deterministic_and_handle_null_effect() -> None:
    values = np.array([0.0, 0.0, 0.0, 0.0])
    low, high = bootstrap_interval(values, seed=7, resamples=100)
    assert (low, high) == (0.0, 0.0)
    assert sign_flip_p_value(values, seed=7, resamples=100) == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

from __future__ import annotations

import pytest

from experiments.mohajer_hybrid_probe.common import (
    eligible_query_ids,
    ndcg_at_k,
    pareto_methods,
    quality_gate_methods,
    stable_query_order,
    strong_mohajer_failure,
)


def test_probe_query_selection_is_stable_and_dataset_specific() -> None:
    qids = ["1", "2", "3", "4", "5"]
    first = stable_query_order("fiqa", qids)
    assert first == stable_query_order("fiqa", reversed(qids))
    assert first != stable_query_order("scifact", qids)
    assert sorted(first) == sorted(qids)


def test_probe_selection_excludes_short_or_duplicate_bm25_lists() -> None:
    queries = {qid: "query" for qid in ("valid", "short", "duplicate", "no-qrels")}
    qrels = {qid: {} for qid in ("valid", "short", "duplicate")}
    candidates = {
        "valid": ["a", "b", "c"],
        "short": ["a", "b"],
        "duplicate": ["a", "a", "b"],
        "no-qrels": ["a", "b", "c"],
    }
    assert eligible_query_ids(
        queries, qrels, candidates, candidates_per_query=3
    ) == {"valid"}


def test_probe_ndcg_uses_full_qrels_with_linear_gain() -> None:
    assert ndcg_at_k(["candidate"], {"candidate": 1, "outside": 2}, 1) == pytest.approx(0.5)


def test_strong_mohajer_gate_requires_both_arms_to_fail_every_query() -> None:
    bm25 = {"a": 0.5, "b": 0.5, "c": 0.5}
    assert strong_mohajer_failure(
        bm25,
        {"a": 0.4, "b": 0.4, "c": 0.4},
        {"a": 0.45, "b": 0.45, "c": 0.45},
    )
    assert not strong_mohajer_failure(
        bm25,
        {"a": 0.4, "b": 0.4, "c": 0.4},
        {"a": 0.51, "b": 0.4, "c": 0.4},
    )


def test_token_pareto_front_keeps_quality_cost_tradeoff() -> None:
    rows = [
        {"method": "cheap", "avg_tokens": 10, "ndcg10": 0.4},
        {"method": "good", "avg_tokens": 20, "ndcg10": 0.5},
        {"method": "dominated", "avg_tokens": 30, "ndcg10": 0.45},
    ]
    assert pareto_methods(rows) == {"cheap", "good"}


def test_quality_gate_requires_effect_size_and_two_query_wins() -> None:
    baseline = {"a": 0.4, "b": 0.4, "c": 0.4}
    methods = {
        "passes": {"a": 0.5, "b": 0.5, "c": 0.4},
        "one_win": {"a": 0.7, "b": 0.4, "c": 0.4},
        "tiny": {"a": 0.41, "b": 0.41, "c": 0.4},
    }
    assert quality_gate_methods(baseline, methods) == {"passes"}

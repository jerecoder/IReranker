import pytest

try:
    from ireranker.evaluation.beir import evaluate_rankers_beir
    from ireranker.rankers import get_ranker
    import ireranker.rankers.nothing_ranker  # noqa: F401
    import ireranker.rankers.mohajer_ranker  # noqa: F401
    from ireranker.types import RankingDataset, RankingTask
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"BEIR dependencies unavailable: {exc}", allow_module_level=True)


def test_beir_eval_returns_rows_for_rankers(dummy_oracle_factory):
    tasks = []
    for t in range(3):
        cands = [f"d{t}-{i}" for i in range(5)]
        y_true = [float(4 - i) for i in range(5)]
        tasks.append(RankingTask(query_id=f"q{t}", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)
    r_nothing = get_ranker("bm25", seed=123, oracle=dummy_oracle_factory())
    r_mohajer = get_ranker("mohajer (ir)", oracle=dummy_oracle_factory())

    rows = evaluate_rankers_beir([r_nothing, r_mohajer], ds, [5])
    names = {row["ranker"] for row in rows}
    assert names == {"bm25", "mohajer (ir)"}
    for row in rows:
        assert row["k"] == 5
        for key in ("NDCG", "MAP", "Recall", "Precision"):
            val = float(row[key])
            assert 0.0 <= val <= 1.0
        assert int(row["Comparisons"]) >= 0
        assert float(row["NDCG_per_comp"]) >= 0.0

from __future__ import annotations

import pytest

try:
    from ireranker.evaluation.beir import (
        dataset_to_beir_qrels,
        evaluate_rankers_beir,
    )
    from ireranker.rankers import get_ranker
    import ireranker.rankers.nothing_ranker  # noqa: F401
    from ireranker.types import RankingDataset, RankingTask
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"BEIR dependencies unavailable: {exc}", allow_module_level=True)


def test_beir_qrels_and_eval_basic(dummy_oracle):
    tasks = []
    cands = [f"d{i}" for i in range(5)]
    y_true = [float(4 - i) for i in range(5)]
    tasks.append(RankingTask(query_id="q0", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)
    r = get_ranker("bm25", oracle=dummy_oracle, seed=0)

    qrels = dataset_to_beir_qrels(ds)
    assert len(qrels) == 1
    qid = ds.tasks[0].query_id
    assert qid in qrels
    assert all(v > 0 for v in qrels[qid].values())

    rows = evaluate_rankers_beir([r], ds, [1, 3, 5])
    assert len(rows) == 3
    for row in rows:
        assert row["ranker"] == "bm25"
        assert row["k"] in (1, 3, 5)
        for key in ("NDCG", "MAP", "Recall", "Precision"):
            val = float(row[key])
            assert 0.0 <= val <= 1.0
        assert int(row["Comparisons"]) >= 0
        assert float(row["NDCG_per_comp"]) >= 0.0

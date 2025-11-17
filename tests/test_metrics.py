from ireranker.evaluation.beir import dataset_to_beir_qrels, evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.types import RankingDataset, RankingTask


def test_beir_metrics_on_minimal(dummy_oracle):
    tasks = []
    for t in range(2):
        cands = [f"d{t}-{i}" for i in range(5)]
        y_true = [float(4 - i) for i in range(5)]
        tasks.append(RankingTask(query_id=f"q{t}", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)
    r = get_ranker("identity", oracle=dummy_oracle)

    qrels = dataset_to_beir_qrels(ds)
    assert len(qrels) == 2
    for rels in qrels.values():
        assert all(v > 0 for v in rels.values())

    rows = evaluate_rankers_beir([r], ds, [1, 3, 5])
    assert len(rows) == 3
    for row in rows:
        assert row["ranker"] == "identity"
        assert row["k"] in (1, 3, 5)
        for key in ("NDCG", "MAP", "Recall", "Precision"):
            val = float(row[key])
            assert 0.0 <= val <= 1.0

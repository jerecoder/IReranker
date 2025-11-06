from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
import ireranker.rankers.baselines  # noqa: F401
from ireranker.types import RankingDataset, RankingTask


def test_beir_eval_returns_rows_for_rankers():
    tasks = []
    for t in range(3):
        cands = [f"d{t}-{i}" for i in range(5)]
        y_true = [float(4 - i) for i in range(5)]
        tasks.append(RankingTask(query_id=f"q{t}", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)
    r_identity = get_ranker("identity")
    r_reverse = get_ranker("reverse")

    rows = evaluate_rankers_beir([r_identity, r_reverse], ds, [5])
    # Expect one row per ranker for k=5
    names = {row["ranker"] for row in rows}
    assert names == {"identity", "reverse"}
    for row in rows:
        assert row["k"] == 5
        for key in ("NDCG", "MAP", "Recall", "Precision"):
            val = float(row[key])
            assert 0.0 <= val <= 1.0

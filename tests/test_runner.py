from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
import ireranker.rankers.RandomRanker  # noqa: F401
import ireranker.rankers.BubbleRanker  # noqa: F401
from ireranker.types import RankingDataset, RankingTask


def test_beir_eval_returns_rows_for_rankers(dummy_oracle_factory):
    tasks = []
    for t in range(3):
        cands = [f"d{t}-{i}" for i in range(5)]
        y_true = [float(4 - i) for i in range(5)]
        tasks.append(RankingTask(query_id=f"q{t}", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)
    r_random = get_ranker("random", seed=123, oracle=dummy_oracle_factory())
    r_bubbly = get_ranker("bubbly", oracle=dummy_oracle_factory())

    rows = evaluate_rankers_beir([r_random, r_bubbly], ds, [5])
    # Expect one row per ranker for k=5
    names = {row["ranker"] for row in rows}
    assert names == {"random", "bubbly"}
    for row in rows:
        assert row["k"] == 5
        for key in ("NDCG", "MAP", "Recall", "Precision"):
            val = float(row[key])
            assert 0.0 <= val <= 1.0
        assert int(row["Comparisons"]) >= 0
        assert float(row["NDCG_per_comp"]) >= 0.0

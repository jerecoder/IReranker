from ireranker.evaluation.beir import dataset_to_beir_qrels, evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import Oracle

# Ensure built-in rankers are registered
import ireranker.rankers.random_ranker  # noqa: F401
import ireranker.rankers.mohajer_ranker  # noqa: F401
from ireranker.types import RankingDataset, RankingTask


def test_beir_metrics_on_minimal(dummy_oracle):
    tasks = []
    for t in range(2):
        cands = [f"d{t}-{i}" for i in range(5)]
        y_true = [float(4 - i) for i in range(5)]
        tasks.append(RankingTask(query_id=f"q{t}", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)
    r = get_ranker("random", oracle=dummy_oracle, seed=0)

    qrels = dataset_to_beir_qrels(ds)
    assert len(qrels) == 2
    for rels in qrels.values():
        assert all(v > 0 for v in rels.values())

    rows = evaluate_rankers_beir([r], ds, [1, 3, 5])
    assert len(rows) == 3
    for row in rows:
        assert row["ranker"] == "random"
        assert row["k"] in (1, 3, 5)
        for key in ("NDCG", "MAP", "Recall", "Precision"):
            val = float(row[key])
            assert 0.0 <= val <= 1.0
        assert int(row["Comparisons"]) >= 0
        assert float(row["NDCG_per_comp"]) >= 0.0


def test_beir_eval_reseeds_rankers():
    class CoinFlipOracle(Oracle):
        def __init__(self) -> None:
            super().__init__()
            import random

            self._rng = random.Random(0)

        def load_dataset(
            self, dataset: str, *, split: str = "test", query_ids=None
        ) -> None:  # noqa: ARG002, ANN001
            return None

        def sample_lt(self, i: int, j: int) -> bool:  # noqa: ARG002
            return bool(self._rng.getrandbits(1))

        def set_seed(self, seed: int | None) -> None:
            import random

            self._rng = random.Random(seed)

    tasks = []
    for t in range(2):
        cands = [f"d{t}-{i}" for i in range(6)]
        y_true = [float(5 - i) for i in range(6)]
        tasks.append(RankingTask(query_id=f"q{t}", candidate_ids=cands, y_true=y_true))
    ds = RankingDataset(tasks=tasks)

    ranker = get_ranker("mohajer", oracle=CoinFlipOracle(), seed=999)

    rows1 = evaluate_rankers_beir([ranker], ds, [3], seed=123)
    rows2 = evaluate_rankers_beir([ranker], ds, [3], seed=123)

    assert rows1 == rows2

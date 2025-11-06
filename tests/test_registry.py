from ireranker.rankers import get_ranker, list_rankers
import ireranker.rankers.baselines  # noqa: F401 - ensure registration side effects
from ireranker.types import RankingDataset, RankingTask


def test_registry_lists_baselines():
    names = list_rankers()
    assert "identity" in names
    assert "reverse" in names
    assert "random" in names


def test_get_ranker_seed_determinism():
    r1 = get_ranker("random", seed=42)
    r2 = get_ranker("random", seed=42)
    t = RankingTask(query_id="q0", candidate_ids=[f"d{i}" for i in range(10)], y_true=None)
    ds = RankingDataset(tasks=[t])
    assert r1.rank(ds.tasks[0]) == r2.rank(ds.tasks[0])

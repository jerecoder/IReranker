from ireranker.rankers import get_ranker, list_rankers
import ireranker.rankers.random_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.mohajer_ranker  # noqa: F401 - ensure registration side effects
from ireranker.types import RankingDataset, RankingTask


def test_registry_lists_baselines():
    names = list_rankers()
    assert "random" in names
    assert "mohajer" in names


def test_get_ranker_seed_determinism(dummy_oracle_factory):
    r1 = get_ranker("random", seed=42, oracle=dummy_oracle_factory())
    r2 = get_ranker("random", seed=42, oracle=dummy_oracle_factory())
    t = RankingTask(query_id="q0", candidate_ids=[f"d{i}" for i in range(10)], y_true=None)
    ds = RankingDataset(tasks=[t])
    assert r1.rank(ds.tasks[0]) == r2.rank(ds.tasks[0])

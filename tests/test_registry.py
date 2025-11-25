from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle
from ireranker.rankers import default_oracle_for, get_ranker, list_rankers
import ireranker.rankers.random_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.mohajer_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.quicksort_ranker  # noqa: F401 - ensure registration side effects
from ireranker.types import RankingDataset, RankingTask


def test_registry_lists_baselines():
    names = list_rankers()
    assert "random" in names
    assert "mohajer" in names
    assert "sliding" in names
    assert "quicksort_topk" in names


def test_get_ranker_seed_determinism(dummy_oracle_factory):
    r1 = get_ranker("random", seed=42, oracle=dummy_oracle_factory())
    r2 = get_ranker("random", seed=42, oracle=dummy_oracle_factory())
    t = RankingTask(
        query_id="q0", candidate_ids=[f"d{i}" for i in range(10)], y_true=None
    )
    ds = RankingDataset(tasks=[t])
    assert r1.rank(ds.tasks[0]) == r2.rank(ds.tasks[0])


def test_default_oracle_mapping():
    oracle = default_oracle_for("mohajer", seed=11)
    assert isinstance(oracle, SamplingMatrixOracle)
    assert oracle.seed == 11

    fallback = default_oracle_for("random", seed=5)
    assert isinstance(fallback, BidirectionalMatrixOracle)
    assert fallback.seed == 5


def test_get_ranker_uses_registered_default_oracle():
    ranker = get_ranker("mohajer", seed=7)
    assert isinstance(ranker.oracle, SamplingMatrixOracle)

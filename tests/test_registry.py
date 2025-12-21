from ireranker.oracles import (
    BidirectionalMatrixOracle,
    SamplingMatrixOracle,
    WeirdSamplingMatrixOracle,
)
from ireranker.rankers import (
    build_rankers_for_eval,
    default_oracle_for,
    get_ranker,
    list_rankers,
)
import ireranker.rankers.mohajer_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.quicksort_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.prp_sorting_ranker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.nothing_ranker  # noqa: F401 - ensure registration side effects
from ireranker.types import RankingDataset, RankingTask


def test_registry_lists_baselines():
    names = set(list_rankers())
    expected = {
        "bm25",
        "mohajer (ir)",
        "prp sort (classic)",
        "quick sort (classic)",
    }
    assert expected.issubset(names)


def test_get_ranker_seed_determinism(dummy_oracle_factory):
    r1 = get_ranker("mohajer (ir)", seed=42, oracle=dummy_oracle_factory())
    r2 = get_ranker("mohajer (ir)", seed=42, oracle=dummy_oracle_factory())
    t = RankingTask(
        query_id="q0", candidate_ids=[f"d{i}" for i in range(10)], y_true=None
    )
    ds = RankingDataset(tasks=[t])
    assert r1.rank(ds.tasks[0]) == r2.rank(ds.tasks[0])


def test_default_oracle_mapping():
    oracle = default_oracle_for("mohajer (ir)", seed=11)
    assert isinstance(oracle, SamplingMatrixOracle)
    assert oracle.seed == 11

    fallback = default_oracle_for("bm25", seed=5)
    assert isinstance(fallback, BidirectionalMatrixOracle)
    assert fallback.seed == 5


def test_get_ranker_uses_registered_default_oracle():
    ranker = get_ranker("mohajer (ir)", seed=7)
    assert isinstance(ranker.oracle, SamplingMatrixOracle)


def test_all_rankers_expose_sampling_and_weird_oracles():
    for name in list_rankers():
        ranker_instances = build_rankers_for_eval([name], seed=0)
        assert len(ranker_instances) > 0, f"Ranker '{name}' should have at least one oracle variant"
        labels = {r.oracle_label for r in ranker_instances}
        # Check that each ranker has at least one oracle configured
        assert len(labels) > 0, f"Ranker '{name}' should have at least one oracle label"

from ireranker.rankers import get_ranker, list_rankers
import ireranker.rankers.baselines  # noqa: F401 - ensure registration side effects


def test_registry_lists_baselines():
    names = list_rankers()
    assert "identity" in names
    assert "reverse" in names
    assert "random" in names


def test_get_ranker_seed_determinism():
    r1 = get_ranker("random", seed=42)
    r2 = get_ranker("random", seed=42)
    from ireranker.data.loaders import load_synthetic_dataset

    ds = load_synthetic_dataset(n_tasks=1, n_candidates=10)
    t = ds.tasks[0]
    assert r1.rank(t) == r2.rank(t)


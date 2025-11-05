from ireranker.data.loaders import load_synthetic_dataset
from ireranker.evaluation import metrics as M
from ireranker.evaluation.runner import evaluate
from ireranker.rankers import get_ranker
import ireranker.rankers.baselines  # noqa: F401


def test_runner_returns_metrics():
    ds = load_synthetic_dataset(n_tasks=3, n_candidates=5)
    r_identity = get_ranker("identity")
    r_reverse = get_ranker("reverse")
    metrics = {"NDCG": M.ndcg_at_k, "MRR": M.mrr}
    results = evaluate([r_identity, r_reverse], ds, metrics, k=5)
    assert set(results.keys()) == {"identity", "reverse"}
    for res in results.values():
        assert set(res.summary.keys()) == {"NDCG", "MRR"}
        # All metric values should be within [0, 1]
        assert 0.0 <= res.summary["NDCG"] <= 1.0
        assert 0.0 <= res.summary["MRR"] <= 1.0


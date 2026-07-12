from __future__ import annotations

from typing import Any

from ireranker.rankers import get_ranker

from experiments.mohajer_hybrid_probe.engine import ProbeSamplingOracle, SharedFlanT5Engine
from experiments.mohajer_hybrid_probe.methods import (
    MethodResult,
    _task,
    run_mohajer,
    run_mohajer_hybrid,
    run_standalone,
)
from experiments.reviewer_response.active_setwise import (
    run_active_setwise,
    run_standard_setwise_randomized,
)


def run_prp_heapsort(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    """Standard partial PRP-Heapsort with the same randomized PRP oracle as Mohajer."""
    oracle = ProbeSamplingOracle(
        engine=engine,
        dataset=str(row["dataset"]),
        queries={str(row["query_id"]): str(row["query"])},
        documents=documents,
        seed=seed,
        token_limit=token_budget,
    )
    ranker = get_ranker("heap sort (classic)", oracle=oracle, seed=seed, top_k=10)
    ranker.set_dataset(str(row["dataset"]), split="test", query_ids=[str(row["query_id"])])
    candidates = [str(value) for value in row["candidates"]]
    order = ranker.rank(_task(row))
    ranking = [candidates[index] for index in order]
    return MethodResult(
        ranking=ranking,
        meter=oracle.meter,
        stage_a_tokens=oracle.meter.total_model_tokens,
        stage_b_tokens=0,
    )


def execute_method(
    method: str,
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    kwargs = {
        "row": row,
        "documents": documents,
        "engine": engine,
        "seed": seed,
        "token_budget": token_budget,
    }
    if method == "prp":
        return run_prp_heapsort(**kwargs)
    if method == "mohajer":
        return run_mohajer(**kwargs)
    if method in {"setwise", "listwise"}:
        return run_standalone(method, **kwargs)
    if method == "mohajer_setwise":
        return run_mohajer_hybrid("setwise", **kwargs)
    if method == "mohajer_listwise":
        return run_mohajer_hybrid("listwise", **kwargs)
    if method == "setwise_randomized":
        return run_standard_setwise_randomized(**kwargs)
    if method == "active_setwise":
        return run_active_setwise(**kwargs)
    raise ValueError(f"Unknown reviewer-response method: {method}")


__all__ = ["MethodResult", "execute_method", "run_prp_heapsort"]

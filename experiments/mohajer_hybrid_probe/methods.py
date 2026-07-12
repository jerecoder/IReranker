from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ireranker.oracles import BudgetExceeded
from ireranker.rankers import get_ranker
from ireranker.types import RankingTask

from experiments.mohajer_hybrid_probe.common import HYBRID_STAGE_A_FRACTION
from experiments.mohajer_hybrid_probe.engine import (
    ProbeBidirectionalOracle,
    ProbeSamplingOracle,
    SharedFlanT5Engine,
    UsageMeter,
)
from experiments.robust04_cross_paradigm.methods import (
    run_listwise_rankgpt,
    run_setwise_heapsort,
)


@dataclass
class MethodResult:
    ranking: list[str]
    meter: UsageMeter
    stage_a_tokens: int
    stage_b_tokens: int


def _task(row: dict[str, Any]) -> RankingTask:
    candidates = [str(value) for value in row["candidates"]]
    qrels = {str(key): int(value) for key, value in row["qrels"].items()}
    return RankingTask(
        query_id=str(row["query_id"]),
        candidate_ids=candidates,
        y_true=[float(qrels.get(doc_id, 0)) for doc_id in candidates],
        dataset_path=str(row.get("dataset_path") or row["dataset"]),
    )


def _sampling_oracle(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> ProbeSamplingOracle:
    return ProbeSamplingOracle(
        engine=engine,
        dataset=str(row["dataset"]),
        queries={str(row["query_id"]): str(row["query"])},
        documents=documents,
        seed=seed,
        token_limit=token_budget,
    )


def _run_oracle_ranker(
    ranker_name: str,
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    oracle = _sampling_oracle(
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=token_budget,
    )
    ranker = get_ranker(ranker_name, oracle=oracle, seed=seed, top_k=10)
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


def run_mohajer(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    return _run_oracle_ranker(
        "mohajer (ir)",
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=token_budget,
    )


def run_mohajer_bubble(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    stage_a_budget = max(1, int(token_budget * HYBRID_STAGE_A_FRACTION))
    oracle = _sampling_oracle(
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=stage_a_budget,
    )
    ranker = get_ranker("mohajer (ir)", oracle=oracle, seed=seed, top_k=10)
    ranker.set_dataset(str(row["dataset"]), split="test", query_ids=[str(row["query_id"])])
    candidates = [str(value) for value in row["candidates"]]
    order = ranker.rank(_task(row))
    stage_a_tokens = oracle.meter.total_model_tokens
    oracle.token_limit = token_budget
    oracle.meter.token_limit = token_budget
    prefix_size = min(len(order), 20)
    prefix = order[:prefix_size]
    suffix = order[prefix_size:]

    # Match the repository's Mohajer+Bubble policy: locally refine the top-2K
    # prefix for K output positions, without resetting the sampled oracle/meter.
    try:
        window = 2
        last_start = len(prefix) - window
        for output_position in range(min(10, len(prefix))):
            start = last_start
            end = start + window
            changed = False
            while True:
                start = max(start, output_position)
                end = min(end, len(prefix))
                if end <= start:
                    end = start + 1
                best = start
                for index in range(start + 1, end):
                    if oracle.lt(prefix[best], prefix[index]):
                        best = index
                best_offset = best - start
                if best_offset:
                    prefix[start], prefix[best] = prefix[best], prefix[start]
                    if not changed:
                        changed = True
                        if last_start != len(prefix) - window and best_offset == end - start - 1:
                            last_start += end - start - 1
                if start == output_position:
                    break
                if not changed:
                    last_start -= 1
                start -= 1
                end -= 1
    except BudgetExceeded:
        pass

    return MethodResult(
        ranking=[candidates[index] for index in prefix + suffix],
        meter=oracle.meter,
        stage_a_tokens=stage_a_tokens,
        stage_b_tokens=oracle.meter.total_model_tokens - stage_a_tokens,
    )


def run_mohajer_hybrid(
    refinement: str,
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
    prefix_size: int = 20,
) -> MethodResult:
    stage_a_budget = max(1, int(token_budget * HYBRID_STAGE_A_FRACTION))
    oracle = _sampling_oracle(
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=stage_a_budget,
    )
    ranker = get_ranker("mohajer (ir)", oracle=oracle, seed=seed, top_k=10)
    ranker.set_dataset(str(row["dataset"]), split="test", query_ids=[str(row["query_id"])])
    candidates = [str(value) for value in row["candidates"]]
    order = ranker.rank(_task(row))
    ranking = [candidates[index] for index in order]
    stage_a_tokens = oracle.meter.total_model_tokens
    oracle.token_limit = token_budget
    oracle.meter.token_limit = token_budget
    prefix = ranking[: min(prefix_size, len(ranking))]
    suffix = ranking[len(prefix) :]

    if refinement == "setwise":
        refined = run_setwise_heapsort(
            str(row["query"]),
            prefix,
            documents,
            engine=engine,
            meter=oracle.meter,
            num_child=2,
            k=10,
        )
    elif refinement == "listwise":
        refined = run_listwise_rankgpt(
            str(row["query"]),
            prefix,
            documents,
            engine=engine,
            meter=oracle.meter,
            window_size=4,
            step_size=2,
            repeats=2,
        )
    else:
        raise ValueError(f"Unknown Mohajer refinement: {refinement}")

    return MethodResult(
        ranking=refined + suffix,
        meter=oracle.meter,
        stage_a_tokens=stage_a_tokens,
        stage_b_tokens=oracle.meter.total_model_tokens - stage_a_tokens,
    )


def run_standalone(
    method: str,
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    candidates = [str(value) for value in row["candidates"]]
    meter = UsageMeter(token_limit=token_budget)
    if method == "setwise":
        ranking = run_setwise_heapsort(
            str(row["query"]),
            candidates,
            documents,
            engine=engine,
            meter=meter,
            num_child=2,
            k=10,
        )
    elif method == "listwise":
        ranking = run_listwise_rankgpt(
            str(row["query"]),
            candidates,
            documents,
            engine=engine,
            meter=meter,
            window_size=4,
            step_size=2,
            repeats=5,
        )
    else:
        raise ValueError(method)
    return MethodResult(
        ranking=ranking,
        meter=meter,
        stage_a_tokens=0,
        stage_b_tokens=meter.total_model_tokens,
    )


def run_bubble(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    oracle = ProbeBidirectionalOracle(
        engine=engine,
        dataset=str(row["dataset"]),
        queries={str(row["query_id"]): str(row["query"])},
        documents=documents,
        seed=seed,
        token_limit=token_budget,
    )
    ranker = get_ranker("bubble sort (classic)", oracle=oracle, seed=seed, top_k=10)
    ranker.set_dataset(str(row["dataset"]), split="test", query_ids=[str(row["query_id"])])
    candidates = [str(value) for value in row["candidates"]]
    order = ranker.rank(_task(row))
    ranking = [candidates[index] for index in order]
    return MethodResult(
        ranking=ranking,
        meter=oracle.meter,
        stage_a_tokens=0,
        stage_b_tokens=oracle.meter.total_model_tokens,
    )

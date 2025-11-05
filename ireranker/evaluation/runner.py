from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from loguru import logger

from ireranker.rankers.base import Ranker
from ireranker.types import RankingDataset

MetricFunc = Callable[..., float]


@dataclass
class EvalResult:
    by_query: Dict[str, Dict[str, float]]
    summary: Dict[str, float]


def evaluate_ranker(
    ranker: Ranker,
    dataset: RankingDataset,
    metrics: Dict[str, MetricFunc],
    *,
    k: int | None = None,
    gains: str = "exp",
) -> EvalResult:
    per_query: Dict[str, Dict[str, float]] = {}
    for task in dataset.tasks:
        indices = ranker.rank(task)
        q_scores: Dict[str, float] = {}
        for name, fn in metrics.items():
            if name.lower() == "ndcg":
                val = fn(task.y_true, indices, k=k, gains=gains)  # type: ignore[arg-type]
            elif name.lower() == "dcg":
                val = fn(task.y_true, indices, k=k, gains=gains)  # type: ignore[arg-type]
            elif name.lower() in {"map", "average_precision"}:
                val = fn(task.y_true, indices, k=k)  # type: ignore[arg-type]
            elif name.lower() in {"p@k", "precision@k", "precision"}:
                if k is None:
                    raise ValueError("precision requires k")
                val = fn(task.y_true, indices, k)  # type: ignore[arg-type]
            else:
                # default signature (y_true, indices)
                val = fn(task.y_true, indices)  # type: ignore[arg-type]
            q_scores[name] = float(val)
        per_query[task.query_id] = q_scores

    # Aggregate
    keys = next(iter(per_query.values())).keys() if per_query else []
    summary = {
        m: float(sum(d[m] for d in per_query.values()) / max(1, len(per_query))) for m in keys
    }
    return EvalResult(by_query=per_query, summary=summary)


def evaluate(
    rankers: List[Ranker],
    dataset: RankingDataset,
    metrics: Dict[str, MetricFunc],
    *,
    k: int | None = None,
    gains: str = "exp",
) -> Dict[str, EvalResult]:
    results: Dict[str, EvalResult] = {}
    for r in rankers:
        logger.info(f"Evaluating ranker: {r.name}")
        results[r.name] = evaluate_ranker(r, dataset, metrics, k=k, gains=gains)
    return results

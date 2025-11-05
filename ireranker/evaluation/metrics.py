from __future__ import annotations

from math import log2
from typing import List, Optional


def _ordered_labels(
    y_true: List[float], rank_indices: List[int], k: Optional[int] = None
) -> List[float]:
    if y_true is None:
        raise ValueError("y_true is required for metric computation")
    if k is None:
        k = len(rank_indices)
    return [y_true[i] for i in rank_indices[:k]]


def precision_at_k(y_true: List[float], rank_indices: List[int], k: int) -> float:
    labels = _ordered_labels(y_true, rank_indices, k)
    rel = [1.0 if v > 0 else 0.0 for v in labels]
    return sum(rel) / max(1, k)


def average_precision(
    y_true: List[float], rank_indices: List[int], k: Optional[int] = None
) -> float:
    labels = _ordered_labels(y_true, rank_indices, k)
    rel = [1 if v > 0 else 0 for v in labels]
    if sum(rel) == 0:
        return 0.0
    hits = 0
    precisions: List[float] = []
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / sum(rel)


def mrr(y_true: List[float], rank_indices: List[int]) -> float:
    labels = _ordered_labels(y_true, rank_indices)
    for i, v in enumerate(labels, start=1):
        if v > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(
    y_true: List[float], rank_indices: List[int], k: Optional[int] = None, *, gains: str = "exp"
) -> float:
    labels = _ordered_labels(y_true, rank_indices, k)
    dcg = 0.0
    for i, rel in enumerate(labels, start=1):
        gain = (2.0**rel - 1.0) if gains == "exp" else float(rel)
        denom = log2(i + 1)
        dcg += gain / denom
    return dcg


def ndcg_at_k(
    y_true: List[float], rank_indices: List[int], k: Optional[int] = None, *, gains: str = "exp"
) -> float:
    if k is None:
        k = len(rank_indices)
    dcg = dcg_at_k(y_true, rank_indices, k, gains=gains)
    # Ideal order is descending by relevance
    ideal_indices = sorted(range(len(y_true)), key=lambda i: y_true[i], reverse=True)
    idcg = dcg_at_k(y_true, ideal_indices, k, gains=gains)
    if idcg == 0:
        return 0.0
    return dcg / idcg

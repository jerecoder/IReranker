from __future__ import annotations

from typing import Dict, List

from beir.retrieval.evaluation import EvaluateRetrieval

from ireranker.rankers.base import Ranker
from ireranker.types import RankingDataset


def dataset_to_beir_qrels(dataset: RankingDataset) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for task in dataset.tasks:
        rels: Dict[str, int] = {}
        if task.y_true is None:
            continue
        for doc_id, rel in zip(task.candidate_ids, task.y_true):
            if rel and rel > 0:
                rels[doc_id] = int(rel)
        qrels[task.query_id] = rels
    return qrels


def ranker_results_to_beir(ranker: Ranker, dataset: RankingDataset) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for task in dataset.tasks:
        indices = ranker.rank(task)
        n = len(indices)
        res: Dict[str, float] = {}
        for pos, idx in enumerate(indices):
            doc_id = task.candidate_ids[idx]
            score = float(n - pos)
            res[doc_id] = score
        results[task.query_id] = res
    return results


def evaluate_rankers_beir(
    rankers: List[Ranker], dataset: RankingDataset, k_values: List[int]
) -> List[Dict[str, float | int | str]]:
    """Evaluate rankers using BEIR metrics and return flattened rows for CSV.

    Returns a list of rows with keys: ranker, k, NDCG, MAP, Recall, Precision.
    """
    qrels = dataset_to_beir_qrels(dataset)
    rows: List[Dict[str, float | int | str]] = []
    for r in rankers:
        res = ranker_results_to_beir(r, dataset)
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, res, k_values)
        for k in k_values:
            rows.append(
                {
                    "ranker": r.name,
                    "k": int(k),
                    "NDCG": float(ndcg.get(f"NDCG@{k}", 0.0)),
                    "MAP": float(_map.get(f"MAP@{k}", 0.0)),
                    "Recall": float(recall.get(f"Recall@{k}", 0.0)),
                    "Precision": float(precision.get(f"P@{k}", 0.0)),
                }
            )
    return rows

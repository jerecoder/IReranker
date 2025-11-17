from __future__ import annotations

import math
from typing import Dict, List, Tuple

try:
    from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback evaluator

    class EvaluateRetrieval:
        """Lightweight evaluator implementing NDCG, MAP, Recall, Precision."""

        @staticmethod
        def evaluate(
            qrels: Dict[str, Dict[str, int]],
            results: Dict[str, Dict[str, float]],
            k_values: List[int],
        ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
            ks = sorted(set(int(k) for k in k_values))
            metrics = {
                "ndcg": {k: 0.0 for k in ks},
                "map": {k: 0.0 for k in ks},
                "recall": {k: 0.0 for k in ks},
                "precision": {k: 0.0 for k in ks},
            }
            total = max(len(qrels), 1)

            for qid, rel_map in qrels.items():
                retrieved = results.get(qid, {})
                ranked = sorted(retrieved.items(), key=lambda kv: kv[1], reverse=True)
                rel_count = sum(1 for v in rel_map.values() if v > 0)

                for k in ks:
                    top_docs = ranked[:k]
                    gains = [rel_map.get(doc_id, 0) for doc_id, _ in top_docs]
                    metrics["ndcg"][k] += EvaluateRetrieval._ndcg(gains, rel_map, k)
                    metrics["map"][k] += EvaluateRetrieval._ap(gains, rel_count)
                    metrics["recall"][k] += EvaluateRetrieval._recall(gains, rel_count)
                    metrics["precision"][k] += EvaluateRetrieval._precision(gains, k)

            ndcg = {f"NDCG@{k}": val / total for k, val in metrics["ndcg"].items()}
            mp = {f"MAP@{k}": val / total for k, val in metrics["map"].items()}
            recall = {f"Recall@{k}": val / total for k, val in metrics["recall"].items()}
            precision = {f"P@{k}": val / total for k, val in metrics["precision"].items()}
            return ndcg, mp, recall, precision

        @staticmethod
        def _ndcg(
            gains: List[int],
            rel_map: Dict[str, int],
            k: int,
        ) -> float:
            dcg = 0.0
            for idx, rel in enumerate(gains):
                if rel <= 0:
                    continue
                dcg += (2**rel - 1) / math.log2(idx + 2)

            ideal_rels = sorted((v for v in rel_map.values() if v > 0), reverse=True)
            ideal = 0.0
            for idx, rel in enumerate(ideal_rels[:k]):
                ideal += (2**rel - 1) / math.log2(idx + 2)
            if ideal == 0:
                return 0.0
            return dcg / ideal

        @staticmethod
        def _ap(gains: List[int], rel_count: int) -> float:
            if rel_count == 0:
                return 0.0
            hits = 0
            acc = 0.0
            for idx, rel in enumerate(gains, start=1):
                if rel > 0:
                    hits += 1
                    acc += hits / idx
            return acc / rel_count if rel_count else 0.0

        @staticmethod
        def _recall(gains: List[int], rel_count: int) -> float:
            if rel_count == 0:
                return 0.0
            hits = sum(1 for rel in gains if rel > 0)
            return hits / rel_count

        @staticmethod
        def _precision(gains: List[int], k: int) -> float:
            if k == 0:
                return 0.0
            hits = sum(1 for rel in gains if rel > 0)
            return hits / k


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

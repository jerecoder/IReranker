from __future__ import annotations

import random
from typing import Dict, Iterable, List

from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
from tqdm import tqdm

from ireranker.rankers.ranker import Ranker
from ireranker.types import RankingDataset, RankingTask

from pyserini.search.lucene import LuceneSearcher
import json
from functools import lru_cache
from pathlib import Path

BM25_TOP_K = 100

def _progress(iterable: Iterable, **kwargs) -> Iterable:
    return tqdm(iterable, **kwargs)

def dataset_to_beir_qrels(dataset: RankingDataset) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for task in dataset.tasks:
        rels: Dict[str, int] = {}
        if task.y_true is None:
            continue
        for doc_id, rel in zip(task.candidate_ids, task.y_true):
            if rel and rel > 0:
                rels[doc_id] = int(rel)
        if rels:
            qrels[task.query_id] = rels
    return qrels


def ranker_results_to_beir(
    ranker: Ranker, dataset: RankingDataset, rng: random.Random
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    tasks = dataset.tasks

    iterator = _progress(
        tasks,
        total=len(tasks),
        desc=f"Ranking ({ranker.name})",
        leave=False,
    )
    for task in iterator:
        shuffled_task = RankingTask(
            query_id=task.query_id,
            candidate_ids=list(task.candidate_ids),
            y_true=list(task.y_true) if task.y_true is not None else None,
            dataset_path=task.dataset_path,
        )

        # rng.shuffle(shuffled_task.candidate_ids)
        shuffled_task.candidate_ids = _bm25_order_candidates(shuffled_task)

        indices = ranker.rank(shuffled_task)
        n = len(indices)
        res: Dict[str, float] = {}
        for pos, idx in enumerate(indices):
            doc_id = shuffled_task.candidate_ids[idx]
            score = float(n - pos)
            res[doc_id] = score
        results[shuffled_task.query_id] = res
    return results


def evaluate_rankers_beir(
    rankers: List[Ranker],
    dataset: RankingDataset,
    k_values: List[int],
    *,
    seed: int | None = None,
) -> List[Dict[str, float | int | str]]:
    """Evaluate rankers using BEIR metrics and return flattened rows for CSV.

    Returns a list of rows with keys: ranker, k, NDCG, MAP, Recall, Precision, Comparisons, NDCG_per_comp.
    """
    qrels = dataset_to_beir_qrels(dataset)
    rows: List[Dict[str, float | int | str]] = []
    base_seed = seed if seed is not None else 0
    iter_rankers = _progress(rankers, desc="Evaluating rankers", leave=True)
    for r in iter_rankers:
        r.set_seed(base_seed)
        r.reset_comparisons()
        rng = random.Random(base_seed)
        res = ranker_results_to_beir(r, dataset, rng)
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, res, k_values)
        total_comparisons = int(r.comparisons)
        for k in k_values:
            ndcg_k = float(ndcg.get(f"NDCG@{k}", 0.0))
            rows.append(
                {
                    "ranker": r.name,
                    "k": int(k),
                    "NDCG": ndcg_k,
                    "MAP": float(_map.get(f"MAP@{k}", 0.0)),
                    "Recall": float(recall.get(f"Recall@{k}", 0.0)),
                    "Precision": float(precision.get(f"P@{k}", 0.0)),
                    "Comparisons": total_comparisons,
                    "NDCG_per_comp": (
                        float(ndcg_k / total_comparisons) if total_comparisons else 0.0
                    ),
                }
            )
    return rows

# ======= BM25 HELPERS ========
@lru_cache(maxsize=None)
def _get_lucene_searcher(dataset_root: str) -> LuceneSearcher:
    """Return a cached LuceneSearcher for the given dataset root."""
    dataset_root = str(Path(dataset_root).resolve())
    index_dir = Path(dataset_root) / "lucene-index"
    searcher = LuceneSearcher(str(index_dir))
    # Optional: tune BM25 if you want non-default parameters
    # searcher.set_bm25(k1=0.9, b=0.4)
    return searcher


@lru_cache(maxsize=None)
def _get_queries(dataset_root: str) -> Dict[str, str]:
    """Load BEIR-style queries.jsonl and cache as {query_id: text}."""
    dataset_root = str(Path(dataset_root).resolve())
    qpath = Path(dataset_root) / "queries.jsonl"
    queries: Dict[str, str] = {}
    with qpath.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]
    return queries


def _bm25_order_candidates(task: RankingTask, k: int = BM25_TOP_K) -> List[str]:
    """Return only the top-k BM25 docs (intersection with candidate_ids)."""
    if task.dataset_path is None:
        # Fall back to original order if we don't know where the index is.
        return task.candidate_ids

    dataset_root = str(Path(task.dataset_path).resolve())
    searcher = _get_lucene_searcher(dataset_root)
    queries = _get_queries(dataset_root)

    try:
        query_text = queries[task.query_id]
    except KeyError:
        # If query text is missing, keep original order.
        return task.candidate_ids

    # 1) Top-k BM25 over the full index
    hits = searcher.search(query_text, k=k)

    # 2) Keep only docs that are in the original candidate set
    candidate_set = set(task.candidate_ids)
    top_ids = [h.docid for h in hits if h.docid in candidate_set]

    # 3) If nothing intersects (unlikely), fall back
    if not top_ids:
        return task.candidate_ids

    return top_ids
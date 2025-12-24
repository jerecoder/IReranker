from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import random
from typing import Dict, Iterable, List

from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
from tqdm import tqdm

from ireranker.config import logger
from ireranker.rankers.ranker import Ranker
from ireranker.types import RankingDataset, RankingTask


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
    dataset_name = Path(tasks[0].dataset_path).name if tasks and tasks[0].dataset_path else ""
    bm25_runs = _load_bm25_run(dataset_name) if dataset_name else {}

    ranker_label = getattr(ranker, "display_name", ranker.name)
    iterator = _progress(
        tasks,
        total=len(tasks),
        desc=f"Ranking ({ranker_label})",
        leave=False,
    )
    for task in iterator:
        shuffled_task = RankingTask(
            query_id=task.query_id,
            candidate_ids=list(task.candidate_ids),
            y_true=list(task.y_true) if task.y_true is not None else None,
            dataset_path=task.dataset_path,
        )

        shuffled_task.candidate_ids = _bm25_order_candidates(shuffled_task, bm25_runs)[:100]
        # rng.shuffle(shuffled_task.candidate_ids)

        indices = ranker.rank(shuffled_task)
        n = len(indices)
        res: Dict[str, float] = {}
        for pos, idx in enumerate(indices):
            doc_id = shuffled_task.candidate_ids[idx]
            score = float(n - pos)
            res[doc_id] = score
        results[shuffled_task.query_id] = res
    return results


def _evaluate_single_ranker(
    ranker: Ranker,
    dataset: RankingDataset,
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
    *,
    seed: int,
) -> List[Dict[str, float | int | str]]:
    ranker.set_seed(seed)
    ranker.reset_comparisons()
    ranker_name = getattr(ranker, "display_name", ranker.name)
    oracle_label = getattr(ranker, "oracle_label", getattr(ranker.oracle, "name", None))
    ranker_base = getattr(ranker, "name", ranker_name)
    rng = random.Random(seed)
    res = ranker_results_to_beir(ranker, dataset, rng)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, res, k_values)
    total_comparisons = int(ranker.comparisons)
    total_cache_hits = int(getattr(ranker, "cache_hits", 0))
    rows: List[Dict[str, float | int | str]] = []
    for k in k_values:
        ndcg_k = float(ndcg.get(f"NDCG@{k}", 0.0))
        rows.append(
            {
                "ranker": ranker_base,
                "oracle": oracle_label,
                "k": int(k),
                "NDCG": ndcg_k,
                "MAP": float(_map.get(f"MAP@{k}", 0.0)),
                "Recall": float(recall.get(f"Recall@{k}", 0.0)),
                "Precision": float(precision.get(f"P@{k}", 0.0)),
                "Comparisons": total_comparisons,
                "CacheHits": total_cache_hits,
                "NDCG_per_comp": float(ndcg_k / total_comparisons) if total_comparisons else 0.0,
            }
        )
    return rows


def evaluate_rankers_beir(
    rankers: List[Ranker],
    dataset: RankingDataset,
    k_values: List[int],
    *,
    seed: int | None = None,
    max_workers: int | None = None,
) -> List[Dict[str, float | int | str]]:
    """Evaluate rankers using BEIR metrics and return flattened rows for CSV.

    Returns a list of rows with keys: ranker, k, NDCG, MAP, Recall, Precision, Comparisons, NDCG_per_comp.
    """
    qrels = dataset_to_beir_qrels(dataset)
    rows: List[Dict[str, float | int | str]] = []
    base_seed = seed if seed is not None else 0

    if max_workers is None or max_workers <= 1:
        iter_rankers = _progress(rankers, desc="Evaluating rankers", leave=True)
        for r in iter_rankers:
            rows.extend(
                _evaluate_single_ranker(
                    r,
                    dataset,
                    qrels,
                    k_values,
                    seed=base_seed,
                )
            )
        return rows

    oracle_ids = [id(r.oracle) for r in rankers if hasattr(r, "oracle")]
    if len(oracle_ids) != len(set(oracle_ids)):
        logger.warning(
            "Parallel evaluation disabled because rankers share oracle instances; "
            "run with distinct oracles to enable max_workers."
        )
        iter_rankers = _progress(rankers, desc="Evaluating rankers", leave=True)
        for r in iter_rankers:
            rows.extend(
                _evaluate_single_ranker(
                    r,
                    dataset,
                    qrels,
                    k_values,
                    seed=base_seed,
                )
            )
        return rows

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_ranker,
                r,
                dataset,
                qrels,
                k_values,
                seed=base_seed,
            ): idx
            for idx, r in enumerate(rankers)
        }
        results: list[List[Dict[str, float | int | str]] | None] = [None] * len(rankers)
        iterator = _progress(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating rankers (parallel)",
            leave=True,
        )
        for fut in iterator:
            idx = futures[fut]
            results[idx] = fut.result()
        for chunk in results:
            if chunk:
                rows.extend(chunk)
    return rows


# ======= BM25 HELPERS ========
_BM25_RUN_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "beir" / "bm25-runs"


@lru_cache(maxsize=8)
def _load_bm25_run(dataset: str) -> Dict[str, List[str]]:
    """Load a TREC BM25 run into {qid: [doc_ids...]}. Missing file -> empty dict."""
    run_path = _BM25_RUN_DIR / f"run.beir.bm25-flat.{dataset}.txt"
    runs: Dict[str, List[str]] = {}
    if not run_path.exists():
        return runs
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, *_ = parts
            runs.setdefault(qid, []).append(doc_id)
    return runs


def _bm25_order_candidates(task: RankingTask, bm25_runs: Dict[str, List[str]]) -> List[str]:
    """Order candidate_ids by BM25 run; keep any missing candidates at the end."""
    order = bm25_runs.get(task.query_id)
    if not order:
        return list(task.candidate_ids)
    seen: set[str] = set()
    ordered: List[str] = []
    for doc_id in order:
        if doc_id in task.candidate_ids and doc_id not in seen:
            ordered.append(doc_id)
            seen.add(doc_id)
    for doc_id in task.candidate_ids:
        if doc_id not in seen:
            ordered.append(doc_id)
    return ordered

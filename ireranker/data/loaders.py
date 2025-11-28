from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Dict, List, NamedTuple, Optional
import zipfile

from ireranker.config import logger
from ireranker.oracles.oracle import load_matrix
from ireranker.types import RankingDataset, RankingTask

# --- BEIR dataset loader utilities -------------------------------------------------------
_BEIR_CFG_CACHE: Optional[Dict[str, object]] = None


class MatrixLoadResult(NamedTuple):
    candidates: Dict[str, List[str]]
    pair_counts: Dict[str, int]
    expected_pairs: Dict[str, int]


def _load_beir_config() -> Dict[str, object]:
    """Load BEIR loader configuration from config/beir_loader.json (if present).

    Falls back to sensible defaults when the config is missing.
    """
    global _BEIR_CFG_CACHE
    if _BEIR_CFG_CACHE is not None:
        return _BEIR_CFG_CACHE

    cfg: Dict[str, object] = {}
    try:
        from ireranker.config import PROJ_ROOT

        cfg_path = PROJ_ROOT / "config" / "beir_loader.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    cfg.setdefault(
        "base_url",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets",
    )
    cfg.setdefault("cache_subdir", "beir")

    _BEIR_CFG_CACHE = cfg
    return cfg


def _beir_supported_name(name: str) -> Optional[str]:
    """Return normalized dataset name (no aliasing)."""
    return name.lower().strip()


def _download_beir_once(canonical: str, base_out: Path) -> str:
    """Download+unzip a BEIR dataset once; no retries.

    Logs URL and ZIP path; cleans partials and raises on failure.
    Returns the dataset directory path as string.
    """
    try:
        from beir import util  # type: ignore
    except ModuleNotFoundError:
        util = None

    cfg = _load_beir_config()
    base_url = str(cfg.get("base_url"))
    base_url = base_url.rstrip("/")
    url = f"{base_url}/{canonical}.zip"
    zip_path = base_out / f"{canonical}.zip"
    ds_dir = base_out / canonical

    try:
        base_out.mkdir(parents=True, exist_ok=True)
        if ds_dir.exists() and ds_dir.is_dir():
            try:
                if zip_path.exists():
                    zip_path.unlink()
                    logger.info(f"Removed leftover BEIR ZIP: {zip_path}")
            except Exception:
                pass
            logger.info(f"Using existing BEIR dataset at: {ds_dir}")
            return str(ds_dir)

        if util is None:
            raise ModuleNotFoundError(
                "beir package is not installed and dataset is missing; cannot download."
            )

        logger.info(f"BEIR download: {url}")
        logger.info(f"Zip path: {zip_path}")
        data_path = util.download_and_unzip(url, str(base_out))
        p = Path(data_path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Expected dataset dir missing: {p}")
        try:
            if zip_path.exists():
                zip_path.unlink()
                logger.info(f"Removed BEIR ZIP: {zip_path}")
        except Exception:
            pass
        logger.info(f"BEIR dataset ready at: {p}")
        return data_path
    except (zipfile.BadZipFile, OSError, ValueError, FileNotFoundError) as e:
        logger.error(
            f"BEIR download/unzip failed for '{canonical}': {e}. Zip: {zip_path}, Dir: {ds_dir}"
        )
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        try:
            if ds_dir.exists():
                shutil.rmtree(ds_dir, ignore_errors=True)
        except Exception:
            pass
        raise


def _load_rerank_matrix(
    dataset: str,
    split: str,
    *,
    max_queries: Optional[int] = None,
    matrix_model: Optional[str] = None,
) -> Optional[MatrixLoadResult]:
    """Load per-query candidate restrictions from a single canonical base path.

    Canonical base: EXTERNAL_DATA_DIR / "reranking-matrices"
    File format: a pickle (.pkl) of a dict keyed by (query_id, doc_id_A, doc_id_B)
    Recursively scans for *.pkl whose filename contains the dataset name and
    selects the most recent match, filtered by matrix_model (substring
    match against filename/parent). Callers should always provide matrix_model
    to keep results segmented by inference model.
    """
    try:
        matrix = load_matrix(dataset, split=split, matrix_model=matrix_model)
        acc: Dict[str, set] = {}
        pair_counts: Dict[str, int] = {}
        for key in list(matrix.keys()):
            if isinstance(key, tuple) and len(key) == 3:
                qid, a, b = key
                if isinstance(qid, str) and isinstance(a, str) and isinstance(b, str):
                    acc.setdefault(qid, set()).update([a, b])
                    pair_counts[qid] = pair_counts.get(qid, 0) + 1
        qid_order = sorted(acc.keys())
        if max_queries is not None:
            keep = set(qid_order[:max_queries])
            if keep:
                matrix_keys = list(matrix.keys())
                for k in matrix_keys:
                    if isinstance(k, tuple) and len(k) == 3 and k[0] not in keep:
                        matrix.pop(k, None)
                acc = {qid: acc[qid] for qid in qid_order if qid in keep}
                pair_counts = {qid: pair_counts.get(qid, 0) for qid in qid_order if qid in keep}
                logger.info(
                    "Filtered rerank matrix for %s/%s down to %d queries",
                    dataset,
                    split,
                    len(keep),
                )
        candidates = {qid: sorted(list(s)) for qid, s in acc.items()}
        expected_pairs = {
            qid: len(docs) * max(len(docs) - 1, 0) for qid, docs in candidates.items()
        }
        logger.info(f"Loaded rerank matrix for {dataset}/{split} (queries: {len(acc)})")
        return MatrixLoadResult(
            candidates=candidates, pair_counts=pair_counts, expected_pairs=expected_pairs
        )
    except Exception as e:
        logger.warning(f"Failed to load rerank matrix for {dataset}/{split}: {e}")
        return None


def _read_beir_qrels(data_path: Path, split: str) -> Dict[str, Dict[str, int]]:
    """Parse the BEIR qrels TSV for the requested split."""
    path = data_path / "qrels" / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing BEIR qrels file: {path}")

    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] == "query-id":
                continue
            if len(row) < 3:
                continue
            qid, doc_id, score = row[0], row[1], row[2]
            try:
                score_val = int(score)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[doc_id] = score_val
    return qrels


def load_beir_dataset(
    dataset: str,
    *,
    split: str = "test",
    max_queries: Optional[int] = None,
    matrix_model: Optional[str] = None,
) -> RankingDataset:
    """Load a BEIR dataset as a RankingDataset using a rerank matrix.

    - Candidate sets come strictly from the rerank matrix for each query.
    - y_true are aligned with candidate_ids using qrels labels (missing defaults to 0).
    - Raises FileNotFoundError if the rerank matrix is not found for the requested model; calling CLI skips dataset.

    Parameters
    - dataset: BEIR dataset name (canonical, lowercased).
    - split: BEIR split to load (usually "test").
    - seed: removed; ranker seeds are handled at ranker construction.
    - matrix_model: rerank matrix model name to filter candidate matrices (required for disambiguation).
    - max_queries: optional limit of queries for a quick run (applied before filtering by matrix).
    """
    # max_queries = None  # force full BEIR dataset; disable query pruning

    canonical = _beir_supported_name(dataset)
    if canonical is None:
        raise ValueError(f"Dataset '{dataset}' not supported via BEIR loader yet.")

    from ireranker.config import EXTERNAL_DATA_DIR

    cfg = _load_beir_config()
    cache_subdir = str(cfg.get("cache_subdir") or "beir")
    base_out = EXTERNAL_DATA_DIR / cache_subdir
    base_out.mkdir(parents=True, exist_ok=True)

    data_path = _download_beir_once(canonical, base_out)

    tasks: List[RankingTask] = []

    matrix = _load_rerank_matrix(
        canonical, split, max_queries=max_queries, matrix_model=matrix_model
    )
    if matrix is None:
        from ireranker.config import EXTERNAL_DATA_DIR

        base_only = EXTERNAL_DATA_DIR / "reranking-matrices"
        logger.info(
            f"No rerank matrix found for {canonical}/{split}. Looked under: {base_only} (recursive *.pkl search)"
        )
        raise FileNotFoundError(
            f"Rerank matrix not found for {canonical}/{split}. Skipping dataset."
        )

    candidates = matrix.candidates
    pair_counts = matrix.pair_counts
    expected_pairs = matrix.expected_pairs

    q_ids = sorted(candidates.keys())
    if max_queries is not None:
        q_ids = q_ids[:max_queries]
    logger.info(f"Using rerank matrix for {len(q_ids)} queries in {canonical}/{split}")

    qrels = _read_beir_qrels(Path(data_path), split)

    matrix_qids = set(candidates.keys())
    qrels_qids = set(qrels.keys())
    shared_qids = matrix_qids & qrels_qids
    matrix_only_qids = matrix_qids - qrels_qids
    qrels_only_qids = qrels_qids - matrix_qids

    logger.info(
        f"Query coverage for {canonical}/{split} -> "
        f"matrix: {len(matrix_qids)}, qrels: {len(qrels_qids)}, "
        f"shared: {len(shared_qids)}, matrix-only: {len(matrix_only_qids)}, "
        f"qrels-only: {len(qrels_only_qids)}"
    )

    doc_gap_counts: list[tuple[str, int]] = []
    total_qrels_docs_missing = 0

    for qid in q_ids:
        rel_map: Dict[str, int] = qrels.get(qid, {})

        matrix_docs = set(candidates.get(qid, []))
        rel_docs = set(rel_map.keys())

        missing_qrels_docs = rel_docs - matrix_docs
        if missing_qrels_docs:
            count = len(missing_qrels_docs)
            total_qrels_docs_missing += count
            doc_gap_counts.append((qid, count))

        cand_ids_set = matrix_docs | rel_docs
        cand_ids = sorted(cand_ids_set)
        y_true = [float(rel_map.get(doc_id, 0)) for doc_id in cand_ids]

        tasks.append(
            RankingTask(
                query_id=qid,
                candidate_ids=cand_ids,
                y_true=y_true,
                dataset_path=str(data_path),
            )
        )

    if doc_gap_counts:
        doc_gap_counts.sort(key=lambda x: x[1], reverse=True)
        top = ", ".join(f"{qid}({cnt})" for qid, cnt in doc_gap_counts[:5])
        logger.warning(
            "Found %d qrels docs missing from matrix candidates across %d queries (top: %s)",
            total_qrels_docs_missing,
            len(doc_gap_counts),
            top,
        )

    pair_gap_counts: list[tuple[str, int]] = []
    total_pair_gaps = 0
    for qid in q_ids:
        expected = expected_pairs.get(qid, 0)
        observed = pair_counts.get(qid, 0)
        gap = max(expected - observed, 0)
        if gap:
            pair_gap_counts.append((qid, gap))
            total_pair_gaps += gap

    if pair_gap_counts:
        pair_gap_counts.sort(key=lambda x: x[1], reverse=True)
        top_pairs = ", ".join(f"{qid}({cnt})" for qid, cnt in pair_gap_counts[:5])
        logger.warning(
            "Found %d missing pairwise comparisons across %d queries in matrix for %s/%s (top: %s)",
            total_pair_gaps,
            len(pair_gap_counts),
            canonical,
            split,
            top_pairs,
        )

    metadata = {
        "dataset": canonical,
        "split": split,
        "matrix_model": matrix_model,
        "missing_qrels_docs_total": total_qrels_docs_missing,
        "missing_qrels_queries": len(doc_gap_counts),
        "missing_qrels_top": doc_gap_counts[:5],
        "missing_pairs_total": total_pair_gaps,
        "missing_pairs_queries": len(pair_gap_counts),
        "missing_pairs_top": pair_gap_counts[:5],
        "pair_counts": {qid: pair_counts.get(qid, 0) for qid in q_ids},
        "expected_pairs": {qid: expected_pairs.get(qid, 0) for qid in q_ids},
    }

    return RankingDataset(tasks=tasks, metadata=metadata)

from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle
import shutil
from typing import Dict, List, Optional
import zipfile

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback to stdlib logging
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ireranker")

from ireranker.types import RankingDataset, RankingTask

# --- BEIR dataset loader utilities -------------------------------------------------------
_BEIR_CFG_CACHE: Optional[Dict[str, object]] = None


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


def _load_rerank_matrix(dataset: str, split: str) -> Optional[Dict[str, List[str]]]:
    """Load per-query candidate restrictions from a single canonical base path.

    Canonical base: EXTERNAL_DATA_DIR / "reranking-matrices"
    File format: a pickle (.pkl) of a dict keyed by (query_id, doc_id_A, doc_id_B)
    Recursively scans for *.pkl whose filename contains the dataset name and
    selects the most recent match.
    """
    try:
        from ireranker.config import EXTERNAL_DATA_DIR

        base = EXTERNAL_DATA_DIR / "reranking-matrices"
        # Try pickle formats
        # Recursively scan a single canonical base for matching *.pkl
        candidates: List[Path] = []
        if base.exists():
            for p in base.rglob("*.pkl"):
                name = p.name.lower()
                if dataset.lower() in name:
                    candidates.append(p)
        if candidates:
            # Choose most recent modification time
            best = max(candidates, key=lambda p: p.stat().st_mtime)
            with open(best, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                acc: Dict[str, set] = {}
                for k in obj.keys():
                    if isinstance(k, tuple) and len(k) == 3:
                        qid, a, b = k
                        if isinstance(qid, str) and isinstance(a, str) and isinstance(b, str):
                            acc.setdefault(qid, set()).update([a, b])
                logger.info(f"Loaded rerank matrix from {best} (queries: {len(acc)})")
                return {qid: sorted(list(s)) for qid, s in acc.items()}
        return None
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
) -> RankingDataset:
    """Load a BEIR dataset as a RankingDataset using a rerank matrix.

    - Candidate sets come strictly from the rerank matrix for each query.
    - y_true are aligned with candidate_ids using qrels labels (missing defaults to 0).
    - Raises FileNotFoundError if the rerank matrix is not found; calling CLI skips dataset.

    Parameters
    - dataset: BEIR dataset name (canonical, lowercased).
    - split: BEIR split to load (usually "test").
    - seed: removed; ranker seeds are handled at ranker construction.
    - max_queries: optional limit of queries for a quick run (applied before filtering by matrix).
    """
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

    matrix = _load_rerank_matrix(canonical, split)
    if matrix is None:
        from ireranker.config import EXTERNAL_DATA_DIR

        base_only = EXTERNAL_DATA_DIR / "reranking-matrices"
        logger.info(
            f"No rerank matrix found for {canonical}/{split}. Looked under: {base_only} (recursive *.pkl search)"
        )
        raise FileNotFoundError(
            f"Rerank matrix not found for {canonical}/{split}. Skipping dataset."
        )

    q_ids = sorted(matrix.keys())
    if max_queries is not None:
        q_ids = q_ids[:max_queries]
    logger.info(f"Using rerank matrix for {len(q_ids)} queries in {canonical}/{split}")

    qrels = _read_beir_qrels(Path(data_path), split)

    for qid in q_ids:
        rel_map: Dict[str, int] = qrels.get(qid, {})

        cand_ids: List[str] = matrix[qid]

        y_true = [float(rel_map.get(doc_id, 0)) for doc_id in cand_ids]

        tasks.append(
            RankingTask(
                query_id=qid,
                candidate_ids=cand_ids,
                y_true=y_true,
                dataset_path=str(data_path),
            )
        )

    return RankingDataset(tasks=tasks)

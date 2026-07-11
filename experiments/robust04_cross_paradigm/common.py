from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "robust04_cross_paradigm"
SNAPSHOT_DIR = ROOT / "data" / "external" / "robust04_cross_paradigm"
QUERIES_PATH = SNAPSHOT_DIR / "queries.jsonl"
DOCUMENTS_PATH = SNAPSHOT_DIR / "documents.jsonl"
TOP100_PATH = SNAPSHOT_DIR / "bm25.robust04.top100.txt"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"
PER_QUERY_DIR = EXP_DIR / "results" / "per_query"
RUNS_DIR = EXP_DIR / "results" / "runs"
LOGS_DIR = EXP_DIR / "results" / "logs"
METRICS_DIR = EXP_DIR / "results" / "metrics"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_snapshot() -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing Robust04 snapshot: {MANIFEST_PATH}. Run prepare_snapshot.py first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for filename, expected in manifest.get("files", {}).items():
        path = SNAPSHOT_DIR / filename
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"Snapshot hash mismatch: {path}")
    queries = read_jsonl(QUERIES_PATH)
    documents = {str(row["doc_id"]): str(row["text"]) for row in read_jsonl(DOCUMENTS_PATH)}
    if len(queries) != 249:
        raise ValueError(f"Expected 249 Robust04 queries, found {len(queries)}")
    for row in queries:
        candidates = row.get("candidates", [])
        if len(candidates) != 100:
            raise ValueError(f"Query {row.get('query_id')} has {len(candidates)} candidates")
        missing = [doc_id for doc_id in candidates if doc_id not in documents]
        if missing:
            raise ValueError(f"Query {row.get('query_id')} has missing documents: {missing[:5]}")
    return queries, documents


def ndcg_at_k(ranking: Iterable[str], qrels: dict[str, int], k: int = 10) -> float:
    ranked = list(ranking)[:k]
    # trec_eval/pytrec_eval ndcg_cut uses the raw relevance level as gain by default.
    dcg = sum(
        max(float(qrels.get(doc_id, 0)), 0.0) / math.log2(i + 2)
        for i, doc_id in enumerate(ranked)
    )
    ideal = sorted((max(float(value), 0.0) for value in qrels.values()), reverse=True)[:k]
    idcg = sum(value / math.log2(i + 2) for i, value in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_trec_run(path: Path, rankings: list[tuple[str, list[str]]], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for qid, ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                handle.write(f"{qid} Q0 {doc_id} {rank} {len(ranking) - rank + 1} {tag}\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

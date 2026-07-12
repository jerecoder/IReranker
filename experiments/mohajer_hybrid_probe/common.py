from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "mohajer_hybrid_probe"
SNAPSHOT_ROOT = ROOT / "data" / "external" / "mohajer_hybrid_probe"
RESULTS_DIR = EXP_DIR / "results"
PER_QUERY_DIR = RESULTS_DIR / "per_query"
RUNS_DIR = RESULTS_DIR / "runs"
METRICS_DIR = RESULTS_DIR / "metrics"

# Hybrid arms share one end-to-end budget. Mohajer receives the first 80%; the
# remaining 20% is reserved for the top-20 refinement. This must be fixed before
# looking at results, otherwise a token-limit exception in Mohajer leaves no
# compute for the second stage.
HYBRID_STAGE_A_FRACTION = 0.80

# Pre-registered order: strongest paper/repository signal first.
DATASET_ORDER = [
    "dl-2019",
    "dl-2020",
    "dbpedia-entity",
    "fiqa",
    "nfcorpus",
    "trec-covid",
    "scifact",
    "webis-touche2020",
]

MOHAJER_FAMILY = {
    "mohajer",
    "mohajer_bubble",
    "mohajer_setwise",
    "mohajer_listwise",
}

METHOD_ORDER = [
    "bm25",
    "mohajer",
    "mohajer_bubble",
    "mohajer_setwise",
    "mohajer_listwise",
    "setwise",
    "listwise",
    "bubble",
]


def snapshot_dir(dataset: str) -> Path:
    return SNAPSHOT_ROOT / dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
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
                handle.write(
                    f"{qid} Q0 {doc_id} {rank} {len(ranking) - rank + 1} {tag}\n"
                )
    temporary.replace(path)


def stable_query_order(dataset: str, qids: Iterable[str]) -> list[str]:
    return sorted(
        (str(qid) for qid in qids),
        key=lambda qid: hashlib.sha256(f"{dataset}:{qid}".encode()).hexdigest(),
    )


def ndcg_at_k(ranking: Iterable[str], qrels: dict[str, int], k: int = 10) -> float:
    ranked = list(ranking)[:k]
    dcg = sum(
        max(float(qrels.get(doc_id, 0)), 0.0) / math.log2(index + 2)
        for index, doc_id in enumerate(ranked)
    )
    ideal = sorted((max(float(value), 0.0) for value in qrels.values()), reverse=True)[:k]
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def load_snapshot(dataset: str) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    directory = snapshot_dir(dataset)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"Snapshot hash mismatch: {path}")
    queries = read_jsonl(directory / "queries.jsonl")
    documents = {
        str(row["doc_id"]): str(row["text"])
        for row in read_jsonl(directory / "documents.jsonl")
    }
    if len(queries) != int(manifest["selected_queries"]):
        raise ValueError(f"Snapshot query count mismatch for {dataset}")
    for row in queries:
        candidates = [str(value) for value in row["candidates"]]
        if len(candidates) != 100 or len(set(candidates)) != 100:
            raise ValueError(f"{dataset}/{row['query_id']} is not a unique top-100")
        missing = [doc_id for doc_id in candidates if doc_id not in documents]
        if missing:
            raise ValueError(f"Missing snapshot documents for {dataset}: {missing[:3]}")
    return queries, documents, manifest


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def strong_mohajer_failure(
    bm25: dict[str, float],
    mohajer: dict[str, float],
    mohajer_bubble: dict[str, float],
    *,
    margin: float = 0.03,
) -> bool:
    qids = sorted(set(bm25) & set(mohajer) & set(mohajer_bubble))
    if not qids:
        raise ValueError("Early-stop gate has no paired queries")
    bm_mean = mean(bm25[qid] for qid in qids)
    deltas = {
        "mohajer": mean(mohajer[qid] for qid in qids) - bm_mean,
        "mohajer_bubble": mean(mohajer_bubble[qid] for qid in qids) - bm_mean,
    }
    wins = {
        "mohajer": sum(mohajer[qid] > bm25[qid] for qid in qids),
        "mohajer_bubble": sum(mohajer_bubble[qid] > bm25[qid] for qid in qids),
    }
    return all(deltas[name] <= -margin and wins[name] == 0 for name in deltas)


def quality_gate_methods(
    bm25: dict[str, float],
    methods: dict[str, dict[str, float]],
    *,
    min_delta: float = 0.02,
    min_wins: int = 2,
) -> set[str]:
    passed: set[str] = set()
    for name, values in methods.items():
        qids = sorted(set(bm25) & set(values))
        if not qids:
            continue
        delta = mean(values[qid] - bm25[qid] for qid in qids)
        wins = sum(values[qid] > bm25[qid] for qid in qids)
        if delta >= min_delta and wins >= min_wins:
            passed.add(name)
    return passed


def pareto_methods(rows: list[dict[str, Any]]) -> set[str]:
    frontier: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_more_tokens = float(other["avg_tokens"]) <= float(row["avg_tokens"])
            no_less_quality = float(other["ndcg10"]) >= float(row["ndcg10"])
            strict = (
                float(other["avg_tokens"]) < float(row["avg_tokens"])
                or float(other["ndcg10"]) > float(row["ndcg10"])
            )
            if no_more_tokens and no_less_quality and strict:
                dominated = True
                break
        if not dominated:
            frontier.add(str(row["method"]))
    return frontier

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "reviewer_response"
SNAPSHOT_DIR = ROOT / "data" / "external" / "reviewer_response" / "dbpedia-entity"
RESULTS_DIR = EXP_DIR / "results"
PER_QUERY_DIR = RESULTS_DIR / "per_query"
RUNS_DIR = RESULTS_DIR / "runs"
METRICS_DIR = RESULTS_DIR / "metrics"
STATUS_PATH = RESULTS_DIR / "overnight_status.json"

DATASET = "dbpedia-entity"
QUERY_COUNT = 20
SELECTION_OFFSET = 3
PILOT_QUERY_IDS = ["INEX_XER-65", "QALD2_te-28", "QALD2_tr-3"]
TOKEN_BUDGETS = [100000, 50000]
STOCHASTIC_SEEDS = [42, 43, 44]
DETERMINISTIC_SEED = 42
HYBRID_STAGE_A_FRACTION = 0.80
SNAPSHOT_PROTOCOL_VERSION = 1
PROTOCOL_VERSION = 1

EXPERIMENT_1_METHODS = ["bm25", "prp", "mohajer", "setwise", "listwise"]
EXPERIMENT_2_METHODS = [
    "bm25",
    "mohajer",
    "setwise",
    "listwise",
    "mohajer_setwise",
    "mohajer_listwise",
]
SHARED_METHODS = ["bm25", "mohajer", "setwise", "listwise"]
STOCHASTIC_METHODS = {"prp", "mohajer", "mohajer_setwise", "mohajer_listwise"}
METHOD_VARIANTS = {
    "bm25": "top100",
    "prp": "randomized_direction_prp_heapsort_top10",
    "mohajer": "randomized_direction_prp_mohajer_top10",
    "setwise": "standalone_heapsort_c3_top10",
    "listwise": "standalone_rankgpt_w4_s2_r5",
    "mohajer_setwise": "mohajer_prp_then_top20_setwise_c3",
    "mohajer_listwise": "mohajer_prp_then_top20_rankgpt_w4_s2_r2",
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_trec_run(path: Path, rankings: list[tuple[str, list[str]]], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for qid, ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                handle.write(f"{qid} Q0 {doc_id} {rank} {len(ranking)-rank+1} {tag}\n")
    temporary.replace(path)


def ndcg_at_k(ranking: Iterable[str], qrels: dict[str, int], k: int = 10) -> float:
    ranked = list(ranking)[:k]
    dcg = sum(
        max(float(qrels.get(doc_id, 0)), 0.0) / math.log2(index + 2)
        for index, doc_id in enumerate(ranked)
    )
    ideal = sorted((max(float(value), 0.0) for value in qrels.values()), reverse=True)[:k]
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def condition_name(method: str, budget: int | None, seed: int) -> str:
    return "bm25" if method == "bm25" else f"{method}_t{budget}_s{seed}"


def method_seeds(method: str) -> list[int]:
    return STOCHASTIC_SEEDS if method in STOCHASTIC_METHODS else [DETERMINISTIC_SEED]


def load_snapshot() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    manifest_path = SNAPSHOT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("snapshot_protocol_version", 0)) != SNAPSHOT_PROTOCOL_VERSION:
        raise ValueError("Reviewer-response snapshot protocol mismatch")
    for filename, expected in manifest["files"].items():
        path = SNAPSHOT_DIR / filename
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"Snapshot hash mismatch: {path}")
    queries = read_jsonl(SNAPSHOT_DIR / "queries.jsonl")
    documents = {
        str(row["doc_id"]): str(row["text"])
        for row in read_jsonl(SNAPSHOT_DIR / "documents.jsonl")
    }
    if len(queries) != QUERY_COUNT:
        raise ValueError(f"Expected {QUERY_COUNT} frozen queries, got {len(queries)}")
    qids = [str(row["query_id"]) for row in queries]
    if set(qids) & set(PILOT_QUERY_IDS):
        raise ValueError("Pilot query leaked into the confirmatory snapshot")
    if qids != list(manifest["selected_query_ids"]):
        raise ValueError("Snapshot query order differs from its manifest")
    for row in queries:
        candidates = [str(value) for value in row["candidates"]]
        if len(candidates) != 100 or len(set(candidates)) != 100:
            raise ValueError(f"Invalid top-100 for {row['query_id']}")
        missing = [doc_id for doc_id in candidates if doc_id not in documents]
        if missing:
            raise ValueError(f"Missing documents for {row['query_id']}: {missing[:3]}")
    return queries, documents, manifest


def pareto_methods(rows: list[dict[str, Any]], cost_key: str) -> set[str]:
    frontier: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_more_cost = float(other[cost_key]) <= float(row[cost_key])
            no_less_quality = float(other["ndcg10"]) >= float(row["ndcg10"])
            strict = (
                float(other[cost_key]) < float(row[cost_key])
                or float(other["ndcg10"]) > float(row["ndcg10"])
            )
            if no_more_cost and no_less_quality and strict:
                dominated = True
                break
        if not dominated:
            frontier.add(str(row["method"]))
    return frontier

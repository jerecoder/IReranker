#!/usr/bin/env python3
"""Freeze Robust04 queries, qrels, BM25 top-100 IDs, and candidate text."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.robust04_cross_paradigm.common import (  # noqa: E402
    DOCUMENTS_PATH,
    MANIFEST_PATH,
    QUERIES_PATH,
    SNAPSHOT_DIR,
    TOP100_PATH,
    sha256,
)
from ireranker.data.public_tasks import build_public_task_dataset  # noqa: E402


SOURCE_RUN = ROOT / "data/external/beir/bm25-runs/run.beir.bm25-flat.robust04.txt"
TASK_DIR = ROOT / "data/external/beir/robust04"


def load_queries() -> dict[str, str]:
    rows: dict[str, str] = {}
    with (TASK_DIR / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["_id"])] = str(row.get("text") or "").strip()
    return rows


def load_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with (TASK_DIR / "qrels/test.tsv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0] == "query-id":
                continue
            qrels.setdefault(str(row[0]), {})[str(row[1])] = int(row[2])
    return qrels


def load_top100() -> tuple[dict[str, list[str]], dict[tuple[str, str], float], list[str]]:
    candidates: dict[str, list[str]] = {}
    scores: dict[tuple[str, str], float] = {}
    output_lines: list[str] = []
    with SOURCE_RUN.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed BM25 row: {line!r}")
            qid, doc_id = parts[0], parts[2]
            docs = candidates.setdefault(qid, [])
            if len(docs) >= 100:
                continue
            if int(parts[3]) != len(docs) + 1:
                raise ValueError(f"Non-contiguous BM25 ranks for query {qid}: {line!r}")
            if doc_id in docs:
                raise ValueError(f"Duplicate BM25 document for query {qid}: {doc_id}")
            docs.append(doc_id)
            scores[(qid, doc_id)] = float(parts[4])
            output_lines.append(line if line.endswith("\n") else line + "\n")
    if len(candidates) != 249 or any(len(docs) != 100 for docs in candidates.values()):
        raise ValueError("Robust04 BM25 run must contain exactly 249 queries x top 100")
    return candidates, scores, output_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="beir-v1.0.0-robust04.flat")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if MANIFEST_PATH.exists() and not args.force:
        print(f"Snapshot already exists: {MANIFEST_PATH}")
        return

    # Rebuild public topics/qrels with the corrected multiline Robust04 title parser.
    # Older local layouts may exist but contain 49 empty query titles.
    build_public_task_dataset("robust04", TASK_DIR.parent)

    os.environ.setdefault("OPENAI_API_KEY", "unused-local-flan-t5-run")
    from pyserini.search.lucene import LuceneSearcher  # type: ignore

    queries = load_queries()
    qrels = load_qrels()
    candidates, scores, output_lines = load_top100()
    if set(candidates) != set(qrels):
        raise ValueError(
            f"BM25/qrels qid mismatch: only_bm25={sorted(set(candidates) - set(qrels))[:5]}, "
            f"only_qrels={sorted(set(qrels) - set(candidates))[:5]}"
        )
    qids = sorted(candidates, key=lambda value: int(value) if value.isdigit() else value)
    missing_queries = [qid for qid in qids if not queries.get(qid)]
    if missing_queries:
        raise ValueError(f"Missing Robust04 query text: {missing_queries[:5]}")

    searcher = LuceneSearcher.from_prebuilt_index(args.index)
    unique_doc_ids = list(dict.fromkeys(doc_id for qid in qids for doc_id in candidates[qid]))
    if len(unique_doc_ids) != 20420:
        raise ValueError(f"Expected 20,420 unique Robust04 top-100 documents, got {len(unique_doc_ids)}")
    documents: dict[str, str] = {}
    for position, doc_id in enumerate(unique_doc_ids, start=1):
        lucene_doc = searcher.doc(doc_id)
        if lucene_doc is None:
            raise FileNotFoundError(f"Document {doc_id} missing from {args.index}")
        raw = json.loads(lucene_doc.raw())
        raw_id = str(raw.get("_id") or raw.get("id") or "").strip()
        if raw_id and raw_id != doc_id:
            raise ValueError(f"Lucene document mismatch: requested {doc_id}, raw id is {raw_id}")
        title = str(raw.get("title") or "").strip()
        text = str(raw.get("text") or raw.get("contents") or "").strip()
        combined = f"{title} {text}".strip()
        if not combined:
            raise ValueError(f"Document {doc_id} has no text")
        documents[doc_id] = combined
        if position % 1000 == 0:
            print(f"Loaded {position}/{len(unique_doc_ids)} unique documents")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    TOP100_PATH.write_text("".join(output_lines), encoding="utf-8")
    with DOCUMENTS_PATH.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(documents):
            handle.write(json.dumps({"doc_id": doc_id, "text": documents[doc_id]}, ensure_ascii=False) + "\n")
    with QUERIES_PATH.open("w", encoding="utf-8") as handle:
        for qid in qids:
            handle.write(json.dumps({
                "query_id": qid,
                "query": queries[qid],
                "candidates": candidates[qid],
                "bm25_scores": [scores[(qid, doc_id)] for doc_id in candidates[qid]],
                "qrels": qrels.get(qid, {}),
            }, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "robust04",
        "queries": len(qids),
        "candidates_per_query": 100,
        "unique_documents": len(documents),
        "qrels_judgments": sum(len(values) for values in qrels.values()),
        "excluded_standard_topic": "672",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pyserini_version": importlib.metadata.version("pyserini"),
        "document_renderer": "strip(title) + single-space + strip(text)",
        "topics_url": "https://trec.nist.gov/data/robust/04.testset.gz",
        "qrels_url": "https://trec.nist.gov/data/robust/qrels.robust2004.txt",
        "source_bm25_run": str(SOURCE_RUN.relative_to(ROOT)),
        "pyserini_index": args.index,
        "files": {
            QUERIES_PATH.name: sha256(QUERIES_PATH),
            DOCUMENTS_PATH.name: sha256(DOCUMENTS_PATH),
            TOP100_PATH.name: sha256(TOP100_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved snapshot: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

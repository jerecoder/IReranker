#!/usr/bin/env python3
"""Freeze public pilot queries, full qrels, BM25 top-100 IDs, and document text."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.common import (  # noqa: E402
    DATASET_ORDER,
    ROOT,
    SNAPSHOT_PROTOCOL_VERSION,
    eligible_query_ids,
    sha256,
    snapshot_dir,
    stable_query_order,
    write_jsonl,
)
from ireranker.data.loaders import _download_beir_once  # noqa: E402


BEIR_ROOT = ROOT / "data/external/beir"
RUN_ROOT = BEIR_ROOT / "bm25-runs"
PUBLIC_TASKS = {"dl-2019", "dl-2020"}
MSMARCO_INDEX = "msmarco-v1-passage"


def ensure_dataset(dataset: str) -> Path:
    return Path(_download_beir_once(dataset, BEIR_ROOT))


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with (path / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = str(row.get("text") or row.get("query") or "").strip()
            if text:
                queries[str(row["_id"])] = text
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels/test.tsv").open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0] == "query-id":
                continue
            qrels.setdefault(str(row[0]), {})[str(row[1])] = int(row[2])
    return qrels


def load_bm25(dataset: str) -> tuple[dict[str, list[str]], dict[tuple[str, str], float]]:
    path = RUN_ROOT / f"run.beir.bm25-flat.{dataset}.txt"
    candidates: dict[str, list[str]] = {}
    scores: dict[tuple[str, str], float] = {}
    row_counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed BM25 row in {path}: {line!r}")
            qid, doc_id = parts[0], parts[2]
            count = row_counts.get(qid, 0)
            if count >= 100:
                continue
            row_counts[qid] = count + 1
            docs = candidates.setdefault(qid, [])
            docs.append(doc_id)
            scores[(qid, doc_id)] = float(parts[4])
    return candidates, scores


def render_document(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or row.get("contents") or "").strip()
    return f"{title} {text}".strip()


def load_beir_documents(dataset_path: Path, wanted: set[str]) -> dict[str, str]:
    documents: dict[str, str] = {}
    with (dataset_path / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = str(row["_id"])
            if doc_id not in wanted:
                continue
            text = render_document(row)
            if not text:
                raise ValueError(f"Empty BEIR document: {doc_id}")
            documents[doc_id] = text
            if len(documents) == len(wanted):
                break
    return documents


def load_msmarco_documents(wanted: set[str]) -> dict[str, str]:
    os.environ.setdefault("OPENAI_API_KEY", "unused-local-flan-t5-run")
    from pyserini.search.lucene import LuceneSearcher  # type: ignore

    searcher = LuceneSearcher.from_prebuilt_index(MSMARCO_INDEX)
    documents: dict[str, str] = {}
    for index, doc_id in enumerate(sorted(wanted), start=1):
        lucene_doc = searcher.doc(doc_id)
        if lucene_doc is None:
            raise FileNotFoundError(f"MS MARCO document missing from index: {doc_id}")
        raw_text = lucene_doc.raw()
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            raw = {"contents": raw_text}
        text = render_document(raw)
        if not text:
            raise ValueError(f"Empty MS MARCO document: {doc_id}")
        documents[doc_id] = text
        if index % 100 == 0:
            print(f"MS MARCO documents: {index}/{len(wanted)}")
    return documents


def prepare_dataset(dataset: str, *, count: int, offset: int, force: bool) -> None:
    output = snapshot_dir(dataset)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (
            int(manifest.get("snapshot_protocol_version", 0)),
            int(manifest["selected_queries"]),
            int(manifest["selection_offset"]),
        )
        requested = (SNAPSHOT_PROTOCOL_VERSION, count, offset)
        if expected[0] != SNAPSHOT_PROTOCOL_VERSION:
            print(
                f"REBUILD snapshot: {dataset} protocol {expected[0]} -> "
                f"{SNAPSHOT_PROTOCOL_VERSION}"
            )
        elif expected != requested:
            raise ValueError(
                f"Existing {dataset} snapshot uses protocol/count/offset={expected}; "
                "pass --force to replace it"
            )
        else:
            print(f"SKIP snapshot: {dataset}")
            return

    dataset_path = ensure_dataset(dataset)
    queries = load_queries(dataset_path)
    qrels = load_qrels(dataset_path)
    candidates, scores = load_bm25(dataset)
    shared = set(queries) & set(qrels) & set(candidates)
    eligible = eligible_query_ids(queries, qrels, candidates)
    ordered = stable_query_order(dataset, eligible)
    selected = ordered[offset : offset + count]
    if len(selected) != count:
        raise ValueError(f"{dataset} only has {len(selected)} selectable queries")
    for qid in selected:
        if len(candidates[qid]) != 100 or len(set(candidates[qid])) != 100:
            raise ValueError(f"{dataset}/{qid} does not have a unique BM25 top-100")

    wanted = {doc_id for qid in selected for doc_id in candidates[qid]}
    documents = (
        load_msmarco_documents(wanted)
        if dataset in PUBLIC_TASKS
        else load_beir_documents(dataset_path, wanted)
    )
    missing = wanted - set(documents)
    if missing:
        raise FileNotFoundError(f"Missing {dataset} documents: {sorted(missing)[:5]}")

    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "queries.jsonl"
    document_path = output / "documents.jsonl"
    write_jsonl(
        query_path,
        (
            {
                "dataset": dataset,
                "query_id": qid,
                "query": queries[qid],
                "candidates": candidates[qid],
                "bm25_scores": [scores[(qid, doc_id)] for doc_id in candidates[qid]],
                "qrels": qrels[qid],
            }
            for qid in selected
        ),
    )
    write_jsonl(
        document_path,
        ({"doc_id": doc_id, "text": documents[doc_id]} for doc_id in sorted(documents)),
    )
    manifest = {
        "snapshot_protocol_version": SNAPSHOT_PROTOCOL_VERSION,
        "dataset": dataset,
        "selected_queries": count,
        "selection_offset": offset,
        "selection": (
            "ascending sha256(dataset + ':' + query_id), restricted to queries "
            "with text, qrels, and exactly 100 unique BM25 candidates"
        ),
        "shared_queries_before_candidate_filter": len(shared),
        "eligible_queries": len(eligible),
        "queries_excluded_for_short_or_duplicate_bm25_run": len(shared - eligible),
        "selected_query_ids": selected,
        "candidates_per_query": 100,
        "unique_documents": len(documents),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_path": str(dataset_path.relative_to(ROOT)),
        "source_bm25_run": str(
            (RUN_ROOT / f"run.beir.bm25-flat.{dataset}.txt").relative_to(ROOT)
        ),
        "pyserini_version": (
            importlib.metadata.version("pyserini") if dataset in PUBLIC_TASKS else None
        ),
        "document_renderer": "strip(title) + single-space + strip(text or contents)",
        "files": {
            query_path.name: sha256(query_path),
            document_path.name: sha256(document_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {dataset}: {count} queries, {len(documents)} documents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    parser.add_argument("--queries-per-dataset", type=int, default=3)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.queries_per_dataset <= 0 or args.selection_offset < 0:
        parser.error("query count must be positive and offset non-negative")
    for dataset in args.datasets:
        prepare_dataset(
            dataset,
            count=args.queries_per_dataset,
            offset=args.selection_offset,
            force=args.force,
        )


if __name__ == "__main__":
    main()

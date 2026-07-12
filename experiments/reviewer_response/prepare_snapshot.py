#!/usr/bin/env python3
"""Freeze 20 unseen DBpedia queries after the three pre-registered pilot queries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.common import (  # noqa: E402
    eligible_query_ids,
    stable_query_order,
)
from experiments.mohajer_hybrid_probe.prepare_snapshots import (  # noqa: E402
    BEIR_ROOT,
    RUN_ROOT,
    ensure_dataset,
    load_beir_documents,
    load_bm25,
    load_qrels,
    load_queries,
)
from experiments.reviewer_response.common import (  # noqa: E402
    DATASET,
    PILOT_QUERY_IDS,
    QUERY_COUNT,
    ROOT,
    SELECTION_OFFSET,
    SNAPSHOT_DIR,
    SNAPSHOT_PROTOCOL_VERSION,
    sha256,
    write_json,
    write_jsonl,
)


def main() -> None:
    manifest_path = SNAPSHOT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (
            SNAPSHOT_PROTOCOL_VERSION,
            QUERY_COUNT,
            SELECTION_OFFSET,
            PILOT_QUERY_IDS,
        )
        actual = (
            int(manifest.get("snapshot_protocol_version", 0)),
            int(manifest.get("selected_queries", 0)),
            int(manifest.get("selection_offset", -1)),
            list(manifest.get("excluded_pilot_query_ids", [])),
        )
        files_valid = all(
            (SNAPSHOT_DIR / filename).exists()
            and sha256(SNAPSHOT_DIR / filename) == digest
            for filename, digest in manifest.get("files", {}).items()
        )
        if actual == expected and files_valid:
            print(f"SKIP verified reviewer-response snapshot: {manifest_path}")
            return
        raise RuntimeError(
            f"Existing reviewer-response snapshot is incompatible: {actual}; "
            f"expected {expected}. Move it aside instead of mixing protocols."
        )

    dataset_path = ensure_dataset(DATASET)
    queries = load_queries(dataset_path)
    qrels = load_qrels(dataset_path)
    candidates, scores = load_bm25(DATASET)
    eligible = eligible_query_ids(queries, qrels, candidates)
    ordered = stable_query_order(DATASET, eligible)
    observed_pilot = ordered[:SELECTION_OFFSET]
    if observed_pilot != PILOT_QUERY_IDS:
        raise RuntimeError(
            "Pilot exclusion mismatch. Refusing to select confirmatory queries: "
            f"expected {PILOT_QUERY_IDS}, observed {observed_pilot}"
        )
    selected = ordered[SELECTION_OFFSET : SELECTION_OFFSET + QUERY_COUNT]
    if len(selected) != QUERY_COUNT:
        raise RuntimeError(f"Only {len(selected)} unseen eligible queries are available")
    if set(selected) & set(PILOT_QUERY_IDS):
        raise RuntimeError("Pilot query leaked into the confirmatory selection")

    wanted = {doc_id for qid in selected for doc_id in candidates[qid]}
    documents = load_beir_documents(dataset_path, wanted)
    missing = wanted - set(documents)
    if missing:
        raise FileNotFoundError(f"Missing DBpedia documents: {sorted(missing)[:5]}")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    query_path = SNAPSHOT_DIR / "queries.jsonl"
    document_path = SNAPSHOT_DIR / "documents.jsonl"
    write_jsonl(
        query_path,
        (
            {
                "dataset": DATASET,
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
        "dataset": DATASET,
        "selected_queries": QUERY_COUNT,
        "selection_offset": SELECTION_OFFSET,
        "excluded_pilot_query_ids": PILOT_QUERY_IDS,
        "selected_query_ids": selected,
        "selection": (
            "ascending sha256(dataset + ':' + query_id) among queries with text, qrels, "
            "and exactly 100 unique BM25 candidates; exclude the first three pilot queries"
        ),
        "eligible_queries": len(eligible),
        "candidates_per_query": 100,
        "unique_documents": len(documents),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_path": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
        "source_bm25_run": str(
            (RUN_ROOT / f"run.beir.bm25-flat.{DATASET}.txt").relative_to(ROOT)
        ).replace("\\", "/"),
        "beir_root": str(BEIR_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "files": {
            query_path.name: sha256(query_path),
            document_path.name: sha256(document_path),
        },
    }
    write_json(manifest_path, manifest)
    print(f"Saved {manifest_path}: {QUERY_COUNT} unseen queries, {len(documents)} documents")


if __name__ == "__main__":
    main()

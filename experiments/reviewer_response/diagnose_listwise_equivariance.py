#!/usr/bin/env python3
"""Test Listwise formatting and ranking equivariance under controlled relabelings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.engine import (  # noqa: E402
    SharedFlanT5Engine,
    UsageMeter,
)
from experiments.reviewer_response.common import (  # noqa: E402
    EXP_DIR,
    RESULTS_DIR,
    SNAPSHOT_DIR,
    load_snapshot,
    sha256,
    write_json,
    write_jsonl,
)
from experiments.reviewer_response.listwise_validity import (  # noqa: E402
    map_labels_to_documents,
    parse_strict_permutation,
)


OUTPUT_DIR = RESULTS_DIR / "listwise_equivariance_diagnostic"
WINDOW_STARTS = [0, 48, 96]
DOCUMENT_SHUFFLE = [2, 0, 3, 1]
LABEL_SHUFFLE = [3, 1, 4, 2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_symbolic(
    query: str,
    labeled_passages: Sequence[tuple[int, str]],
) -> str:
    body = "\n\n".join(
        f"Passage [{label}]: {passage.strip()}" for label, passage in labeled_passages
    )
    return (
        f"Query: {query}\n\n{body}\n\n"
        "Order all passages from most relevant to least relevant. Use every passage "
        "identifier exactly once. Output only four bracketed identifiers separated by >. "
        "Do not add an explanation or copy the input order unless it is the relevance order.\n"
        "Ranking:"
    )


def render_minimal(
    query: str,
    labeled_passages: Sequence[tuple[int, str]],
) -> str:
    body = "\n".join(f"[{label}] {passage.strip()}" for label, passage in labeled_passages)
    return (
        f"Rank these passages for the query: {query}\n{body}\n"
        "Return a complete relevance permutation using each bracketed passage number once, "
        "with > between numbers and no other text.\nAnswer:"
    )


def render_rank_then_emit(
    query: str,
    labeled_passages: Sequence[tuple[int, str]],
) -> str:
    body = "\n\n".join(
        f"Document [{label}]\n{passage.strip()}" for label, passage in labeled_passages
    )
    return (
        f"Search query: {query}\n\n{body}\n\n"
        "Silently determine the relevance order of all four documents. In the final answer, "
        "write only their bracketed identifiers from best to worst, separated by >. Include "
        "all four identifiers exactly once.\nFinal ranking:"
    )


def presentations(
    document_ids: list[str], passages: list[str]
) -> list[tuple[str, list[tuple[int, str, str]]]]:
    identity = [
        (index + 1, document_ids[index], passages[index]) for index in range(4)
    ]
    document_shuffle = [
        (position + 1, document_ids[index], passages[index])
        for position, index in enumerate(DOCUMENT_SHUFFLE)
    ]
    label_shuffle = [
        (LABEL_SHUFFLE[index], document_ids[index], passages[index]) for index in range(4)
    ]
    return [
        ("identity", identity),
        ("document_shuffle", document_shuffle),
        ("label_shuffle", label_shuffle),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--passage-tokens", type=int, default=100)
    parser.add_argument("--encoder-max-tokens", type=int, default=768)
    parser.add_argument("--queries", type=int, default=3)
    args = parser.parse_args()
    if args.queries <= 0:
        parser.error("--queries must be positive")

    queries, documents, _ = load_snapshot()
    selected = queries[: min(args.queries, len(queries))]
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=args.query_tokens,
        passage_tokens=args.passage_tokens,
        encoder_max_tokens=args.encoder_max_tokens,
    )
    variants: list[
        tuple[str, Callable[[str, Sequence[tuple[int, str]]], str]]
    ] = [
        ("symbolic", render_symbolic),
        ("minimal", render_minimal),
        ("rank_then_emit", render_rank_then_emit),
    ]

    records: list[dict[str, Any]] = []
    for row in selected:
        qid = str(row["query_id"])
        query = engine.truncate_query(str(row["query"]))
        candidate_ids = [str(value) for value in row["candidates"]]
        for start in WINDOW_STARTS:
            ids = candidate_ids[start : start + 4]
            if len(ids) != 4:
                raise RuntimeError(f"Incomplete window: {qid}/{start}")
            texts = [engine.truncate_passage(documents[doc_id]) for doc_id in ids]
            case_id = f"{qid}:{start}"
            for presentation, labeled in presentations(ids, texts):
                label_to_document = {label: doc_id for label, doc_id, _ in labeled}
                labeled_passages = [(label, passage) for label, _, passage in labeled]
                for variant, renderer in variants:
                    prompt = renderer(query, labeled_passages)
                    meter = UsageMeter()
                    output = engine.generate(
                        [prompt],
                        meter=meter,
                        max_new_tokens=32,
                        decoder_prefix=False,
                        document_counts=[4],
                    )[0]
                    parsed = parse_strict_permutation(output, 4)
                    mapped = (
                        list(map_labels_to_documents(parsed.labels, label_to_document))
                        if parsed.valid
                        else None
                    )
                    presented_document_order = [doc_id for _, doc_id, _ in labeled]
                    record = {
                        "case_id": case_id,
                        "variant": variant,
                        "presentation": presentation,
                        "query_id": qid,
                        "window_start": start,
                        "presented_labels": [label for label, _, _ in labeled],
                        "presented_document_order": presented_document_order,
                        "label_to_document": {
                            str(label): doc_id for label, doc_id in label_to_document.items()
                        },
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt_tokens": meter.encoder_nonpad_tokens,
                        "decoder_tokens": meter.decoder_tokens,
                        "raw_output": output,
                        "raw_output_repr": repr(output),
                        "strict_valid": parsed.valid,
                        "strict_labels": list(parsed.labels),
                        "strict_failure_reason": parsed.reason,
                        "mapped_document_ranking": mapped,
                        "copies_presented_order": mapped == presented_document_order,
                    }
                    records.append(record)
                    print(
                        f"{case_id} {variant}/{presentation} valid={parsed.valid} "
                        f"labels={list(parsed.labels)} mapped={mapped} output={output!r}"
                    )

    summaries: dict[str, Any] = {}
    case_ids = sorted({str(row["case_id"]) for row in records})
    for variant, _ in variants:
        subset = [row for row in records if row["variant"] == variant]
        valid = [row for row in subset if row["strict_valid"]]
        paired_cases = 0
        exact_document_agreements = {"document_shuffle": 0, "label_shuffle": 0}
        top1_agreements = {"document_shuffle": 0, "label_shuffle": 0}
        for case_id in case_ids:
            case = {
                str(row["presentation"]): row
                for row in subset
                if row["case_id"] == case_id
            }
            required_presentations = {"identity", "document_shuffle", "label_shuffle"}
            if set(case) != required_presentations:
                raise RuntimeError(f"Incomplete diagnostic case: {variant}/{case_id}")
            if not all(case[name]["strict_valid"] for name in required_presentations):
                continue
            paired_cases += 1
            identity = case["identity"]["mapped_document_ranking"]
            for perturbation in ("document_shuffle", "label_shuffle"):
                ranking = case[perturbation]["mapped_document_ranking"]
                exact_document_agreements[perturbation] += ranking == identity
                top1_agreements[perturbation] += ranking[0] == identity[0]
        total = len(subset)
        copy_count = sum(bool(row["copies_presented_order"]) for row in valid)
        summaries[variant] = {
            "outputs": total,
            "strict_valid": len(valid),
            "strict_valid_rate": len(valid) / total,
            "valid_complete_case_triplets": paired_cases,
            "copies_presented_order": copy_count,
            "copies_presented_order_rate_among_valid": copy_count / len(valid) if valid else None,
            "exact_document_agreement": exact_document_agreements,
            "top1_document_agreement": top1_agreements,
            "top1_document_agreement_rate": {
                name: value / paired_cases if paired_cases else None
                for name, value in top1_agreements.items()
            },
            "accepted": (
                len(valid) == total
                and paired_cases == len(case_ids)
                and all(value / paired_cases >= 8 / 9 for value in top1_agreements.values())
            )
            if paired_cases
            else False,
        }

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "status": "complete",
        "created_utc": utc_now(),
        "diagnostic_only_no_ndcg": True,
        "model": args.model,
        "model_revision": args.model_revision,
        "git_commit": git_commit,
        "snapshot_manifest_sha256": sha256(SNAPSHOT_DIR / "manifest.json"),
        "selected_query_ids": [str(row["query_id"]) for row in selected],
        "window_starts": WINDOW_STARTS,
        "document_shuffle": DOCUMENT_SHUFFLE,
        "label_shuffle": LABEL_SHUFFLE,
        "acceptance_rule": (
            "100% strict validity and top-1 mapped-document agreement on at least 8/9 "
            "cases under both document-order and label reassignment perturbations"
        ),
        "variants": summaries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "raw_outputs.jsonl", records)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "raw_outputs_sha256": sha256(OUTPUT_DIR / "raw_outputs.jsonl"),
            "summary_sha256": sha256(OUTPUT_DIR / "summary.json"),
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in [Path(__file__), EXP_DIR / "listwise_validity.py"]
            },
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

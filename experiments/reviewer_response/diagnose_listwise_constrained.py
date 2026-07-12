#!/usr/bin/env python3
"""Validate constrained Listwise decoding under order and label perturbations."""

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
from experiments.reviewer_response.constrained_listwise import (  # noqa: E402
    ConstrainedPermutationDecoder,
    render_constrained_listwise,
)
from experiments.reviewer_response.diagnose_listwise_equivariance import (  # noqa: E402
    DOCUMENT_SHUFFLE,
    LABEL_SHUFFLE,
    WINDOW_STARTS,
    presentations,
    render_minimal,
    render_rank_then_emit,
    render_symbolic,
)
from experiments.reviewer_response.listwise_validity import (  # noqa: E402
    map_labels_to_documents,
    parse_strict_permutation,
)


OUTPUT_DIR = RESULTS_DIR / "listwise_constrained_diagnostic"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
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
        query_tokens=32,
        passage_tokens=100,
        encoder_max_tokens=768,
    )
    decoder = ConstrainedPermutationDecoder(engine)
    variants: list[
        tuple[str, Callable[[str, Sequence[tuple[int, str]]], str]]
    ] = [
        ("constrained_minimal", render_minimal),
        ("constrained_symbolic", render_symbolic),
        ("constrained_rank_then_emit", render_rank_then_emit),
        ("constrained_direct", render_constrained_listwise),
    ]

    records: list[dict[str, Any]] = []
    for row in selected:
        query_id = str(row["query_id"])
        query = engine.truncate_query(str(row["query"]))
        candidate_ids = [str(value) for value in row["candidates"]]
        for start in WINDOW_STARTS:
            document_ids = candidate_ids[start : start + 4]
            texts = [engine.truncate_passage(documents[doc_id]) for doc_id in document_ids]
            case_id = f"{query_id}:{start}"
            for presentation, labeled in presentations(document_ids, texts):
                label_to_document = {label: doc_id for label, doc_id, _ in labeled}
                labeled_passages = [(label, passage) for label, _, passage in labeled]
                for variant, renderer in variants:
                    prompt = renderer(query, labeled_passages)
                    meter = UsageMeter()
                    output = decoder.decode(
                        prompt,
                        labels=sorted(label_to_document),
                        meter=meter,
                        document_count=4,
                    )
                    parsed = parse_strict_permutation(output, 4)
                    mapped = (
                        list(map_labels_to_documents(parsed.labels, label_to_document))
                        if parsed.valid
                        else None
                    )
                    presented_order = [doc_id for _, doc_id, _ in labeled]
                    records.append(
                        {
                            "case_id": case_id,
                            "variant": variant,
                            "presentation": presentation,
                            "query_id": query_id,
                            "window_start": start,
                            "presented_labels": [label for label, _, _ in labeled],
                            "presented_document_order": presented_order,
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
                            "copies_presented_order": mapped == presented_order,
                        }
                    )
                    print(
                        f"{case_id} {variant}/{presentation} valid={parsed.valid} "
                        f"labels={list(parsed.labels)} mapped={mapped} output={output!r}"
                    )

    case_ids = sorted({str(row["case_id"]) for row in records})
    summaries: dict[str, Any] = {}
    for variant, _ in variants:
        subset = [row for row in records if row["variant"] == variant]
        valid = [row for row in subset if row["strict_valid"]]
        paired_cases = 0
        exact = {"document_shuffle": 0, "label_shuffle": 0}
        top1 = {"document_shuffle": 0, "label_shuffle": 0}
        for case_id in case_ids:
            case = {
                str(row["presentation"]): row
                for row in subset
                if row["case_id"] == case_id
            }
            required = {"identity", "document_shuffle", "label_shuffle"}
            if set(case) != required:
                raise RuntimeError(f"Incomplete case {variant}/{case_id}")
            if not all(case[name]["strict_valid"] for name in required):
                continue
            paired_cases += 1
            identity = case["identity"]["mapped_document_ranking"]
            for perturbation in ("document_shuffle", "label_shuffle"):
                ranking = case[perturbation]["mapped_document_ranking"]
                exact[perturbation] += ranking == identity
                top1[perturbation] += ranking[0] == identity[0]
        total = len(subset)
        summaries[variant] = {
            "outputs": total,
            "strict_valid": len(valid),
            "strict_valid_rate": len(valid) / total,
            "valid_complete_case_triplets": paired_cases,
            "copies_presented_order": sum(
                bool(row["copies_presented_order"]) for row in valid
            ),
            "exact_document_agreement": exact,
            "top1_document_agreement": top1,
            "top1_document_agreement_rate": {
                name: value / paired_cases if paired_cases else None
                for name, value in top1.items()
            },
            "accepted": (
                len(valid) == total
                and paired_cases == len(case_ids)
                and all(value / paired_cases >= 8 / 9 for value in top1.values())
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
        "decoder": "greedy_prefix_constrained_complete_permutation",
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
        "accepted_variants": [
            name for name, values in summaries.items() if values["accepted"]
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "raw_outputs.jsonl", records)
    write_json(OUTPUT_DIR / "summary.json", summary)
    source_paths = [
        Path(__file__),
        EXP_DIR / "constrained_listwise.py",
        EXP_DIR / "diagnose_listwise_equivariance.py",
        EXP_DIR / "listwise_validity.py",
    ]
    write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "raw_outputs_sha256": sha256(OUTPUT_DIR / "raw_outputs.jsonl"),
            "summary_sha256": sha256(OUTPUT_DIR / "summary.json"),
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in source_paths
            },
        },
    )
    print(json.dumps(summary, indent=2))
    if not summary["accepted_variants"]:
        raise RuntimeError("No constrained Listwise prompt passed the equivariance gate")
    print(f"Saved {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

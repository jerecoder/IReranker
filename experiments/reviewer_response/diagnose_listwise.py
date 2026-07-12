#!/usr/bin/env python3
"""Record exact FLAN-T5 Listwise outputs without consulting relevance labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.engine import (  # noqa: E402
    SharedFlanT5Engine,
    UsageMeter,
)
from experiments.robust04_cross_paradigm.methods import render_listwise  # noqa: E402
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
    compact_listwise_prompt,
    legacy_repaired_permutation,
    parse_strict_permutation,
)


OUTPUT_DIR = RESULTS_DIR / "listwise_diagnostic"
WINDOW_STARTS = [0, 48, 96]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    queries, documents, manifest = load_snapshot()
    selected = queries[: min(args.queries, len(queries))]
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=args.query_tokens,
        passage_tokens=args.passage_tokens,
        encoder_max_tokens=args.encoder_max_tokens,
    )
    variants: list[tuple[str, Callable[[str, list[str]], str], int]] = [
        ("standard_prompt_max20", render_listwise, 20),
        ("standard_prompt_max64", render_listwise, 64),
        ("compact_prompt_max32", compact_listwise_prompt, 32),
    ]

    records: list[dict[str, Any]] = []
    for row in selected:
        qid = str(row["query_id"])
        query = engine.truncate_query(str(row["query"]))
        candidate_ids = [str(value) for value in row["candidates"]]
        for start in WINDOW_STARTS:
            ids = candidate_ids[start : start + 4]
            if len(ids) != 4:
                raise RuntimeError(f"Diagnostic window is incomplete: {qid}/{start}")
            passages = [engine.truncate_passage(documents[doc_id]) for doc_id in ids]
            for variant, renderer, max_new_tokens in variants:
                prompt = renderer(query, passages)
                meter = UsageMeter()
                output = engine.generate(
                    [prompt],
                    meter=meter,
                    max_new_tokens=max_new_tokens,
                    decoder_prefix=False,
                    document_counts=[4],
                )[0]
                parsed = parse_strict_permutation(output, 4)
                legacy = legacy_repaired_permutation(ids, output)
                record = {
                    "variant": variant,
                    "query_id": qid,
                    "window_start": start,
                    "candidate_ids": ids,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": meter.encoder_nonpad_tokens,
                    "max_new_tokens": max_new_tokens,
                    "decoder_tokens": meter.decoder_tokens,
                    "raw_output": output,
                    "raw_output_repr": repr(output),
                    "strict_valid": parsed.valid,
                    "strict_labels": list(parsed.labels),
                    "strict_failure_reason": parsed.reason,
                    "legacy_repaired_ranking": legacy,
                }
                records.append(record)
                print(
                    f"{variant} qid={qid} start={start} valid={parsed.valid} "
                    f"labels={list(parsed.labels)} output={output!r}"
                )

    summary_variants: dict[str, Any] = {}
    for variant, _, _ in variants:
        subset = [row for row in records if row["variant"] == variant]
        summary_variants[variant] = {
            "outputs": len(subset),
            "strict_valid": sum(bool(row["strict_valid"]) for row in subset),
            "strict_valid_rate": (
                sum(bool(row["strict_valid"]) for row in subset) / len(subset)
            ),
            "failure_reasons": {
                reason: sum(row["strict_failure_reason"] == reason for row in subset)
                for reason in sorted({str(row["strict_failure_reason"]) for row in subset})
            },
            "unique_raw_outputs": sorted({str(row["raw_output"]) for row in subset}),
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
        "device": args.device,
        "git_commit": git_commit,
        "model_output_cache": False,
        "snapshot_manifest_sha256": sha256(SNAPSHOT_DIR / "manifest.json"),
        "selected_query_ids": [str(row["query_id"]) for row in selected],
        "window_starts": WINDOW_STARTS,
        "variants": summary_variants,
        "decision_rule": (
            "Do not rerun Listwise evaluation until one fixed protocol achieves 100% strict "
            "validity on this diagnostic and a separate preflight sample."
        ),
        "snapshot_selected_queries": manifest["selected_query_ids"],
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
                for path in [
                    Path(__file__),
                    EXP_DIR / "listwise_validity.py",
                    ROOT / "experiments/robust04_cross_paradigm/methods.py",
                    ROOT / "experiments/mohajer_hybrid_probe/engine.py",
                ]
            },
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

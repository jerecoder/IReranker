#!/usr/bin/env python3
"""Fail fast before the ordered multi-dataset screen spends real compute."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.common import (  # noqa: E402
    DATASET_ORDER,
    EXP_DIR,
    HYBRID_STAGE_A_FRACTION,
    load_snapshot,
    sha256,
    snapshot_dir,
)
from experiments.mohajer_hybrid_probe.engine import SharedFlanT5Engine  # noqa: E402
from experiments.mohajer_hybrid_probe.methods import (  # noqa: E402
    MethodResult,
    run_bubble,
    run_mohajer,
    run_mohajer_bubble,
    run_mohajer_hybrid,
    run_standalone,
)
from experiments.robust04_cross_paradigm.methods import (  # noqa: E402
    render_listwise,
    render_setwise,
)


def validate_result(name: str, result: MethodResult, candidates: list[str]) -> None:
    if len(result.ranking) != len(candidates) or set(result.ranking) != set(candidates):
        raise RuntimeError(f"{name} smoke test did not return a full permutation")
    if result.meter.total_model_tokens < 0:
        raise RuntimeError(f"{name} reported a negative token count")
    if result.stage_a_tokens + result.stage_b_tokens != result.meter.total_model_tokens:
        raise RuntimeError(f"{name} stage token accounting is inconsistent")


def run_smoke(
    engine: SharedFlanT5Engine,
    row: dict[str, Any],
    documents: dict[str, str],
    budget: int,
) -> dict[str, Any]:
    smoke_row = dict(row)
    smoke_row["candidates"] = [str(value) for value in row["candidates"][:6]]
    kwargs = {
        "row": smoke_row,
        "documents": documents,
        "engine": engine,
        "seed": 42,
        "token_budget": budget,
    }
    outputs = {
        "mohajer": run_mohajer(**kwargs),
        "mohajer_bubble": run_mohajer_bubble(**kwargs),
        "setwise": run_standalone("setwise", **kwargs),
        "mohajer_setwise": run_mohajer_hybrid("setwise", **kwargs),
        "listwise": run_standalone("listwise", **kwargs),
        "mohajer_listwise": run_mohajer_hybrid("listwise", **kwargs),
        "bubble": run_bubble(**kwargs),
    }
    candidates = smoke_row["candidates"]
    for name, result in outputs.items():
        validate_result(name, result, candidates)
    return {
        name: {
            "ranking": result.ranking,
            "tokens": result.meter.total_model_tokens,
            "stage_a_tokens": result.stage_a_tokens,
            "stage_b_tokens": result.stage_b_tokens,
            "invalid_outputs": result.meter.invalid_outputs,
            "inconsistent_outputs": result.meter.inconsistent_outputs,
        }
        for name, result in outputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--passage-tokens", type=int, default=100)
    parser.add_argument("--encoder-max-tokens", type=int, default=768)
    parser.add_argument("--smoke-budget", type=int, default=20000)
    args = parser.parse_args()

    if min(
        args.query_tokens,
        args.passage_tokens,
        args.encoder_max_tokens,
        args.smoke_budget,
    ) <= 0:
        parser.error("token limits and smoke budget must be positive")

    import torch

    device_type = torch.device(args.device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 15:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; require at least 15 GiB")

    snapshots = {dataset: load_snapshot(dataset) for dataset in args.datasets}
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=args.query_tokens,
        passage_tokens=args.passage_tokens,
        encoder_max_tokens=args.encoder_max_tokens,
    )

    max_lengths = {"pairwise": 0, "setwise": 0, "listwise": 0}
    max_examples: dict[str, str] = {}
    for dataset, (queries, documents, _) in snapshots.items():
        for row in queries:
            query = engine.truncate_query(str(row["query"]))
            passages = [
                engine.truncate_passage(documents[str(doc_id)])
                for doc_id in row["candidates"]
            ]
            longest = sorted(
                passages,
                key=lambda text: len(
                    engine.tokenizer(text, add_special_tokens=False).input_ids
                ),
                reverse=True,
            )
            prompts = {
                "pairwise": engine.render_pairwise(query, longest[0], longest[1]),
                "setwise": render_setwise(query, longest[:3]),
                "listwise": render_listwise(query, longest[:4]),
            }
            for name, prompt in prompts.items():
                length = len(engine.tokenizer(prompt).input_ids)
                if length > max_lengths[name]:
                    max_lengths[name] = length
                    max_examples[name] = f"{dataset}/{row['query_id']}"

    oversized = {
        name: length
        for name, length in max_lengths.items()
        if length > args.encoder_max_tokens
    }
    if oversized:
        raise RuntimeError(
            f"Rendered prompts exceed the shared encoder limit: {oversized}. "
            "Reduce --passage-tokens."
        )

    first_dataset = args.datasets[0]
    first_queries, first_documents, _ = snapshots[first_dataset]
    smoke = run_smoke(engine, first_queries[0], first_documents, args.smoke_budget)
    malformed = {
        method: values["invalid_outputs"]
        for method, values in smoke.items()
        if values["invalid_outputs"]
    }
    if malformed:
        print(f"PREFLIGHT WARNING: malformed outputs were repaired and counted: {malformed}")

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    manifests = {
        dataset: {
            "sha256": sha256(snapshot_dir(dataset) / "manifest.json"),
            "contents": manifest,
        }
        for dataset, (_, _, manifest) in snapshots.items()
    }
    provenance = {
        "protocol_version": 2,
        "git_commit": git_commit,
        "git_tracked_files_dirty": bool(git_status),
        "command": sys.argv,
        "datasets": args.datasets,
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if device_type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "sentencepiece": importlib.metadata.version("sentencepiece"),
        "python": sys.version,
        "query_tokens": args.query_tokens,
        "passage_tokens": args.passage_tokens,
        "encoder_max_tokens": args.encoder_max_tokens,
        "hybrid_stage_a_fraction": HYBRID_STAGE_A_FRACTION,
        "model_output_cache": False,
        "max_rendered_prompt_tokens": max_lengths,
        "max_prompt_examples": max_examples,
        "smoke": smoke,
        "snapshot_manifests": manifests,
    }
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    provenance["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = EXP_DIR / "results/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()

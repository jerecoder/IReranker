#!/usr/bin/env python3
"""Fail fast on the frozen snapshot, CUDA, prompts, and every method family."""

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

from experiments.mohajer_hybrid_probe.engine import SharedFlanT5Engine  # noqa: E402
from experiments.robust04_cross_paradigm.methods import (  # noqa: E402
    render_listwise,
    render_setwise,
)
from experiments.reviewer_response.common import (  # noqa: E402
    EXPERIMENT_1_METHODS,
    EXPERIMENT_2_METHODS,
    EXP_DIR,
    HYBRID_STAGE_A_FRACTION,
    PILOT_QUERY_IDS,
    QUERY_COUNT,
    RESULTS_DIR,
    SNAPSHOT_DIR,
    TOKEN_BUDGETS,
    load_snapshot,
    method_seeds,
    sha256,
    write_json,
)
from experiments.reviewer_response.methods import execute_method  # noqa: E402


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
    parser.add_argument("--smoke-budget", type=int, default=20000)
    args = parser.parse_args()
    if min(
        args.query_tokens,
        args.passage_tokens,
        args.encoder_max_tokens,
        args.smoke_budget,
    ) <= 0:
        parser.error("all token limits must be positive")

    import torch

    device_type = torch.device(args.device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 15:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; require at least 15 GiB")

    queries, documents, manifest = load_snapshot()
    qids = [str(row["query_id"]) for row in queries]
    if len(qids) != QUERY_COUNT or set(qids) & set(PILOT_QUERY_IDS):
        raise RuntimeError("Confirmatory query isolation check failed")

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
    for row in queries:
        query = engine.truncate_query(str(row["query"]))
        passages = [
            engine.truncate_passage(documents[str(doc_id)]) for doc_id in row["candidates"]
        ]
        longest = sorted(
            passages,
            key=lambda text: len(engine.tokenizer(text, add_special_tokens=False).input_ids),
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
                max_examples[name] = str(row["query_id"])
    oversized = {
        name: length
        for name, length in max_lengths.items()
        if length > args.encoder_max_tokens
    }
    if oversized:
        raise RuntimeError(
            f"Rendered prompts exceed encoder limit: {oversized}. Reduce passage tokens."
        )

    smoke_row = dict(queries[0])
    smoke_row["candidates"] = [str(value) for value in queries[0]["candidates"][:6]]
    smoke: dict[str, Any] = {}
    methods = sorted((set(EXPERIMENT_1_METHODS) | set(EXPERIMENT_2_METHODS)) - {"bm25"})
    for method in methods:
        result = execute_method(
            method,
            row=smoke_row,
            documents=documents,
            engine=engine,
            seed=42,
            token_budget=args.smoke_budget,
        )
        candidates = smoke_row["candidates"]
        if len(result.ranking) != len(candidates) or set(result.ranking) != set(candidates):
            raise RuntimeError(f"{method} smoke test returned an invalid permutation")
        if result.stage_a_tokens + result.stage_b_tokens != result.meter.total_model_tokens:
            raise RuntimeError(f"{method} stage token accounting is inconsistent")
        smoke[method] = {
            "tokens": result.meter.total_model_tokens,
            "logical_comparisons": result.meter.logical_comparisons,
            "prompt_instances": result.meter.directional_prompt_instances,
            "invalid_outputs": result.meter.invalid_outputs,
            "inconsistent_outputs": result.meter.inconsistent_outputs,
            "stage_a_tokens": result.stage_a_tokens,
            "stage_b_tokens": result.stage_b_tokens,
        }
    malformed = {
        method: values["invalid_outputs"]
        for method, values in smoke.items()
        if values["invalid_outputs"]
    }
    if malformed:
        print(f"PREFLIGHT WARNING: malformed outputs were repaired and counted: {malformed}")

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance = {
        "git_commit": git_commit,
        "git_tracked_files_dirty": bool(git_status),
        "command": sys.argv,
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
        "query_count": QUERY_COUNT,
        "experiment_1_llm_query_runs": QUERY_COUNT
        * len(TOKEN_BUDGETS)
        * sum(
            len(method_seeds(method))
            for method in EXPERIMENT_1_METHODS
            if method != "bm25"
        ),
        "experiment_2_incremental_llm_query_runs": QUERY_COUNT
        * len(TOKEN_BUDGETS)
        * sum(
            len(method_seeds(method))
            for method in EXPERIMENT_2_METHODS
            if method not in EXPERIMENT_1_METHODS
        ),
        "pilot_queries_excluded": PILOT_QUERY_IDS,
        "snapshot_manifest_sha256": sha256(SNAPSHOT_DIR / "manifest.json"),
        "snapshot_manifest": manifest,
        "max_rendered_prompt_tokens": max_lengths,
        "max_prompt_query_ids": max_examples,
        "smoke": smoke,
    }
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    provenance["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EXP_DIR / "results/provenance.json", provenance)
    print(json.dumps(provenance, indent=2))
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()

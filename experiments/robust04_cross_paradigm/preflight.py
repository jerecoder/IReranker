#!/usr/bin/env python3
"""Fail fast on data, dependency, CUDA, prompt-length, and inference problems."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.robust04_cross_paradigm.common import (  # noqa: E402
    EXP_DIR,
    MANIFEST_PATH,
    load_snapshot,
)
from experiments.robust04_cross_paradigm.engine import (  # noqa: E402
    SharedFlanT5Engine,
    SharedSamplingOracle,
    UsageMeter,
)
from experiments.robust04_cross_paradigm.methods import (  # noqa: E402
    render_listwise,
    render_setwise,
    run_listwise_rankgpt,
    run_prp_heapsort,
    run_setwise_heapsort,
)
from ireranker.rankers import get_ranker  # noqa: E402
from ireranker.types import RankingTask  # noqa: E402


def validate_permutation(name: str, ranking: list[str], candidates: list[str]) -> None:
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise RuntimeError(f"{name} smoke test returned an invalid permutation: {ranking}")


def meter_snapshot(meter: UsageMeter) -> dict[str, int]:
    return {
        "logical_comparisons": meter.logical_comparisons,
        "choice_events": meter.choice_events,
        "prompt_instances": meter.directional_prompt_instances,
        "document_instances": meter.document_instances,
        "generation_invocations": meter.generation_invocations,
        "encoder_nonpad_tokens": meter.encoder_nonpad_tokens,
        "encoder_padded_slots": meter.encoder_padded_slots,
        "decoder_tokens": meter.decoder_tokens,
        "total_model_tokens": meter.total_model_tokens,
        "invalid_outputs": meter.invalid_outputs,
        "inconsistent_outputs": meter.inconsistent_outputs,
    }


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
    args = parser.parse_args()
    if min(args.query_tokens, args.passage_tokens, args.encoder_max_tokens) <= 0:
        parser.error("token limits must be positive")

    import accelerate
    import torch
    import transformers

    device_type = torch.device(args.device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 20:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; require at least 20 GiB")

    queries, documents = load_snapshot()
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        encoder_max_tokens=args.encoder_max_tokens,
        query_tokens=args.query_tokens,
        passage_tokens=args.passage_tokens,
    )

    max_lengths = {"pairwise": 0, "setwise": 0, "listwise": 0}
    max_examples: dict[str, str] = {}
    for row in queries:
        query = engine.truncate_query(str(row["query"]))
        candidates = [str(value) for value in row["candidates"]]
        passages = [engine.truncate_passage(documents[doc_id]) for doc_id in candidates]
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

    oversized = {name: length for name, length in max_lengths.items() if length > args.encoder_max_tokens}
    if oversized:
        raise RuntimeError(
            f"Prompts exceed the shared encoder limit: {oversized}. "
            "Reduce --passage-tokens before running."
        )

    first = queries[0]
    qid = str(first["query_id"])
    candidates = [str(value) for value in first["candidates"][:4]]
    qrels = {str(key): int(value) for key, value in first["qrels"].items()}
    smoke_rankings: dict[str, list[str]] = {"bm25": list(candidates)}
    smoke_meters: dict[str, UsageMeter] = {}

    oracle = SharedSamplingOracle(
        engine=engine,
        queries={qid: str(first["query"])},
        documents=documents,
        seed=42,
    )
    ranker = get_ranker("mohajer (ir)", oracle=oracle, seed=42, top_k=10)
    ranker.set_dataset("robust04", split="test", query_ids=[qid])
    task = RankingTask(
        query_id=qid,
        candidate_ids=list(candidates),
        y_true=[float(qrels.get(doc_id, 0)) for doc_id in candidates],
        dataset_path=str(ROOT / "data/external/beir/robust04"),
    )
    indices = ranker.rank(task)
    smoke_rankings["mohajer"] = [candidates[index] for index in indices]
    smoke_meters["mohajer"] = oracle.meter

    for name, runner, kwargs in (
        ("prp", run_prp_heapsort, {"k": 10}),
        ("setwise", run_setwise_heapsort, {"num_child": 2, "k": 10}),
        (
            "listwise",
            run_listwise_rankgpt,
            {"window_size": 4, "step_size": 2, "repeats": 5},
        ),
    ):
        meter = UsageMeter()
        smoke_rankings[name] = runner(
            str(first["query"]),
            list(candidates),
            documents,
            engine=engine,
            meter=meter,
            **kwargs,
        )
        smoke_meters[name] = meter

    for name, ranking in smoke_rankings.items():
        validate_permutation(name, ranking, candidates)
    malformed = {
        name: meter.invalid_outputs
        for name, meter in smoke_meters.items()
        if meter.invalid_outputs
    }
    if malformed:
        print(
            "PREFLIGHT WARNING: malformed model outputs were repaired into full "
            f"permutations and will be counted in invalid_outputs: {malformed}"
        )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
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
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "python": sys.version,
        "query_tokens": args.query_tokens,
        "passage_tokens": args.passage_tokens,
        "encoder_max_tokens": args.encoder_max_tokens,
        "max_rendered_prompt_tokens": max_lengths,
        "max_prompt_query_ids": max_examples,
        "smoke_rankings": smoke_rankings,
        "smoke_usage": {name: meter_snapshot(meter) for name, meter in smoke_meters.items()},
        "smoke_malformed_outputs_repaired": malformed,
        "snapshot_manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    }
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    provenance["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = EXP_DIR / "results/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != provenance:
            raise RuntimeError(
                f"Existing provenance is incompatible with this run: {output}. "
                "Start fresh instead of resuming mixed configurations."
            )
    else:
        output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()

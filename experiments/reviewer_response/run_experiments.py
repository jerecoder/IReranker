#!/usr/bin/env python3
"""Run both frozen reviewer-response experiments with one loaded FLAN-T5 model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.engine import (  # noqa: E402
    SharedFlanT5Engine,
    UsageMeter,
)
from experiments.robust04_cross_paradigm.methods import (  # noqa: E402
    render_listwise,
    render_setwise,
)
from experiments.reviewer_response.common import (  # noqa: E402
    DATASET,
    DETERMINISTIC_SEED,
    EXPERIMENT_1_METHODS,
    EXPERIMENT_2_METHODS,
    EXP_DIR,
    HYBRID_STAGE_A_FRACTION,
    METHOD_VARIANTS,
    PER_QUERY_DIR,
    PROTOCOL_VERSION,
    QUERY_COUNT,
    RESULTS_DIR,
    RUNS_DIR,
    SNAPSHOT_DIR,
    STATUS_PATH,
    STOCHASTIC_SEEDS,
    TOKEN_BUDGETS,
    condition_name,
    load_snapshot,
    method_seeds,
    ndcg_at_k,
    sha256,
    write_csv,
    write_json,
    write_trec_run,
)
from experiments.reviewer_response.methods import (  # noqa: E402
    MethodResult,
    execute_method,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def meter_row(meter: UsageMeter) -> dict[str, Any]:
    prompts = meter.directional_prompt_instances
    return {
        "logical_comparisons": meter.logical_comparisons,
        "choice_events": meter.choice_events,
        "prompt_instances": prompts,
        "generation_invocations": meter.generation_invocations,
        "document_instances": meter.document_instances,
        "avg_documents_per_prompt": meter.document_instances / prompts if prompts else 0.0,
        "encoder_nonpad_tokens": meter.encoder_nonpad_tokens,
        "encoder_padded_slots": meter.encoder_padded_slots,
        "decoder_tokens": meter.decoder_tokens,
        "total_model_tokens": meter.total_model_tokens,
        "inference_seconds": meter.inference_seconds,
        "invalid_outputs": meter.invalid_outputs,
        "inconsistent_outputs": meter.inconsistent_outputs,
    }


def validate_ranking(ranking: list[str], candidates: list[str], label: str) -> None:
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise RuntimeError(f"{label} did not return a full candidate permutation")


def completion_matches(
    done_path: Path,
    csv_path: Path,
    run_path: Path,
    signature: dict[str, Any],
) -> bool:
    try:
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        return (
            marker["status"] == "complete"
            and marker["signature"] == signature
            and int(marker["rows"]) == QUERY_COUNT
            and marker["csv_sha256"] == sha256(csv_path)
            and marker["run_sha256"] == sha256(run_path)
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return False


def warm_up(
    engine: SharedFlanT5Engine,
    row: dict[str, Any],
    documents: dict[str, str],
) -> None:
    query = engine.truncate_query(str(row["query"]))
    ids = [str(value) for value in row["candidates"][:4]]
    passages = [engine.truncate_passage(documents[doc_id]) for doc_id in ids]
    engine.generate(
        [
            engine.render_pairwise(query, passages[0], passages[1]),
            engine.render_pairwise(query, passages[1], passages[0]),
        ],
        meter=UsageMeter(),
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[2, 2],
    )
    engine.generate(
        [render_setwise(query, passages[:3])],
        meter=UsageMeter(),
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[3],
    )
    engine.generate(
        [render_listwise(query, passages)],
        meter=UsageMeter(),
        max_new_tokens=20,
        decoder_prefix=False,
        document_counts=[4],
    )


def run_condition(
    *,
    method: str,
    budget: int | None,
    seed: int,
    queries: list[dict[str, Any]],
    documents: dict[str, str],
    engine: SharedFlanT5Engine | None,
    base_signature: dict[str, Any],
    resume: bool,
    status: dict[str, Any],
) -> None:
    condition = condition_name(method, budget, seed)
    csv_path = PER_QUERY_DIR / f"{condition}.csv"
    run_path = RUNS_DIR / f"{condition}.txt"
    done_path = PER_QUERY_DIR / f"{condition}.done"
    qids = [str(row["query_id"]) for row in queries]
    signature = {
        **base_signature,
        "method": method,
        "variant": METHOD_VARIANTS[method],
        "budget": budget,
        "seed": seed,
        "condition": condition,
        "query_ids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
    }
    if resume and completion_matches(done_path, csv_path, run_path, signature):
        print(f"SKIP verified condition: {condition}")
        status["completed_conditions"] = sorted(
            set(status.get("completed_conditions", [])) | {condition}
        )
        write_json(STATUS_PATH, status)
        return
    done_path.unlink(missing_ok=True)
    status.update({"current_condition": condition, "current_query": None, "updated_utc": utc_now()})
    write_json(STATUS_PATH, status)

    result_rows: list[dict[str, Any]] = []
    rankings: list[tuple[str, list[str]]] = []
    condition_started = time.perf_counter()
    for index, row in enumerate(queries, start=1):
        qid = str(row["query_id"])
        status.update({"current_query": qid, "query_index": index, "updated_utc": utc_now()})
        write_json(STATUS_PATH, status)
        candidates = [str(value) for value in row["candidates"]]
        qrels = {str(key): int(value) for key, value in row["qrels"].items()}
        torch = engine.torch if engine is not None else None
        if engine is not None and engine.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            torch.cuda.synchronize(engine.device)
        started = time.perf_counter()
        if method == "bm25":
            output = MethodResult(list(candidates), UsageMeter(), 0, 0)
        else:
            if engine is None or budget is None:
                raise RuntimeError(f"{method} requires an engine and token budget")
            output = execute_method(
                method,
                row=row,
                documents=documents,
                engine=engine,
                seed=seed,
                token_budget=budget,
            )
        if engine is not None and engine.device_type == "cuda":
            torch.cuda.synchronize(engine.device)
        wall_seconds = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(engine.device))
            if engine is not None and engine.device_type == "cuda"
            else 0
        )
        validate_ranking(output.ranking, candidates, f"{condition}/{qid}")
        rankings.append((qid, output.ranking))
        result_rows.append(
            {
                "dataset": DATASET,
                "condition": condition,
                "method": method,
                "variant": METHOD_VARIANTS[method],
                "token_budget": budget if budget is not None else "",
                "seed": seed,
                "query_id": qid,
                "ndcg10": ndcg_at_k(output.ranking, qrels, 10),
                "stage_a_tokens": output.stage_a_tokens,
                "stage_b_tokens": output.stage_b_tokens,
                **meter_row(output.meter),
                "query_wall_seconds": wall_seconds,
                "peak_gpu_memory_bytes": peak_memory,
            }
        )
        write_csv(csv_path, result_rows)
        write_trec_run(run_path, rankings, condition)
        print(
            f"{condition}: {index}/{QUERY_COUNT} qid={qid} "
            f"ndcg10={result_rows[-1]['ndcg10']:.4f} "
            f"tokens={output.meter.total_model_tokens} wall={wall_seconds:.1f}s"
        )

    marker = {
        "status": "complete",
        "signature": signature,
        "rows": len(result_rows),
        "csv_sha256": sha256(csv_path),
        "run_sha256": sha256(run_path),
        "condition_wall_seconds": time.perf_counter() - condition_started,
        "completed_utc": utc_now(),
    }
    write_json(done_path, marker)
    status["completed_conditions"] = sorted(
        set(status.get("completed_conditions", [])) | {condition}
    )
    status.update({"current_query": None, "updated_utc": utc_now()})
    write_json(STATUS_PATH, status)


def condition_plan(methods: list[str]) -> list[tuple[str, int | None, int]]:
    plan: list[tuple[str, int | None, int]] = []
    if "bm25" in methods:
        plan.append(("bm25", None, DETERMINISTIC_SEED))
    for budget in TOKEN_BUDGETS:
        for method in methods:
            if method == "bm25":
                continue
            for seed in method_seeds(method):
                plan.append((method, budget, seed))
    return plan


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    queries, documents, _ = load_snapshot()
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=args.query_tokens,
        passage_tokens=args.passage_tokens,
        encoder_max_tokens=args.encoder_max_tokens,
    )
    warm_up(engine, queries[0], documents)

    source_paths = [
        Path(__file__),
        EXP_DIR / "common.py",
        EXP_DIR / "methods.py",
        ROOT / "experiments/mohajer_hybrid_probe/common.py",
        ROOT / "experiments/mohajer_hybrid_probe/engine.py",
        ROOT / "experiments/mohajer_hybrid_probe/methods.py",
        ROOT / "experiments/robust04_cross_paradigm/engine.py",
        ROOT / "experiments/robust04_cross_paradigm/methods.py",
        ROOT / "ireranker/rankers/mohajer_ranker.py",
        ROOT / "ireranker/rankers/prp_sorting_ranker.py",
        ROOT / "ireranker/rankers/ranker.py",
        ROOT / "ireranker/oracles/oracle.py",
    ]
    base_signature = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "query_count": QUERY_COUNT,
        "token_budgets": TOKEN_BUDGETS,
        "stochastic_seeds": STOCHASTIC_SEEDS,
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "gpu": (
            engine.torch.cuda.get_device_name(engine.device)
            if engine.device_type == "cuda"
            else args.device
        ),
        "torch": str(engine.torch.__version__),
        "cuda": str(engine.torch.version.cuda),
        "transformers": importlib.metadata.version("transformers"),
        "sentencepiece": importlib.metadata.version("sentencepiece"),
        "python": sys.version,
        "query_tokens": args.query_tokens,
        "passage_tokens": args.passage_tokens,
        "encoder_max_tokens": args.encoder_max_tokens,
        "hybrid_stage_a_fraction": HYBRID_STAGE_A_FRACTION,
        "model_output_cache": False,
        "snapshot_manifest_sha256": sha256(SNAPSHOT_DIR / "manifest.json"),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
    }
    status: dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "current_experiment": 1,
        "current_condition": None,
        "current_query": None,
        "completed_conditions": [],
        "query_count": QUERY_COUNT,
        "token_budgets": TOKEN_BUDGETS,
        "stochastic_seeds": STOCHASTIC_SEEDS,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(STATUS_PATH, status)
    overall_started = time.perf_counter()
    try:
        experiment_1_started = time.perf_counter()
        for method, budget, seed in condition_plan(EXPERIMENT_1_METHODS):
            run_condition(
                method=method,
                budget=budget,
                seed=seed,
                queries=queries,
                documents=documents,
                engine=None if method == "bm25" else engine,
                base_signature=base_signature,
                resume=args.resume,
                status=status,
            )
        status["experiment_1_wall_seconds"] = time.perf_counter() - experiment_1_started
        status["experiment_1_complete_utc"] = utc_now()
        status["current_experiment"] = 2
        write_json(STATUS_PATH, status)

        experiment_2_started = time.perf_counter()
        extension_methods = [
            method for method in EXPERIMENT_2_METHODS if method not in EXPERIMENT_1_METHODS
        ]
        for method, budget, seed in condition_plan(extension_methods):
            run_condition(
                method=method,
                budget=budget,
                seed=seed,
                queries=queries,
                documents=documents,
                engine=engine,
                base_signature=base_signature,
                resume=args.resume,
                status=status,
            )
        status["experiment_2_incremental_wall_seconds"] = (
            time.perf_counter() - experiment_2_started
        )
        status["experiment_2_complete_utc"] = utc_now()
        status.update(
            {
                "status": "inference_complete",
                "current_condition": None,
                "current_query": None,
                "overall_wall_seconds": time.perf_counter() - overall_started,
                "updated_utc": utc_now(),
            }
        )
        write_json(STATUS_PATH, status)
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "overall_wall_seconds": time.perf_counter() - overall_started,
                "updated_utc": utc_now(),
            }
        )
        write_json(STATUS_PATH, status)
        raise


if __name__ == "__main__":
    main()

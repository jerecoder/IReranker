#!/usr/bin/env python3
"""One-seed corrected standard Setwise vs true Active-Setwise experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.engine import (  # noqa: E402
    SharedFlanT5Engine,
    UsageMeter,
)
from experiments.reviewer_response.active_setwise import (  # noqa: E402
    run_active_setwise,
    run_standard_setwise_randomized,
)
from experiments.reviewer_response.analyze import (  # noqa: E402
    bootstrap_interval,
    sign_flip_p_value,
)
from experiments.reviewer_response.common import (  # noqa: E402
    DATASET,
    EXP_DIR,
    QUERY_COUNT,
    SNAPSHOT_DIR,
    TOKEN_BUDGETS,
    load_snapshot,
    mean,
    ndcg_at_k,
    sha256,
    write_csv,
    write_json,
    write_trec_run,
)
from experiments.robust04_cross_paradigm.methods import render_setwise  # noqa: E402


OUTPUT_DIR = EXP_DIR / "results" / "true_active_setwise_single_seed"
PER_QUERY_DIR = OUTPUT_DIR / "per_query"
RUNS_DIR = OUTPUT_DIR / "runs"
METRICS_DIR = OUTPUT_DIR / "metrics"
STATUS_PATH = OUTPUT_DIR / "status.json"
SEED = 42
METHODS = ["setwise_randomized", "active_setwise"]
VARIANTS = {
    "bm25": "top100",
    "setwise_randomized": "standard_heapsort_c3_randomized_presentation_top10",
    "active_setwise": "mohajer_groups_c3_setwise_tournament_top10",
}


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def condition_name(method: str, budget: int | None) -> str:
    return "bm25" if method == "bm25" else f"{method}_t{budget}_s{SEED}"


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
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_condition(
    *,
    method: str,
    budget: int | None,
    queries: list[dict[str, Any]],
    documents: dict[str, str],
    engine: SharedFlanT5Engine | None,
    base_signature: dict[str, Any],
    resume: bool,
    status: dict[str, Any],
) -> None:
    condition = condition_name(method, budget)
    csv_path = PER_QUERY_DIR / f"{condition}.csv"
    run_path = RUNS_DIR / f"{condition}.txt"
    done_path = PER_QUERY_DIR / f"{condition}.done"
    qids = [str(row["query_id"]) for row in queries]
    signature = {
        **base_signature,
        "method": method,
        "variant": VARIANTS[method],
        "budget": budget,
        "seed": SEED,
        "condition": condition,
        "query_ids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
    }
    if resume and completion_matches(done_path, csv_path, run_path, signature):
        print(f"SKIP verified: {condition}")
        return
    done_path.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    rankings: list[tuple[str, list[str]]] = []
    started_condition = time.perf_counter()
    for index, row in enumerate(queries, start=1):
        qid = str(row["query_id"])
        status.update(
            {
                "current_condition": condition,
                "current_query": qid,
                "query_index": index,
                "updated_utc": utc_now(),
            }
        )
        write_json(STATUS_PATH, status)
        candidates = [str(value) for value in row["candidates"]]
        qrels = {str(key): int(value) for key, value in row["qrels"].items()}
        torch = engine.torch if engine is not None else None
        if engine is not None and engine.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            torch.cuda.synchronize(engine.device)
        started = time.perf_counter()
        if method == "bm25":
            ranking = candidates
            meter = UsageMeter()
        elif method == "setwise_randomized":
            if engine is None or budget is None:
                raise RuntimeError("Setwise requires an engine and budget")
            result = run_standard_setwise_randomized(
                row=row,
                documents=documents,
                engine=engine,
                seed=SEED,
                token_budget=budget,
            )
            ranking, meter = result.ranking, result.meter
        elif method == "active_setwise":
            if engine is None or budget is None:
                raise RuntimeError("Active-Setwise requires an engine and budget")
            result = run_active_setwise(
                row=row,
                documents=documents,
                engine=engine,
                seed=SEED,
                token_budget=budget,
            )
            ranking, meter = result.ranking, result.meter
        else:
            raise ValueError(method)
        if engine is not None and engine.device_type == "cuda":
            torch.cuda.synchronize(engine.device)
        wall_seconds = time.perf_counter() - started
        if len(ranking) != len(candidates) or set(ranking) != set(candidates):
            raise RuntimeError(f"Invalid permutation: {condition}/{qid}")
        if method != "bm25" and meter.logical_comparisons != 0:
            raise RuntimeError(f"Pairwise comparisons leaked into {condition}/{qid}")
        peak_memory = (
            int(torch.cuda.max_memory_allocated(engine.device))
            if engine is not None and engine.device_type == "cuda"
            else 0
        )
        rankings.append((qid, ranking))
        rows.append(
            {
                "dataset": DATASET,
                "condition": condition,
                "method": method,
                "variant": VARIANTS[method],
                "token_budget": budget if budget is not None else "",
                "seed": SEED,
                "query_id": qid,
                "ndcg10": ndcg_at_k(ranking, qrels, 10),
                **meter_row(meter),
                "query_wall_seconds": wall_seconds,
                "peak_gpu_memory_bytes": peak_memory,
            }
        )
        write_csv(csv_path, rows)
        write_trec_run(run_path, rankings, condition)
        print(
            f"{condition}: {index}/{QUERY_COUNT} qid={qid} "
            f"ndcg10={rows[-1]['ndcg10']:.4f} tokens={meter.total_model_tokens} "
            f"prompts={meter.directional_prompt_instances} wall={wall_seconds:.1f}s"
        )
    write_json(
        done_path,
        {
            "status": "complete",
            "signature": signature,
            "rows": len(rows),
            "csv_sha256": sha256(csv_path),
            "run_sha256": sha256(run_path),
            "condition_wall_seconds": time.perf_counter() - started_condition,
            "completed_utc": utc_now(),
        },
    )


def load_verified(method: str, budget: int | None) -> list[dict[str, str]]:
    condition = condition_name(method, budget)
    csv_path = PER_QUERY_DIR / f"{condition}.csv"
    run_path = RUNS_DIR / f"{condition}.txt"
    done_path = PER_QUERY_DIR / f"{condition}.done"
    marker = json.loads(done_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or int(marker.get("rows", -1)) != QUERY_COUNT
        or marker.get("csv_sha256") != sha256(csv_path)
        or marker.get("run_sha256") != sha256(run_path)
    ):
        raise ValueError(f"Unverified analysis input: {condition}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != QUERY_COUNT or len({row["query_id"] for row in rows}) != QUERY_COUNT:
        raise ValueError(f"Invalid analysis rows: {condition}")
    return rows


def analyze() -> None:
    baseline_rows = load_verified("bm25", None)
    baseline = {row["query_id"]: float(row["ndcg10"]) for row in baseline_rows}
    summary: list[dict[str, Any]] = [
        {
            "method": "bm25",
            "variant": VARIANTS["bm25"],
            "token_budget": "",
            "queries": QUERY_COUNT,
            "seed": SEED,
            "ndcg10": mean(baseline.values()),
            "delta_vs_bm25": 0.0,
            "avg_tokens": 0.0,
            "avg_gpu_seconds": 0.0,
            "avg_wall_seconds": 0.0,
            "avg_prompt_instances": 0.0,
            "avg_document_instances": 0.0,
            "invalid_outputs": 0,
        }
    ]
    tests: list[dict[str, Any]] = []
    for budget in TOKEN_BUDGETS:
        method_values: dict[str, dict[str, float]] = {}
        for method in METHODS:
            rows = load_verified(method, budget)
            values = {row["query_id"]: float(row["ndcg10"]) for row in rows}
            method_values[method] = values
            summary.append(
                {
                    "method": method,
                    "variant": VARIANTS[method],
                    "token_budget": budget,
                    "queries": QUERY_COUNT,
                    "seed": SEED,
                    "ndcg10": mean(values.values()),
                    "delta_vs_bm25": mean(
                        values[qid] - baseline[qid] for qid in sorted(values)
                    ),
                    "avg_tokens": mean(float(row["total_model_tokens"]) for row in rows),
                    "avg_gpu_seconds": mean(float(row["inference_seconds"]) for row in rows),
                    "avg_wall_seconds": mean(float(row["query_wall_seconds"]) for row in rows),
                    "avg_prompt_instances": mean(float(row["prompt_instances"]) for row in rows),
                    "avg_document_instances": mean(
                        float(row["document_instances"]) for row in rows
                    ),
                    "invalid_outputs": sum(float(row["invalid_outputs"]) for row in rows),
                }
            )
        qids = sorted(method_values["active_setwise"])
        differences = np.array(
            [
                method_values["active_setwise"][qid]
                - method_values["setwise_randomized"][qid]
                for qid in qids
            ]
        )
        low, high = bootstrap_interval(differences, seed=20260712, resamples=10000)
        tests.append(
            {
                "token_budget": budget,
                "method_a": "active_setwise",
                "method_b": "setwise_randomized",
                "queries": QUERY_COUNT,
                "seed": SEED,
                "mean_delta_ndcg10": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "p_value_sign_flip": sign_flip_p_value(
                    differences, seed=20260712, resamples=10000
                ),
                "wins": int(np.count_nonzero(differences > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                "losses": int(np.count_nonzero(differences < -1e-12)),
            }
        )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS_DIR / "summary.csv", summary)
    write_csv(METRICS_DIR / "paired_tests.csv", tests)
    write_json(
        METRICS_DIR / "recommendation.json",
        {
            "exploratory_single_seed": True,
            "pairwise_preprocessing": False,
            "all_oracle_events_are_randomized_presentation_setwise": True,
            "primary_comparison": tests[0],
            "next_step": (
                "If the 100k point is promising, run seeds 43 and 44. Otherwise stop "
                "Active-Setwise expansion."
            ),
        },
    )
    print(json.dumps({"summary": summary, "paired_tests": tests}, indent=2))


def warm_up_setwise(
    engine: SharedFlanT5Engine,
    row: dict[str, Any],
    documents: dict[str, str],
) -> None:
    query = engine.truncate_query(str(row["query"]))
    ids = [str(value) for value in row["candidates"][:3]]
    passages = [engine.truncate_passage(documents[doc_id]) for doc_id in ids]
    engine.generate(
        [render_setwise(query, passages)],
        meter=UsageMeter(),
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[3],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    queries, documents, _ = load_snapshot()
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=32,
        passage_tokens=100,
        encoder_max_tokens=768,
    )
    warm_up_setwise(engine, queries[0], documents)
    source_paths = [
        Path(__file__),
        EXP_DIR / "active_multiway.py",
        EXP_DIR / "active_setwise.py",
        ROOT / "experiments/mohajer_hybrid_probe/engine.py",
        ROOT / "experiments/robust04_cross_paradigm/methods.py",
    ]
    base_signature = {
        "protocol_version": 1,
        "dataset": DATASET,
        "query_count": QUERY_COUNT,
        "seed": SEED,
        "token_budgets": TOKEN_BUDGETS,
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "model_output_cache": False,
        "pairwise_preprocessing": False,
        "setwise_presentation_randomized": True,
        "snapshot_manifest_sha256": sha256(SNAPSHOT_DIR / "manifest.json"),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
    }
    status: dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now(),
        "current_condition": None,
        "current_query": None,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(STATUS_PATH, status)
    started = time.perf_counter()
    try:
        run_condition(
            method="bm25",
            budget=None,
            queries=queries,
            documents=documents,
            engine=None,
            base_signature=base_signature,
            resume=args.resume,
            status=status,
        )
        for budget in TOKEN_BUDGETS:
            for method in METHODS:
                run_condition(
                    method=method,
                    budget=budget,
                    queries=queries,
                    documents=documents,
                    engine=engine,
                    base_signature=base_signature,
                    resume=args.resume,
                    status=status,
                )
        analyze()
        status.update(
            {
                "status": "complete",
                "current_condition": None,
                "current_query": None,
                "wall_seconds": time.perf_counter() - started,
                "completed_utc": utc_now(),
            }
        )
        write_json(STATUS_PATH, status)
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
                "updated_utc": utc_now(),
            }
        )
        write_json(STATUS_PATH, status)
        raise


if __name__ == "__main__":
    main()

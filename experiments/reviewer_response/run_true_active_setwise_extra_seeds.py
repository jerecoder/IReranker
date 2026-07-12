#!/usr/bin/env python3
"""Extend the corrected Setwise pilot with seeds 43 and 44 only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.engine import SharedFlanT5Engine  # noqa: E402
from experiments.reviewer_response.analyze import (  # noqa: E402
    bootstrap_interval,
    holm_adjust,
    sign_flip_p_value,
)
from experiments.reviewer_response.common import (  # noqa: E402
    DATASET,
    EXP_DIR,
    QUERY_COUNT,
    SNAPSHOT_DIR,
    load_snapshot,
    mean,
    sha256,
    write_csv,
    write_json,
)
from experiments.reviewer_response import run_true_active_setwise_quick as quick  # noqa: E402


BUDGET = 50000
NEW_SEEDS = [43, 44]
ALL_SEEDS = [42, 43, 44]
METHODS = ["setwise_randomized", "active_setwise"]
STATUS_PATH = quick.OUTPUT_DIR / "extra_seeds_status.json"
MULTISEED_METRICS_DIR = quick.METRICS_DIR / "multiseed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verified_rows(
    method: str,
    seed: int,
    *,
    model: str,
    model_revision: str,
    snapshot_hash: str,
) -> list[dict[str, str]]:
    rows = quick.load_verified(method, BUDGET, seed)
    condition = quick.condition_name(method, BUDGET, seed)
    marker_path = quick.PER_QUERY_DIR / f"{condition}.done"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    signature = marker.get("signature", {})
    expected = {
        "protocol_version": 2,
        "dataset": DATASET,
        "query_count": QUERY_COUNT,
        "model": model,
        "model_revision": model_revision,
        "model_output_cache": False,
        "pairwise_preprocessing": False,
        "setwise_max_documents_per_prompt": 3,
        "snapshot_manifest_sha256": snapshot_hash,
        "method": method,
        "budget": BUDGET,
        "seed": seed,
        "condition": condition,
        "setwise_presentation": "seeded_randomized_per_choice_event",
    }
    mismatches = {
        key: {"expected": value, "observed": signature.get(key)}
        for key, value in expected.items()
        if signature.get(key) != value
    }
    current_sources = {
        "experiments/reviewer_response/active_multiway.py": sha256(
            EXP_DIR / "active_multiway.py"
        ),
        "experiments/reviewer_response/active_setwise.py": sha256(
            EXP_DIR / "active_setwise.py"
        ),
    }
    recorded_sources = signature.get("source_sha256", {})
    for path, expected_hash in current_sources.items():
        if recorded_sources.get(path) != expected_hash:
            mismatches[f"source_sha256:{path}"] = {
                "expected": expected_hash,
                "observed": recorded_sources.get(path),
            }
    if mismatches:
        raise ValueError(f"Incompatible condition {condition}: {mismatches}")
    if any(float(row["invalid_outputs"]) != 0 for row in rows):
        raise ValueError(f"Malformed Setwise output in {condition}")
    if any(float(row["logical_comparisons"]) != 0 for row in rows):
        raise ValueError(f"Pairwise prompt leaked into {condition}")
    return rows


def fixed_rows(
    *, model: str, model_revision: str, snapshot_hash: str
) -> list[dict[str, str]]:
    rows = quick.load_verified("setwise", BUDGET, 42)
    condition = quick.condition_name("setwise", BUDGET, 42)
    marker = json.loads(
        (quick.PER_QUERY_DIR / f"{condition}.done").read_text(encoding="utf-8")
    )
    signature = marker.get("signature", {})
    expected = {
        "protocol_version": 2,
        "dataset": DATASET,
        "query_count": QUERY_COUNT,
        "model": model,
        "model_revision": model_revision,
        "model_output_cache": False,
        "pairwise_preprocessing": False,
        "setwise_max_documents_per_prompt": 3,
        "snapshot_manifest_sha256": snapshot_hash,
        "method": "setwise",
        "budget": BUDGET,
        "seed": 42,
        "setwise_presentation": "fixed",
    }
    mismatches = {
        key: {"expected": value, "observed": signature.get(key)}
        for key, value in expected.items()
        if signature.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible fixed Setwise control: {mismatches}")
    if any(float(row["invalid_outputs"]) != 0 for row in rows):
        raise ValueError("Malformed Setwise output in fixed control")
    if any(float(row["logical_comparisons"]) != 0 for row in rows):
        raise ValueError("Pairwise prompt leaked into fixed control")
    return rows


def per_query_average(rows: list[dict[str, str]]) -> dict[str, float]:
    if len(rows) % QUERY_COUNT != 0:
        raise ValueError(f"Row count {len(rows)} is not divisible by {QUERY_COUNT}")
    expected_repetitions = len(rows) // QUERY_COUNT
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(float(row["ndcg10"]))
    if len(grouped) != QUERY_COUNT:
        raise ValueError(f"Expected {QUERY_COUNT} query groups, got {len(grouped)}")
    if any(len(values) != expected_repetitions for values in grouped.values()):
        raise ValueError("Queries do not have equal seed coverage")
    return {query_id: mean(values) for query_id, values in grouped.items()}


def summary_row(
    method: str,
    rows: list[dict[str, str]],
    *,
    seeds: int,
    baseline: dict[str, float],
) -> dict[str, Any]:
    per_query = per_query_average(rows)
    return {
        "method": method,
        "variant": quick.VARIANTS[method],
        "token_budget": BUDGET,
        "queries": QUERY_COUNT,
        "seeds": seeds,
        "ndcg10": mean(per_query.values()),
        "delta_vs_bm25": mean(
            per_query[query_id] - baseline[query_id] for query_id in sorted(per_query)
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


def analyze(*, model: str, model_revision: str, snapshot_hash: str) -> None:
    baseline_rows = quick.load_verified("bm25", None, 42)
    baseline = {row["query_id"]: float(row["ndcg10"]) for row in baseline_rows}
    fixed = fixed_rows(
        model=model, model_revision=model_revision, snapshot_hash=snapshot_hash
    )
    randomized_by_seed = {
        seed: verified_rows(
            "setwise_randomized",
            seed,
            model=model,
            model_revision=model_revision,
            snapshot_hash=snapshot_hash,
        )
        for seed in ALL_SEEDS
    }
    active_by_seed = {
        seed: verified_rows(
            "active_setwise",
            seed,
            model=model,
            model_revision=model_revision,
            snapshot_hash=snapshot_hash,
        )
        for seed in ALL_SEEDS
    }
    randomized = [row for seed in ALL_SEEDS for row in randomized_by_seed[seed]]
    active = [row for seed in ALL_SEEDS for row in active_by_seed[seed]]

    summary = [
        {
            "method": "bm25",
            "variant": quick.VARIANTS["bm25"],
            "token_budget": "",
            "queries": QUERY_COUNT,
            "seeds": 1,
            "ndcg10": mean(baseline.values()),
            "delta_vs_bm25": 0.0,
            "avg_tokens": 0.0,
            "avg_gpu_seconds": 0.0,
            "avg_wall_seconds": 0.0,
            "avg_prompt_instances": 0.0,
            "avg_document_instances": 0.0,
            "invalid_outputs": 0,
        },
        summary_row("setwise", fixed, seeds=1, baseline=baseline),
        summary_row(
            "setwise_randomized", randomized, seeds=len(ALL_SEEDS), baseline=baseline
        ),
        summary_row("active_setwise", active, seeds=len(ALL_SEEDS), baseline=baseline),
    ]

    seed_summary: list[dict[str, Any]] = []
    for method, by_seed in (
        ("setwise_randomized", randomized_by_seed),
        ("active_setwise", active_by_seed),
    ):
        for seed in ALL_SEEDS:
            row = summary_row(method, by_seed[seed], seeds=1, baseline=baseline)
            row["seed"] = seed
            seed_summary.append(row)

    values = {
        "setwise": per_query_average(fixed),
        "setwise_randomized": per_query_average(randomized),
        "active_setwise": per_query_average(active),
    }
    tests: list[dict[str, Any]] = []
    query_ids = sorted(baseline)
    for index, (method_a, method_b) in enumerate(
        [
            ("active_setwise", "setwise"),
            ("active_setwise", "setwise_randomized"),
            ("setwise_randomized", "setwise"),
        ]
    ):
        differences = np.array(
            [values[method_a][query_id] - values[method_b][query_id] for query_id in query_ids]
        )
        statistical_seed = 20260720 + index
        low, high = bootstrap_interval(
            differences, seed=statistical_seed, resamples=10000
        )
        tests.append(
            {
                "token_budget": BUDGET,
                "method_a": method_a,
                "method_b": method_b,
                "queries": QUERY_COUNT,
                "seeds_a": 1 if method_a == "setwise" else len(ALL_SEEDS),
                "seeds_b": 1 if method_b == "setwise" else len(ALL_SEEDS),
                "mean_delta_ndcg10": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "p_value_sign_flip": sign_flip_p_value(
                    differences, seed=statistical_seed, resamples=10000
                ),
                "wins": int(np.count_nonzero(differences > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                "losses": int(np.count_nonzero(differences < -1e-12)),
            }
        )
    for row, adjusted in zip(
        tests, holm_adjust([float(row["p_value_sign_flip"]) for row in tests])
    ):
        row["p_value_holm"] = adjusted
        row["significant_holm_0.05"] = adjusted < 0.05

    active_summary = next(row for row in summary if row["method"] == "active_setwise")
    fixed_summary = next(row for row in summary if row["method"] == "setwise")
    gate = (
        float(active_summary["ndcg10"]) > float(fixed_summary["ndcg10"])
        and float(active_summary["avg_tokens"]) < float(fixed_summary["avg_tokens"])
    )
    MULTISEED_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(MULTISEED_METRICS_DIR / "summary.csv", summary)
    write_csv(MULTISEED_METRICS_DIR / "seed_summary.csv", seed_summary)
    write_csv(MULTISEED_METRICS_DIR / "paired_tests.csv", tests)
    write_json(
        MULTISEED_METRICS_DIR / "recommendation.json",
        {
            "exploratory_three_seed": True,
            "token_budget": BUDGET,
            "fixed_setwise_reused": True,
            "bm25_reused": True,
            "active_efficiency_gate_vs_fixed_setwise": gate,
            "comparisons": tests,
            "interpretation": (
                "Use seed-averaged query-paired NDCG deltas. All randomized and active "
                "conditions use seeds 42, 43, and 44; fixed Setwise is deterministic."
            ),
        },
    )
    print(json.dumps({"summary": summary, "paired_tests": tests}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    queries, documents, _ = load_snapshot()
    snapshot_hash = sha256(SNAPSHOT_DIR / "manifest.json")
    # Fail before loading model weights if the reusable seed-42 controls do not match.
    fixed_rows(
        model=args.model,
        model_revision=args.model_revision,
        snapshot_hash=snapshot_hash,
    )
    for method in METHODS:
        verified_rows(
            method,
            42,
            model=args.model,
            model_revision=args.model_revision,
            snapshot_hash=snapshot_hash,
        )

    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=32,
        passage_tokens=100,
        encoder_max_tokens=768,
    )
    quick.warm_up_setwise(engine, queries[0], documents)
    source_paths = [
        Path(__file__),
        Path(quick.__file__),
        EXP_DIR / "active_multiway.py",
        EXP_DIR / "active_setwise.py",
        ROOT / "experiments/mohajer_hybrid_probe/engine.py",
        ROOT / "experiments/robust04_cross_paradigm/methods.py",
    ]
    base_signature = {
        "protocol_version": 2,
        "dataset": DATASET,
        "query_count": QUERY_COUNT,
        "extension_seeds": NEW_SEEDS,
        "token_budgets": [BUDGET],
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "model_output_cache": False,
        "pairwise_preprocessing": False,
        "setwise_max_documents_per_prompt": 3,
        "snapshot_manifest_sha256": snapshot_hash,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
    }
    status: dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now(),
        "new_seeds": NEW_SEEDS,
        "current_condition": None,
        "current_query": None,
    }
    write_json(STATUS_PATH, status)
    started = time.perf_counter()
    try:
        for seed in NEW_SEEDS:
            for method in METHODS:
                quick.run_condition(
                    method=method,
                    budget=BUDGET,
                    queries=queries,
                    documents=documents,
                    engine=engine,
                    base_signature=base_signature,
                    resume=True,
                    status=status,
                    seed=seed,
                    status_path=STATUS_PATH,
                )
        analyze(
            model=args.model,
            model_revision=args.model_revision,
            snapshot_hash=snapshot_hash,
        )
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

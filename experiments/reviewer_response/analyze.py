#!/usr/bin/env python3
"""Audit and analyze the two frozen reviewer-response experiments separately."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.reviewer_response.common import (  # noqa: E402
    DETERMINISTIC_SEED,
    EXPERIMENT_1_METHODS,
    EXPERIMENT_2_METHODS,
    METRICS_DIR,
    METHOD_VARIANTS,
    PER_QUERY_DIR,
    QUERY_COUNT,
    RESULTS_DIR,
    RUNS_DIR,
    STATUS_PATH,
    TOKEN_BUDGETS,
    condition_name,
    mean,
    method_seeds,
    pareto_methods,
    sha256,
    write_csv,
    write_json,
)


NUMERIC_FIELDS = {
    "ndcg10",
    "stage_a_tokens",
    "stage_b_tokens",
    "logical_comparisons",
    "choice_events",
    "prompt_instances",
    "generation_invocations",
    "document_instances",
    "avg_documents_per_prompt",
    "encoder_nonpad_tokens",
    "encoder_padded_slots",
    "decoder_tokens",
    "total_model_tokens",
    "inference_seconds",
    "invalid_outputs",
    "inconsistent_outputs",
    "query_wall_seconds",
    "peak_gpu_memory_bytes",
}


def expected_conditions(methods: list[str]) -> list[tuple[str, int | None, int]]:
    expected = [("bm25", None, DETERMINISTIC_SEED)]
    for budget in TOKEN_BUDGETS:
        for method in methods:
            if method == "bm25":
                continue
            for seed in method_seeds(method):
                expected.append((method, budget, seed))
    return expected


def load_experiment_rows(methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_qids: set[str] | None = None
    for method, budget, seed in expected_conditions(methods):
        condition = condition_name(method, budget, seed)
        csv_path = PER_QUERY_DIR / f"{condition}.csv"
        run_path = RUNS_DIR / f"{condition}.txt"
        done_path = PER_QUERY_DIR / f"{condition}.done"
        try:
            marker = json.loads(done_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing completed condition: {condition}") from exc
        if (
            marker.get("status") != "complete"
            or int(marker.get("rows", -1)) != QUERY_COUNT
            or marker.get("csv_sha256") != sha256(csv_path)
            or marker.get("run_sha256") != sha256(run_path)
        ):
            raise ValueError(f"Completion marker/hash mismatch: {condition}")
        signature = marker.get("signature", {})
        if (
            signature.get("condition") != condition
            or signature.get("method") != method
            or signature.get("variant") != METHOD_VARIANTS[method]
            or signature.get("budget") != budget
            or int(signature.get("seed", -1)) != seed
        ):
            raise ValueError(f"Completion signature mismatch: {condition}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            condition_rows = [dict(row) for row in csv.DictReader(handle)]
        if len(condition_rows) != QUERY_COUNT:
            raise ValueError(f"Wrong row count in {condition}: {len(condition_rows)}")
        qids = [str(row["query_id"]) for row in condition_rows]
        if len(set(qids)) != QUERY_COUNT:
            raise ValueError(f"Duplicate query IDs in {condition}")
        if expected_qids is None:
            expected_qids = set(qids)
        elif set(qids) != expected_qids:
            raise ValueError(f"Query set mismatch in {condition}")
        for row in condition_rows:
            if (
                row["condition"] != condition
                or row["method"] != method
                or row["variant"] != METHOD_VARIANTS[method]
                or int(row["seed"]) != seed
            ):
                raise ValueError(f"CSV identity mismatch in {condition}")
            row["token_budget"] = int(row["token_budget"]) if row["token_budget"] else None
            for field in NUMERIC_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def query_seed_average(group: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in group:
        buckets.setdefault(str(row["query_id"]), []).append(row)
    return {
        qid: {
            field: mean(float(row[field]) for row in qrows)
            for field in NUMERIC_FIELDS
        }
        for qid, qrows in buckets.items()
    }


def bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if values.ndim != 1 or not values.size:
        raise ValueError("Bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    samples = values[indices].mean(axis=1)
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def sign_flip_p_value(values: np.ndarray, *, seed: int, resamples: int) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, values.size))
    null_values = np.abs((signs * values).mean(axis=1))
    extreme = int(np.count_nonzero(null_values >= observed - 1e-15))
    return float((extreme + 1) / (resamples + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(sorted(range(total), key=lambda idx: p_values[idx])):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def analyze_experiment(
    number: int,
    methods: list[str],
    comparisons: list[tuple[str, str]],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = load_experiment_rows(methods)
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), row["token_budget"]), []).append(row)
    per_query = {key: query_seed_average(group) for key, group in grouped.items()}
    bm25 = per_query[("bm25", None)]
    summary: list[dict[str, Any]] = []
    for (method, budget), values in grouped.items():
        query_values = per_query[(method, budget)]
        qids = sorted(query_values)
        ndcg_values = np.array([query_values[qid]["ndcg10"] for qid in qids])
        low, high = bootstrap_interval(
            ndcg_values,
            seed=bootstrap_seed,
            resamples=resamples,
        )
        summary.append(
            {
                "experiment": number,
                "method": method,
                "variant": METHOD_VARIANTS[method],
                "token_budget": budget if budget is not None else "",
                "queries": len(qids),
                "seeds": len({int(row["seed"]) for row in values}),
                "ndcg10": float(ndcg_values.mean()),
                "ndcg10_ci_low": low,
                "ndcg10_ci_high": high,
                "delta_vs_bm25": mean(
                    query_values[qid]["ndcg10"] - bm25[qid]["ndcg10"] for qid in qids
                ),
                "avg_tokens": mean(row["total_model_tokens"] for row in values),
                "avg_gpu_seconds": mean(row["inference_seconds"] for row in values),
                "avg_wall_seconds": mean(row["query_wall_seconds"] for row in values),
                "avg_prompt_instances": mean(row["prompt_instances"] for row in values),
                "avg_document_instances": mean(row["document_instances"] for row in values),
                "avg_logical_comparisons": mean(
                    row["logical_comparisons"] for row in values
                ),
                "avg_stage_a_tokens": mean(row["stage_a_tokens"] for row in values),
                "avg_stage_b_tokens": mean(row["stage_b_tokens"] for row in values),
                "invalid_outputs": sum(row["invalid_outputs"] for row in values),
                "inconsistent_outputs": sum(
                    row["inconsistent_outputs"] for row in values
                ),
            }
        )
    for budget in TOKEN_BUDGETS:
        budget_rows = [
            row
            for row in summary
            if row["method"] == "bm25" or row["token_budget"] == budget
        ]
        token_frontier = pareto_methods(budget_rows, "avg_tokens")
        time_frontier = pareto_methods(budget_rows, "avg_gpu_seconds")
        for row in summary:
            if row["token_budget"] == budget:
                row["pareto_tokens"] = row["method"] in token_frontier
                row["pareto_gpu_time"] = row["method"] in time_frontier
    for row in summary:
        if row["method"] == "bm25":
            row["pareto_tokens"] = True
            row["pareto_gpu_time"] = True

    tests: list[dict[str, Any]] = []
    for budget in TOKEN_BUDGETS:
        for left, right in comparisons:
            left_values = per_query[(left, budget)]
            right_values = per_query[(right, budget)]
            qids = sorted(set(left_values) & set(right_values))
            differences = np.array(
                [left_values[qid]["ndcg10"] - right_values[qid]["ndcg10"] for qid in qids]
            )
            low, high = bootstrap_interval(
                differences,
                seed=bootstrap_seed,
                resamples=resamples,
            )
            tests.append(
                {
                    "experiment": number,
                    "token_budget": budget,
                    "method_a": left,
                    "method_b": right,
                    "queries": len(qids),
                    "mean_delta_ndcg10": float(differences.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value_sign_flip": sign_flip_p_value(
                        differences,
                        seed=bootstrap_seed,
                        resamples=resamples,
                    ),
                }
            )
    adjusted = holm_adjust([float(row["p_value_sign_flip"]) for row in tests])
    for row, value in zip(tests, adjusted):
        row["p_value_holm"] = value
        row["significant_holm_0.05"] = value < 0.05

    output_dir = METRICS_DIR / f"experiment_{number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "summary.csv",
        sorted(
            summary,
            key=lambda row: (
                -1 if row["token_budget"] == "" else int(row["token_budget"]),
                -float(row["ndcg10"]),
            ),
        ),
    )
    write_csv(output_dir / "paired_tests.csv", tests)
    primary = tests[0] if tests else None
    recommendation = {
        "experiment": number,
        "confirmatory_query_count": QUERY_COUNT,
        "pilot_queries_excluded": True,
        "primary_comparison": primary,
        "all_conditions_complete_and_hash_verified": True,
        "interpretation_rule": (
            "Use query-paired NDCG deltas at fixed model-token caps; report actual tokens "
            "and GPU time. Do not compare raw prompt counts across paradigms."
        ),
    }
    write_json(output_dir / "recommendation.json", recommendation)
    return {"summary": summary, "tests": tests, "recommendation": recommendation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["1", "2", "both"], default="both")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")

    outputs: dict[str, Any] = {}
    if args.experiment in {"1", "both"}:
        outputs["experiment_1"] = analyze_experiment(
            1,
            EXPERIMENT_1_METHODS,
            [("mohajer", "prp"), ("mohajer", "setwise"), ("mohajer", "listwise")],
            resamples=args.resamples,
            bootstrap_seed=args.bootstrap_seed,
        )["recommendation"]
    if args.experiment in {"2", "both"}:
        outputs["experiment_2"] = analyze_experiment(
            2,
            EXPERIMENT_2_METHODS,
            [
                ("mohajer_listwise", "listwise"),
                ("mohajer_setwise", "setwise"),
                ("mohajer_setwise", "mohajer"),
                ("mohajer_listwise", "mohajer"),
            ],
            resamples=args.resamples,
            bootstrap_seed=args.bootstrap_seed,
        )["recommendation"]

    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status.update({"status": "complete", "analysis_complete": True})
    write_json(STATUS_PATH, status)
    write_json(RESULTS_DIR / "analysis_manifest.json", outputs)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

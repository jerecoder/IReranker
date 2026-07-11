#!/usr/bin/env python3
"""Validate and aggregate the complete Robust04 cross-paradigm experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.robust04_cross_paradigm.common import (  # noqa: E402
    METRICS_DIR,
    PER_QUERY_DIR,
    RUNS_DIR,
    sha256,
    write_csv,
)


NUMERIC_COLUMNS = [
    "ndcg10",
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
]

COUNTER_COLUMNS = {
    "logical_comparisons",
    "choice_events",
    "prompt_instances",
    "generation_invocations",
    "document_instances",
    "encoder_nonpad_tokens",
    "encoder_padded_slots",
    "decoder_tokens",
    "total_model_tokens",
    "invalid_outputs",
    "inconsistent_outputs",
    "peak_gpu_memory_bytes",
}

VARIANTS = {
    "bm25": "top100",
    "mohajer": "sampling_shared_prp_prompt",
    "prp": "bidirectional_heapsort",
    "setwise": "heapsort_c3",
    "listwise": "rankgpt_w4_s2_r5",
}


def condition_name(method: str, budget: int | None, seed: int) -> str:
    if method == "bm25":
        return "bm25"
    return f"{method}_{VARIANTS[method]}_t{budget}_s{seed}"


def expected_condition_specs(
    token_budgets: Iterable[int], seeds: Iterable[int]
) -> dict[str, tuple[str, str, int | None, int]]:
    specs = {"bm25": ("bm25", VARIANTS["bm25"], None, 42)}
    for method in ("mohajer", "prp", "setwise", "listwise"):
        method_seeds = list(seeds) if method == "mohajer" else [42]
        for budget in token_budgets:
            for seed in method_seeds:
                name = condition_name(method, int(budget), int(seed))
                specs[name] = (method, VARIANTS[method], int(budget), int(seed))
    return specs


def _parse_numeric(raw: str | None, column: str, location: str) -> float:
    if raw is None or not str(raw).strip():
        raise ValueError(f"Missing {column} in {location}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Non-numeric {column}={raw!r} in {location}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column}={raw!r} in {location}")
    if column != "ndcg10" and value < 0:
        raise ValueError(f"Negative {column}={value} in {location}")
    if column in COUNTER_COLUMNS and not value.is_integer():
        raise ValueError(f"Non-integral counter {column}={value} in {location}")
    return value


def _validate_resource_invariants(row: dict[str, Any], location: str) -> None:
    if not 0.0 <= float(row["ndcg10"]) <= 1.0:
        raise ValueError(f"NDCG@10 outside [0, 1] in {location}")
    expected_tokens = float(row["encoder_nonpad_tokens"]) + float(row["decoder_tokens"])
    if float(row["total_model_tokens"]) != expected_tokens:
        raise ValueError(f"Token accounting mismatch in {location}")
    if float(row["encoder_padded_slots"]) < float(row["encoder_nonpad_tokens"]):
        raise ValueError(f"Padded encoder slots below non-padding tokens in {location}")
    prompts = float(row["prompt_instances"])
    documents = float(row["document_instances"])
    expected_ratio = documents / prompts if prompts else 0.0
    if not math.isclose(
        float(row["avg_documents_per_prompt"]), expected_ratio, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"Documents-per-prompt accounting mismatch in {location}")
    budget = row["token_budget"]
    if budget is not None and float(row["total_model_tokens"]) > budget:
        raise ValueError(f"Actual model tokens exceed budget in {location}")


def load_rows(
    *,
    token_budgets: list[int],
    seeds: list[int],
    expected_queries: int,
    allow_incomplete: bool = False,
    per_query_dir: Path = PER_QUERY_DIR,
    runs_dir: Path = RUNS_DIR,
) -> list[dict[str, Any]]:
    specs = expected_condition_specs(token_budgets, seeds)
    existing = {path.stem: path for path in per_query_dir.glob("*.csv")}
    if not existing:
        raise FileNotFoundError(f"No result CSVs found in {per_query_dir}")
    if allow_incomplete:
        paths = existing
    else:
        missing = sorted(set(specs) - set(existing))
        extra = sorted(set(existing) - set(specs))
        if missing or extra:
            raise ValueError(f"Condition set mismatch: missing={missing}, extra={extra}")
        paths = {name: existing[name] for name in specs}

    required = {
        "dataset",
        "condition",
        "method",
        "variant",
        "token_budget",
        "seed",
        "query_id",
        *NUMERIC_COLUMNS,
    }
    all_rows: list[dict[str, Any]] = []
    qids_by_condition: dict[str, set[str]] = {}
    base_signatures: dict[str, str] = {}

    for condition, path in sorted(paths.items()):
        done_path = per_query_dir / f"{condition}.done"
        marker: dict[str, Any] | None = None
        if not allow_incomplete:
            try:
                marker = json.loads(done_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid or missing completion marker: {condition}") from exc
            run_path = runs_dir / f"{condition}.txt"
            if (
                marker.get("status") != "complete"
                or marker.get("per_query_sha256") != sha256(path)
                or not run_path.exists()
                or marker.get("run_sha256") != sha256(run_path)
            ):
                raise ValueError(f"Completion marker/hash mismatch: {condition}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_columns = sorted(required - set(reader.fieldnames or []))
            if missing_columns:
                raise ValueError(f"Missing columns in {path}: {missing_columns}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty condition CSV: {path}")
        seen_qids: set[str] = set()
        parsed_rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(rows, start=2):
            location = f"{path}:{line_number}"
            qid = str(raw["query_id"]).strip()
            if not qid or qid in seen_qids:
                raise ValueError(f"Missing or duplicate query_id={qid!r} in {location}")
            seen_qids.add(qid)
            if str(raw["dataset"]).strip() != "robust04":
                raise ValueError(f"Unexpected dataset in {location}")
            if str(raw["condition"]).strip() != condition:
                raise ValueError(f"Condition/file mismatch in {location}")
            try:
                seed = int(raw["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid seed in {location}") from exc
            raw_budget = str(raw["token_budget"]).strip()
            budget = int(raw_budget) if raw_budget else None
            parsed: dict[str, Any] = dict(raw)
            parsed["seed"] = seed
            parsed["token_budget"] = budget
            parsed["query_id"] = qid
            for column in NUMERIC_COLUMNS:
                parsed[column] = _parse_numeric(raw.get(column), column, location)
            _validate_resource_invariants(parsed, location)

            if condition in specs:
                expected_method, expected_variant, expected_budget, expected_seed = specs[condition]
                actual = (
                    str(raw["method"]).strip(),
                    str(raw["variant"]).strip(),
                    budget,
                    seed,
                )
                if actual != (expected_method, expected_variant, expected_budget, expected_seed):
                    raise ValueError(f"Condition metadata mismatch in {location}: {actual}")
            parsed_rows.append(parsed)
        if not allow_incomplete and len(parsed_rows) != expected_queries:
            raise ValueError(
                f"Condition {condition} has {len(parsed_rows)} queries; expected {expected_queries}"
            )
        if marker is not None:
            signature = marker.get("signature")
            if not isinstance(signature, dict):
                raise ValueError(f"Missing run signature in completion marker: {condition}")
            ordered_qids = [str(row["query_id"]) for row in parsed_rows]
            qid_hash = hashlib.sha256("\n".join(ordered_qids).encode()).hexdigest()
            if (
                int(marker.get("rows", -1)) != len(parsed_rows)
                or signature.get("condition") != condition
                or int(signature.get("query_count", -1)) != len(parsed_rows)
                or signature.get("query_ids_sha256") != qid_hash
            ):
                raise ValueError(f"Completion signature/query mismatch: {condition}")
            condition_fields = {
                "condition",
                "method",
                "variant",
                "token_budget",
                "seed",
                "query_count",
                "query_ids_sha256",
            }
            base_signatures[condition] = json.dumps(
                {key: value for key, value in signature.items() if key not in condition_fields},
                sort_keys=True,
                separators=(",", ":"),
            )
        qids_by_condition[condition] = seen_qids
        all_rows.extend(parsed_rows)

    if not allow_incomplete:
        reference_name = next(iter(qids_by_condition))
        reference_qids = qids_by_condition[reference_name]
        mismatched = [
            name for name, qids in qids_by_condition.items() if qids != reference_qids
        ]
        if mismatched:
            raise ValueError(
                f"Query sets differ from {reference_name}: {sorted(mismatched)}"
            )
        if len(set(base_signatures.values())) != 1:
            raise ValueError("Completed conditions mix incompatible experiment configurations")
    return all_rows


def condition_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return str(row["method"]), str(row["variant"]), row["token_budget"]


def query_seed_average(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    return {
        qid: {
            column: float(np.mean([float(row[column]) for row in group]))
            for column in NUMERIC_COLUMNS
        }
        for qid, group in by_query.items()
    }


def bootstrap(
    values: np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap requires a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    n = values.size
    sampled = values[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(sampled, alpha / 2)),
        float(np.quantile(sampled, 1 - alpha / 2)),
    )


def sign_flip_p_value(
    differences: np.ndarray, *, seed: int = 42, resamples: int = 10000
) -> float:
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("sign-flip test requires paired differences")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, differences.size))
    null_statistics = np.abs((signs * differences).mean(axis=1))
    extreme = int(np.count_nonzero(null_statistics >= observed - 1e-15))
    return float((extreme + 1) / (resamples + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(sorted(range(total), key=lambda i: p_values[i])):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def mark_pareto(summary: list[dict[str, Any]]) -> None:
    for row in summary:
        for flag, cost_column in (
            ("pareto_tokens", "avg_total_model_tokens"),
            ("pareto_gpu_time", "avg_inference_seconds"),
        ):
            row[flag] = True
            for other in summary:
                if other is row:
                    continue
                no_more_cost = float(other[cost_column]) <= float(row[cost_column])
                no_less_quality = float(other["ndcg10"]) >= float(row["ndcg10"])
                strict = (
                    float(other[cost_column]) < float(row[cost_column])
                    or float(other["ndcg10"]) > float(row["ndcg10"])
                )
                if no_more_cost and no_less_quality and strict:
                    row[flag] = False
                    break


def arm_id(key: tuple[str, str, int | None]) -> str:
    method, variant, budget = key
    return f"{method}:{variant}:t{budget if budget is not None else 'none'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-budgets", type=int, nargs="+", default=[25000, 50000, 75000, 100000, 125000]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--expected-queries", type=int, default=249)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if any(value <= 0 for value in args.token_budgets) or len(set(args.token_budgets)) != len(args.token_budgets):
        parser.error("--token-budgets must contain unique positive integers")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.expected_queries <= 0 or args.resamples <= 0:
        parser.error("--expected-queries and --resamples must be positive")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between zero and one")

    rows = load_rows(
        token_budgets=args.token_budgets,
        seeds=args.seeds,
        expected_queries=args.expected_queries,
        allow_incomplete=args.allow_incomplete,
    )
    grouped: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(condition_key(row), []).append(row)

    query_values = {key: query_seed_average(group) for key, group in grouped.items()}
    seed_values = {
        key: sorted({int(row["seed"]) for row in group}) for key, group in grouped.items()
    }
    summary: list[dict[str, Any]] = []
    for key, per_query in query_values.items():
        method, variant, budget = key
        qids = sorted(per_query)
        ndcg_values = np.array([per_query[qid]["ndcg10"] for qid in qids], dtype=float)
        mean, low, high = bootstrap(
            ndcg_values, seed=args.bootstrap_seed, resamples=args.resamples, alpha=args.alpha
        )
        group_rows = grouped[key]
        output_row = {
            "arm_id": arm_id(key),
            "method": method,
            "variant": variant,
            "token_budget": budget if budget is not None else "",
            "queries": len(qids),
            "seeds": len(seed_values[key]),
            "seed_values": " ".join(str(value) for value in seed_values[key]),
            "ndcg10": mean,
            "ndcg10_ci_low": low,
            "ndcg10_ci_high": high,
            **{
                f"avg_{column}": float(np.mean([per_query[qid][column] for qid in qids]))
                for column in NUMERIC_COLUMNS
                if column not in {"ndcg10", "avg_documents_per_prompt"}
            },
            "macro_avg_documents_per_prompt": float(
                np.mean([per_query[qid]["avg_documents_per_prompt"] for qid in qids])
            ),
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.resamples,
            "ci_alpha": args.alpha,
        }
        total_prompts = sum(per_query[qid]["prompt_instances"] for qid in qids)
        total_documents = sum(per_query[qid]["document_instances"] for qid in qids)
        output_row["aggregate_documents_per_prompt"] = (
            total_documents / total_prompts if total_prompts else 0.0
        )
        for column in ("total_model_tokens", "inference_seconds", "query_wall_seconds"):
            values = np.array([per_query[qid][column] for qid in qids], dtype=float)
            _, metric_low, metric_high = bootstrap(
                values, seed=args.bootstrap_seed, resamples=args.resamples, alpha=args.alpha
            )
            output_row[f"avg_{column}_ci_low"] = metric_low
            output_row[f"avg_{column}_ci_high"] = metric_high
        summary.append(output_row)

    mark_pareto(summary)

    comparisons: list[dict[str, Any]] = []
    for key_a, key_b in combinations(sorted(query_values, key=str), 2):
        if key_a[2] != key_b[2] and None not in (key_a[2], key_b[2]):
            continue
        common_qids = sorted(set(query_values[key_a]) & set(query_values[key_b]))
        if not common_qids:
            continue
        diffs = np.array(
            [
                query_values[key_a][qid]["ndcg10"] - query_values[key_b][qid]["ndcg10"]
                for qid in common_qids
            ],
            dtype=float,
        )
        delta, low, high = bootstrap(
            diffs, seed=args.bootstrap_seed, resamples=args.resamples, alpha=args.alpha
        )
        comparisons.append({
            "arm_a": arm_id(key_a),
            "method_a": key_a[0],
            "variant_a": key_a[1],
            "token_budget_a": key_a[2] if key_a[2] is not None else "",
            "seeds_a": " ".join(str(value) for value in seed_values[key_a]),
            "arm_b": arm_id(key_b),
            "method_b": key_b[0],
            "variant_b": key_b[1],
            "token_budget_b": key_b[2] if key_b[2] is not None else "",
            "seeds_b": " ".join(str(value) for value in seed_values[key_b]),
            "shared_token_budget": key_a[2] if key_a[2] == key_b[2] else "",
            "queries": len(common_qids),
            "delta_ndcg10": delta,
            "ci_low": low,
            "ci_high": high,
            "p_value_raw": sign_flip_p_value(
                diffs, seed=args.bootstrap_seed, resamples=args.resamples
            ),
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.resamples,
            "ci_alpha": args.alpha,
        })

    adjusted = holm_adjust([float(row["p_value_raw"]) for row in comparisons])
    for row, adjusted_p in zip(comparisons, adjusted):
        row["p_value_holm"] = adjusted_p
        row["significant_holm05"] = adjusted_p < 0.05

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        METRICS_DIR / "summary.csv",
        sorted(summary, key=lambda row: (str(row["method"]), float(row["avg_total_model_tokens"]))),
    )
    write_csv(METRICS_DIR / "paired_bootstrap.csv", comparisons)
    print(f"Saved {METRICS_DIR / 'summary.csv'}")
    print(f"Saved {METRICS_DIR / 'paired_bootstrap.csv'}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for cost_column, label, filename in (
        ("avg_total_model_tokens", "Average model tokens per query", "ndcg_vs_tokens.png"),
        (
            "avg_inference_seconds",
            "Average synchronized GPU inference seconds per query",
            "ndcg_vs_gpu_time.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in summary:
            ax.scatter(
                float(row[cost_column]),
                float(row["ndcg10"]),
                label=f"{row['method']} {row['token_budget']}",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("NDCG@10")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(METRICS_DIR / filename, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()

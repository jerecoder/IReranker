#!/usr/bin/env python3
"""Summarize the exploratory screen without treating three-query pilots as confirmation."""

from __future__ import annotations

import csv
import json
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mohajer_hybrid_probe.common import (  # noqa: E402
    DATASET_ORDER,
    METRICS_DIR,
    MOHAJER_FAMILY,
    PER_QUERY_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    mean,
    pareto_methods,
    sha256,
    write_csv,
)


def load_completed() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for done_path in sorted(PER_QUERY_DIR.glob("*/*.done")):
        dataset = done_path.parent.name
        condition = done_path.stem
        csv_path = done_path.parent / f"{condition}.csv"
        run_path = RUNS_DIR / dataset / f"{condition}.txt"
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") != "complete"
            or marker.get("csv_sha256") != sha256(csv_path)
            or marker.get("run_sha256") != sha256(run_path)
        ):
            raise ValueError(f"Completion marker/hash mismatch: {dataset}/{condition}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["ndcg10"] = float(row["ndcg10"])
                row["total_model_tokens"] = float(row["total_model_tokens"])
                row["inference_seconds"] = float(row["inference_seconds"])
                row["query_wall_seconds"] = float(row["query_wall_seconds"])
                row["stage_a_tokens"] = float(row["stage_a_tokens"])
                row["stage_b_tokens"] = float(row["stage_b_tokens"])
                row["token_budget"] = int(row["token_budget"]) if row["token_budget"] else None
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No completed conditions under {PER_QUERY_DIR}")
    return rows


def main() -> None:
    rows = load_completed()
    grouped: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]), row["token_budget"])
        grouped.setdefault(key, []).append(row)

    bm25 = {
        dataset: {str(row["query_id"]): float(row["ndcg10"]) for row in group}
        for (dataset, method, budget), group in grouped.items()
        if method == "bm25" and budget is None
    }
    summary: list[dict[str, Any]] = []
    for (dataset, method, budget), group in grouped.items():
        values = {str(row["query_id"]): float(row["ndcg10"]) for row in group}
        baseline = bm25[dataset]
        qids = sorted(set(values) & set(baseline))
        summary.append({
            "dataset": dataset,
            "method": method,
            "token_budget": budget if budget is not None else "",
            "queries": len(qids),
            "seeds": len({row["seed"] for row in group}),
            "ndcg10": mean(values[qid] for qid in qids),
            "delta_vs_bm25": mean(values[qid] - baseline[qid] for qid in qids),
            "query_wins_vs_bm25": sum(values[qid] > baseline[qid] for qid in qids),
            "avg_tokens": mean(float(row["total_model_tokens"]) for row in group),
            "avg_stage_a_tokens": mean(float(row["stage_a_tokens"]) for row in group),
            "avg_stage_b_tokens": mean(float(row["stage_b_tokens"]) for row in group),
            "avg_gpu_seconds": mean(float(row["inference_seconds"]) for row in group),
            "avg_wall_seconds": mean(float(row["query_wall_seconds"]) for row in group),
        })

    for dataset in DATASET_ORDER:
        budgets = sorted(
            {
                int(row["token_budget"])
                for row in summary
                if row["dataset"] == dataset and row["token_budget"] != ""
            }
        )
        for budget in budgets:
            arms = [
                row for row in summary
                if row["dataset"] == dataset
                and (row["token_budget"] == budget or row["method"] == "bm25")
            ]
            frontier = pareto_methods(arms)
            for row in summary:
                if row["dataset"] == dataset and row["token_budget"] == budget:
                    row["pareto_tokens"] = row["method"] in frontier
        for row in summary:
            if row["dataset"] == dataset and row["method"] == "bm25":
                row["pareto_tokens"] = True

    # A small descriptive AUC over observed budget points. This is exploratory only.
    for row in summary:
        points = sorted(
            (
                float(other["avg_tokens"]),
                float(other["delta_vs_bm25"]),
            )
            for other in summary
            if other["dataset"] == row["dataset"]
            and other["method"] == row["method"]
            and other["token_budget"] != ""
        )
        auc = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            auc += (x1 - x0) * (y0 + y1) / 2
        row["delta_token_auc"] = auc if len(points) >= 2 else ""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        METRICS_DIR / "screen_summary.csv",
        sorted(
            summary,
            key=lambda row: (
                DATASET_ORDER.index(str(row["dataset"])),
                float(row["avg_tokens"]),
                str(row["method"]),
            ),
        ),
    )

    decisions_path = RESULTS_DIR / "screen_decisions.json"
    decisions = (
        json.loads(decisions_path.read_text(encoding="utf-8"))
        if decisions_path.exists() else []
    )
    recommendation = {
        "exploratory_only": True,
        "reason": "Three pilot queries and one seed are insufficient for confirmatory claims.",
        "first_qualifying_dataset": next(
            (
                row for row in decisions
                if row.get("decision") == "validate_on_fresh_queries"
            ),
            None,
        ),
        "next_step": (
            "Run the selected arm and all standalone controls on fresh hash-selected queries "
            "before spending full-dataset compute."
        ),
        "mohajer_family": sorted(MOHAJER_FAMILY),
    }
    (METRICS_DIR / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {METRICS_DIR / 'screen_summary.csv'}")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()

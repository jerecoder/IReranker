#!/usr/bin/env python3
"""Build CSVs, tables, and plots for the A1fp listwise/setwise experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "a1fp_listwise_setwise"
RUNS_DIR = EXP_DIR / "runs"
LOGS_DIR = EXP_DIR / "logs"
METRICS_DIR = EXP_DIR / "metrics"
OURS_SOURCE = ROOT / "reports" / "limit_comparisons_experiment.csv"

BASELINE_ROWS = [
    ("listwise", "dl19", "rankgpt_w4_s2_r1", RUNS_DIR / "listwise.rankgpt.flant5xl.w4s2.r1.dl19.txt", LOGS_DIR / "listwise.rankgpt.flant5xl.w4s2.r1.dl19.log"),
    ("listwise", "dl19", "rankgpt_w4_s2_r3", RUNS_DIR / "listwise.rankgpt.flant5xl.w4s2.r3.dl19.txt", LOGS_DIR / "listwise.rankgpt.flant5xl.w4s2.r3.dl19.log"),
    ("listwise", "dl19", "rankgpt_w4_s2_r5", RUNS_DIR / "listwise.rankgpt.flant5xl.w4s2.r5.dl19.txt", LOGS_DIR / "listwise.rankgpt.flant5xl.w4s2.r5.dl19.log"),
    ("listwise", "dl20", "rankgpt_w4_s2_r1", RUNS_DIR / "listwise.rankgpt.flant5xl.w4s2.r1.dl20.txt", LOGS_DIR / "listwise.rankgpt.flant5xl.w4s2.r1.dl20.log"),
    ("listwise", "dl20", "rankgpt_w4_s2_r3", RUNS_DIR / "listwise.rankgpt.flant5xl.w4s2.r3.dl20.txt", LOGS_DIR / "listwise.rankgpt.flant5xl.w4s2.r3.dl20.log"),
    ("listwise", "dl20", "rankgpt_w4_s2_r5", RUNS_DIR / "listwise.rankgpt.flant5xl.w4s2.r5.dl20.txt", LOGS_DIR / "listwise.rankgpt.flant5xl.w4s2.r5.dl20.log"),
    ("setwise", "dl19", "heapsort_c3_k10", RUNS_DIR / "setwise.heapsort.flant5xl.c3.dl19.txt", LOGS_DIR / "setwise.heapsort.flant5xl.c3.dl19.log"),
    ("setwise", "dl20", "heapsort_c3_k10", RUNS_DIR / "setwise.heapsort.flant5xl.c3.dl20.txt", LOGS_DIR / "setwise.heapsort.flant5xl.c3.dl20.log"),
]

OURS_FALLBACK = {
    ("dl19", 100): 61.41,
    ("dl19", 150): 67.79,
    ("dl19", 200): 68.70,
    ("dl19", 250): 68.73,
    ("dl19", 300): 68.73,
    ("dl20", 100): 59.95,
    ("dl20", 150): 65.06,
    ("dl20", 200): 67.66,
    ("dl20", 250): 67.62,
    ("dl20", 300): 67.62,
}

NUMERIC_BASELINE_COLUMNS = [
    "ndcg10",
    "avg_calls",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_total_tokens",
    "avg_time_sec",
]


def fmt(value: float | str | None, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return ""
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def average(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def parse_log(path: Path) -> dict[str, float | None]:
    result = {
        "avg_calls": None,
        "avg_prompt_tokens": None,
        "avg_completion_tokens": None,
        "avg_time_sec": None,
    }
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "avg_calls": r"Avg comparisons:\s*([0-9.eE+-]+)",
        "avg_prompt_tokens": r"Avg prompt tokens:\s*([0-9.eE+-]+)",
        "avg_completion_tokens": r"Avg completion tokens:\s*([0-9.eE+-]+)",
        "avg_time_sec": r"Avg time per query:\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def eval_ndcg(run_file: Path, dataset: str) -> float | None:
    if not run_file.exists():
        return None
    topic = "dl19-passage" if dataset == "dl19" else "dl20-passage"
    cmd = [
        sys.executable,
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        "-l",
        "2",
        "-m",
        "ndcg_cut.10",
        topic,
        str(run_file),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return None
    match = re.search(r"ndcg_cut_10\s+all\s+([0-9.]+)", proc.stdout)
    return float(match.group(1)) if match else None


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_baselines() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method, dataset, variant, run_file, log_file in BASELINE_ROWS:
        log_values = parse_log(log_file)
        prompt = log_values["avg_prompt_tokens"]
        completion = log_values["avg_completion_tokens"]
        total = None if prompt is None or completion is None else prompt + completion
        row = {
            "method": method,
            "dataset": dataset,
            "variant": variant,
            "ndcg10": fmt(eval_ndcg(run_file, dataset)),
            "avg_calls": fmt(log_values["avg_calls"]),
            "avg_prompt_tokens": fmt(prompt),
            "avg_completion_tokens": fmt(completion),
            "avg_total_tokens": fmt(total),
            "avg_time_sec": fmt(log_values["avg_time_sec"]),
        }
        rows.append(row)

    write_csv(
        METRICS_DIR / "baselines_summary.csv",
        rows,
        [
            "method",
            "dataset",
            "variant",
            "ndcg10",
            "avg_calls",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "avg_time_sec",
        ],
    )
    return rows


def build_baseline_avgs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["variant"]), []).append(row)

    avg_rows: list[dict[str, str]] = []
    for (method, variant), group in grouped.items():
        out = {"method": method, "variant": variant}
        for column in NUMERIC_BASELINE_COLUMNS:
            out[column.replace("ndcg10", "avg_ndcg10")] = fmt(
                average(maybe_float(row[column]) for row in group)
            )
        avg_rows.append(out)

    write_csv(
        METRICS_DIR / "baselines_summary_avg.csv",
        avg_rows,
        [
            "method",
            "variant",
            "avg_ndcg10",
            "avg_calls",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "avg_time_sec",
        ],
    )
    return avg_rows


def load_ours() -> list[dict[str, str]]:
    rows_by_key: dict[tuple[str, int], dict[str, str]] = {}
    if OURS_SOURCE.exists():
        with OURS_SOURCE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for source_row in reader:
                if source_row.get("Ranker") != "mohajer (ir)":
                    continue
                if source_row.get("Oracle") != "Sampling":
                    continue
                dataset = source_row.get("Dataset", "")
                if dataset not in {"dl-2019", "dl-2020"}:
                    continue
                budget = int(float(source_row.get("Budget", "0")))
                if budget not in {100, 150, 200, 250, 300}:
                    continue
                short_dataset = "dl19" if dataset == "dl-2019" else "dl20"
                ndcg = float(source_row["NDCG@10"]) * 100.0
                rows_by_key[(short_dataset, budget)] = {
                    "method": "mohajer",
                    "dataset": short_dataset,
                    "variant": "randomized_direction_oracle",
                    "budget_calls": str(budget),
                    "ndcg10": fmt(ndcg, digits=2),
                    "avg_calls": fmt(float(source_row["average_comparison_per_task"])),
                    "avg_prompt_tokens": "",
                    "avg_completion_tokens": "",
                    "avg_total_tokens": "",
                    "source": "reports/limit_comparisons_experiment.csv;tokens_unavailable",
                }

    for key, ndcg in OURS_FALLBACK.items():
        if key in rows_by_key:
            continue
        dataset, budget = key
        rows_by_key[key] = {
            "method": "mohajer",
            "dataset": dataset,
            "variant": "randomized_direction_oracle",
            "budget_calls": str(budget),
            "ndcg10": fmt(ndcg, digits=2),
            "avg_calls": str(budget),
            "avg_prompt_tokens": "",
            "avg_completion_tokens": "",
            "avg_total_tokens": "",
            "source": "paper_ndcg_calls_only",
        }

    rows = [rows_by_key[(dataset, budget)] for dataset in ("dl19", "dl20") for budget in (100, 150, 200, 250, 300)]
    write_csv(
        METRICS_DIR / "ours_existing.csv",
        rows,
        [
            "method",
            "dataset",
            "variant",
            "budget_calls",
            "ndcg10",
            "avg_calls",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "source",
        ],
    )
    return rows


def build_ours_avg(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["variant"], row["budget_calls"]), []).append(row)

    avg_rows: list[dict[str, str]] = []
    for (method, variant, budget), group in grouped.items():
        avg_rows.append(
            {
                "method": method,
                "variant": variant,
                "budget_calls": budget,
                "avg_ndcg10": fmt(average(maybe_float(row["ndcg10"]) for row in group), digits=2),
                "avg_calls": fmt(average(maybe_float(row["avg_calls"]) for row in group)),
                "avg_prompt_tokens": "",
                "avg_completion_tokens": "",
                "avg_total_tokens": "",
                "source": ";".join(sorted({row["source"] for row in group})),
            }
        )

    write_csv(
        METRICS_DIR / "ours_existing_avg.csv",
        avg_rows,
        [
            "method",
            "variant",
            "budget_calls",
            "avg_ndcg10",
            "avg_calls",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "source",
        ],
    )
    return avg_rows


def build_table(ours_avg: list[dict[str, str]], baseline_avg: list[dict[str, str]]) -> None:
    lines = [
        "| Method | Variant | Avg NDCG@10 | Avg calls | Avg prompt tokens | Avg completion tokens | Avg total tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(ours_avg, key=lambda r: int(r["budget_calls"])):
        lines.append(
            "| Mohajer randomized | "
            f"B={row['budget_calls']} | {row['avg_ndcg10']} | {row['avg_calls']} | "
            f"{row['avg_prompt_tokens']} | {row['avg_completion_tokens']} | {row['avg_total_tokens']} |"
        )

    labels = {
        ("listwise", "rankgpt_w4_s2_r1"): "RankGPT listwise w4/s2/r1",
        ("listwise", "rankgpt_w4_s2_r3"): "RankGPT listwise w4/s2/r3",
        ("listwise", "rankgpt_w4_s2_r5"): "RankGPT listwise w4/s2/r5",
        ("setwise", "heapsort_c3_k10"): "Setwise heapsort c=3/k=10",
    }
    for key, label in labels.items():
        match = next((row for row in baseline_avg if (row["method"], row["variant"]) == key), None)
        if match is None:
            continue
        ndcg = maybe_float(match["avg_ndcg10"])
        ndcg_percent = "" if ndcg is None else fmt(ndcg * 100.0, digits=2)
        lines.append(
            f"| {label} | - | {ndcg_percent} | {match['avg_calls']} | "
            f"{match['avg_prompt_tokens']} | {match['avg_completion_tokens']} | {match['avg_total_tokens']} |"
        )

    (METRICS_DIR / "a1fp_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_calls(ours_avg: list[dict[str, str]], baseline_avg: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for row in ours_avg:
        x = maybe_float(row["avg_calls"])
        y = maybe_float(row["avg_ndcg10"])
        if x is None or y is None:
            continue
        ax.scatter(x, y, color="#1f77b4", label="Mohajer randomized" if row is ours_avg[0] else None)
        ax.annotate(f"B={row['budget_calls']}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    baseline_colors = {"listwise": "#d62728", "setwise": "#2ca02c"}
    for row in baseline_avg:
        x = maybe_float(row["avg_calls"])
        y_raw = maybe_float(row["avg_ndcg10"])
        if x is None or y_raw is None:
            continue
        y = y_raw * 100.0
        label = f"{row['method']} baseline"
        ax.scatter(x, y, color=baseline_colors.get(row["method"], "#555555"), marker="s", label=label)
        ax.annotate(row["variant"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Avg raw LLM calls")
    ax.set_ylabel("Avg NDCG@10")
    ax.set_title("A1fp TREC DL2019/2020")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "a1fp_ndcg_vs_calls.png", dpi=200)
    fig.savefig(METRICS_DIR / "a1fp_ndcg_vs_calls.pdf")
    plt.close(fig)


def plot_tokens(ours_avg: list[dict[str, str]], baseline_avg: list[dict[str, str]]) -> bool:
    points: list[tuple[str, str, float, float]] = []
    for row in ours_avg:
        x = maybe_float(row["avg_total_tokens"])
        y = maybe_float(row["avg_ndcg10"])
        if x is not None and y is not None:
            points.append(("Mohajer randomized", f"B={row['budget_calls']}", x, y))
    for row in baseline_avg:
        x = maybe_float(row["avg_total_tokens"])
        y_raw = maybe_float(row["avg_ndcg10"])
        if x is not None and y_raw is not None:
            points.append((row["method"], row["variant"], x, y_raw * 100.0))

    if not points:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, label, x, y in points:
        ax.scatter(x, y, label=method)
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Avg total generated-model tokens")
    ax.set_ylabel("Avg NDCG@10")
    ax.set_title("A1fp TREC DL2019/2020")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "a1fp_ndcg_vs_tokens.png", dpi=200)
    fig.savefig(METRICS_DIR / "a1fp_ndcg_vs_tokens.pdf")
    plt.close(fig)
    return True


def write_caption(has_token_plot: bool) -> None:
    if has_token_plot:
        caption = (
            "NDCG@10 versus total generated-model tokens on TREC DL2019/2020 with Flan-T5-XL. "
            "Listwise uses RankGPT-style sliding-window permutation generation (w=4, s=2); "
            "setwise uses Zhuang et al.'s setwise.heapsort with c=3. Mohajer with the "
            "randomized-direction oracle is shown at fixed pairwise call budgets."
        )
    else:
        caption = (
            "NDCG@10 versus raw LLM calls on TREC DL2019/2020 with Flan-T5-XL. Since listwise "
            "and setwise calls contain multiple documents, raw calls are reported only as a "
            "secondary cost axis; token-normalized comparison is the fairer cross-paradigm "
            "cost metric. Baseline token points are filled only after the official llm-rankers "
            "generation runs complete."
        )
    (METRICS_DIR / "a1fp_caption.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows = build_baselines()
    baseline_avg = build_baseline_avgs(baseline_rows)
    ours_rows = load_ours()
    ours_avg = build_ours_avg(ours_rows)
    build_table(ours_avg, baseline_avg)
    plot_calls(ours_avg, baseline_avg)
    has_token_plot = plot_tokens(ours_avg, baseline_avg)
    write_caption(has_token_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

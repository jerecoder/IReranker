#!/usr/bin/env python3
"""Combine fresh Mohajer and llm-rankers results on token/compute axes."""

import csv
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/trec_covid_cross_paradigm"
METHODS = [
    ("pairwise", "prp_heapsort", "pairwise.prp.heapsort"),
    ("setwise", "heapsort_c3", "setwise.heapsort.c3"),
    ("listwise", "rankgpt_w4_s2_r1", "listwise.rankgpt.w4s2.r1"),
    ("listwise", "rankgpt_w4_s2_r3", "listwise.rankgpt.w4s2.r3"),
    ("listwise", "rankgpt_w4_s2_r5", "listwise.rankgpt.w4s2.r5"),
]


def metric(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.eE+-]+)", text)
    if not match:
        raise RuntimeError(f"Missing {label!r} in baseline log")
    return float(match.group(1))


def ndcg(run_path: Path) -> float:
    command = [
        sys.executable, "-m", "pyserini.eval.trec_eval", "-c", "-m", "ndcg_cut.10",
        "beir-v1.0.0-trec-covid-test", str(run_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(r"ndcg_cut_10\s+all\s+([0-9.]+)", result.stdout)
    if not match:
        raise RuntimeError(result.stdout)
    return float(match.group(1))


def main() -> None:
    rows = []
    ours_path = EXP / "metrics/ours.csv"
    with ours_path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))

    for method, variant, stem in METHODS:
        log_path = EXP / f"logs/{stem}.log"
        run_path = EXP / f"runs/{stem}.txt"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        prompt = metric(text, "Avg prompt tokens")
        completion = metric(text, "Avg completion tokens")
        rows.append({
            "method": method,
            "variant": variant,
            "budget": "",
            "ndcg10": ndcg(run_path),
            "avg_calls": metric(text, "Avg comparisons"),
            "avg_comparisons": metric(text, "Avg comparisons"),
            "avg_prompt_tokens": prompt,
            "avg_completion_tokens": completion,
            "avg_total_tokens": prompt + completion,
            "avg_time_sec": metric(text, "Avg time per query"),
            "total_comparisons": "",
        })

    fields = [
        "method", "variant", "budget", "ndcg10", "avg_calls", "avg_comparisons",
        "avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens",
        "avg_time_sec", "total_comparisons",
    ]
    output = EXP / "metrics/cross_paradigm_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Augment limit_comparisons_experiment.csv with latency-based estimates.

Adds 6 columns (GPU x model) at the end:
  estimated_ms_{gpu}_{model}
Each value = average_comparison_per_task * mean_ms (from inference_time_results*.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

GPU_KEYS = ("A100", "H100", "H200")
MODEL_KEYS = ("flan-t5-xl", "qwen-instruct")


def _gpu_key(name: str | None) -> str | None:
    if not name:
        return None
    upper = str(name).upper()
    if "A100" in upper:
        return "A100"
    if "H100" in upper:
        return "H100"
    if "H200" in upper:
        return "H200"
    return None


def _model_key(model: str | None) -> str | None:
    if not model:
        return None
    lower = str(model).lower()
    if "flan-t5-xl" in lower:
        return "flan-t5-xl"
    if "qwen" in lower and "instruct" in lower:
        return "qwen-instruct"
    return None


def _load_latency(latency_dir: Path) -> Dict[Tuple[str, str], float]:
    files = sorted(latency_dir.glob("inference_time_results*.csv"))
    if not files:
        raise FileNotFoundError(f"No inference_time_results*.csv files in {latency_dir}")

    frames = []
    for path in files:
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["gpu_key"] = df.get("gpu_name", "").apply(_gpu_key)
        df["model_key"] = df.get("model", "").apply(_model_key)
        df = df[df["gpu_key"].notna() & df["model_key"].notna()]
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No usable latency rows found after filtering.")

    merged = pd.concat(frames, ignore_index=True)
    grouped = (
        merged.groupby(["gpu_key", "model_key"], as_index=False)["mean_ms"].mean()
    )

    latency: Dict[Tuple[str, str], float] = {}
    for _, row in grouped.iterrows():
        latency[(row["gpu_key"], row["model_key"])] = float(row["mean_ms"])
    return latency


def _column_name(gpu: str, model: str) -> str:
    model_key = model.replace("-", "_")
    return f"estimated_ms_{gpu.lower()}_{model_key}"


def _ensure_latency_keys(latency: Dict[Tuple[str, str], float]) -> None:
    missing = []
    for gpu in GPU_KEYS:
        for model in MODEL_KEYS:
            if (gpu, model) not in latency:
                missing.append(f"{gpu}/{model}")
    if missing:
        raise KeyError("Missing latency entries for: " + ", ".join(missing))


def refine_csv(
    input_path: Path,
    latency_dir: Path,
    output_path: Path,
    *,
    round_digits: int | None = None,
) -> None:
    df = pd.read_csv(input_path)
    if "average_comparison_per_task" not in df.columns:
        raise KeyError("CSV is missing 'average_comparison_per_task' column.")

    latency = _load_latency(latency_dir)
    _ensure_latency_keys(latency)

    avg_comps = pd.to_numeric(df["average_comparison_per_task"], errors="coerce").fillna(0.0)
    new_cols = []
    for gpu in GPU_KEYS:
        for model in MODEL_KEYS:
            col = _column_name(gpu, model)
            df[col] = avg_comps * latency[(gpu, model)]
            if round_digits is not None:
                df[col] = df[col].round(round_digits)
            new_cols.append(col)

    ordered = [c for c in df.columns if c not in new_cols] + new_cols
    df = df[ordered]
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment limit_comparisons_experiment.csv with latency-based estimates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/limit_comparisons_experiment.csv"),
        help="Path to limit_comparisons_experiment.csv.",
    )
    parser.add_argument(
        "--latency-dir",
        type=Path,
        default=Path("data/external/latency"),
        help="Directory containing inference_time_results*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to overwriting --input).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=2,
        help="Round new columns to this many decimals (use -1 to skip).",
    )
    args = parser.parse_args()

    output_path = args.output or args.input
    round_digits = None if args.round < 0 else args.round
    refine_csv(args.input, args.latency_dir, output_path, round_digits=round_digits)
    print(f"Saved refined CSV to {output_path}")


if __name__ == "__main__":
    main()

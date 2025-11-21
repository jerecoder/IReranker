from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd
import typer

from ireranker.config import FIGURES_DIR, REPORTS_DIR

app = typer.Typer(help="Plot helpers for BEIR evaluation outputs.")


@app.command()
def ndcg_bar(
    dataset: str = typer.Argument(..., help="Dataset folder under reports/beir-metrics."),
    k: int = typer.Option(10, help="Cutoff level to plot."),
    summary_root: Path = typer.Option(
        REPORTS_DIR / "beir-metrics",
        "--summary-root",
        help="Root directory containing per-dataset summary.csv files.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to save the PNG (defaults to reports/figures/<dataset>_ndcg@<k>.png).",
    ),
) -> None:
    """Plot NDCG@k per ranker from a generated BEIR summary CSV."""
    summary_path = summary_root / dataset / "summary.csv"
    if output is None:
        output = FIGURES_DIR / f"{dataset}_ndcg@{k}.png"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv for dataset '{dataset}': {summary_path}")

    df = pd.read_csv(summary_path)
    if "ranker" not in df.columns or "NDCG" not in df.columns or "k" not in df.columns:
        raise ValueError(f"summary.csv missing expected columns: {summary_path}")

    filtered = df[df["k"] == k]
    if filtered.empty:
        raise ValueError(f"No rows for k={k} in {summary_path}")

    plot_data = (
        filtered[["ranker", "NDCG"]].set_index("ranker").sort_values("NDCG", ascending=True)
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for plotting; install with `pip install .[dev]`."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3 + 0.4 * len(plot_data)))
    plot_data.plot(kind="barh", legend=False, ax=ax, color="#1f77b4")
    ax.set_xlabel(f"NDCG@{k}")
    ax.set_ylabel("Ranker")
    ax.set_title(f"{dataset} - NDCG@{k} per ranker")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    logger.success(f"Saved plot to {output}")


if __name__ == "__main__":
    app()

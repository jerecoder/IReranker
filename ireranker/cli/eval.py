from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger
import typer

from ireranker.config import REPORTS_DIR
from ireranker.data.loaders import load_synthetic_dataset
from ireranker.evaluation import metrics as M
from ireranker.evaluation.reporting import save_per_query_csv, save_summary_csv
from ireranker.evaluation.runner import evaluate
from ireranker.rankers import get_ranker, list_rankers

app = typer.Typer(help="Evaluate rerankers over a dataset.")


@app.command()
def available():
    """List available rankers."""
    for name in list_rankers():
        typer.echo(name)


@app.command()
def run(
    rankers: List[str] = typer.Option(["identity", "reverse", "random"], help="Rankers to run"),
    k: Optional[int] = typer.Option(5, help="Cutoff for @k metrics (None = all)"),
    gains: str = typer.Option("exp", help="Gain function for DCG/NDCG: exp|linear"),
    out_dir: Path = typer.Option(REPORTS_DIR / "metrics", help="Output directory for reports"),
    synthetic_tasks: int = typer.Option(5, help="Number of synthetic tasks"),
    synthetic_candidates: int = typer.Option(8, help="Number of candidates per task"),
    seed: Optional[int] = typer.Option(123, help="Seed for random ranker"),
):
    ds = load_synthetic_dataset(n_tasks=synthetic_tasks, n_candidates=synthetic_candidates)
    rs = [get_ranker(r, seed=seed) for r in rankers]
    metrics = {
        "P@k": M.precision_at_k,
        "MAP": M.average_precision,
        "MRR": M.mrr,
        "NDCG": M.ndcg_at_k,
    }
    results = evaluate(rs, ds, metrics, k=k, gains=gains)
    save_summary_csv(results, out_dir / "summary.csv")
    save_per_query_csv(results, out_dir / "per_query")
    logger.success(f"Saved evaluation reports to {out_dir}")


def main():  # entry point convenience
    app()


if __name__ == "__main__":
    main()

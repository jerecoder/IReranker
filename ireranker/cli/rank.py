from __future__ import annotations

from typing import Optional

import typer

from ireranker.data.loaders import load_synthetic_dataset
from ireranker.rankers import get_ranker, list_rankers

app = typer.Typer(help="Run a single ranker on a small dataset.")


@app.command()
def list():
    for name in list_rankers():
        typer.echo(name)


@app.command()
def run(
    name: str = typer.Argument(..., help="Ranker name"),
    seed: Optional[int] = typer.Option(123, help="Random seed (if applicable)"),
):
    r = get_ranker(name, seed=seed)
    ds = load_synthetic_dataset(n_tasks=1, n_candidates=6)
    task = ds.tasks[0]
    idx = r.rank(task)
    typer.echo({"query_id": task.query_id, "ranking": idx})


def main():
    app()


if __name__ == "__main__":
    main()

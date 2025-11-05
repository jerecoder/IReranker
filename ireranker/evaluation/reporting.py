from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ireranker.evaluation.runner import EvalResult


def save_summary_csv(results: Dict[str, EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, res in results.items():
        row = {"ranker": name}
        row.update(res.summary)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def save_per_query_csv(results: Dict[str, EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        rows = []
        for qid, metrics in res.by_query.items():
            row = {"query_id": qid}
            row.update(metrics)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"{name}_per_query.csv", index=False)

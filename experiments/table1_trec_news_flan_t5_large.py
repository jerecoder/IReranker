#!/usr/bin/env python3
"""Run the Table 1 paired-bootstrap experiment on TREC-NEWS/FLAN-T5-Large.

This keeps the original rankers, oracles, budgets, seeds, NDCG cutoff, and
bootstrap settings. It changes only the dataset, matrix model, and output
directory. A precomputed TREC-NEWS test matrix whose path contains
``flan-t5-large`` must be available under data/external/reranking-matrices.
"""

from pathlib import Path

import table1_queries_pairs_ci as experiment


experiment.DATASETS = ["trec-news"]
experiment.DEFAULT_MODEL = "flan-t5-large"

experiment.OUT_DIR = Path(
    "reports/significance_testing/paired_bootstrap/table1_trec_news_flan_t5_large"
)
experiment.RAW_PATH = experiment.OUT_DIR / "raw_runs.csv"
experiment.QUERY_PATH = experiment.OUT_DIR / "query_ndcg.csv"
experiment.SIG_PATH = experiment.OUT_DIR / "paired_bootstrap_pairs.csv"


if __name__ == "__main__":
    experiment.main()

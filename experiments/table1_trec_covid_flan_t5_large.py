#!/usr/bin/env python3
"""Run the Table 1 paired-bootstrap experiment on TREC-COVID/FLAN-T5-Large.

Use ``--oracle-mode live-flan`` to compute FLAN-T5-Large comparisons on demand
with a persistent cache instead of reading a precomputed rerank matrix.
"""

from pathlib import Path

import table1_queries_pairs_ci as experiment


experiment.DATASETS = ["trec-covid"]
experiment.DEFAULT_MODEL = "flan-t5-large"

experiment.OUT_DIR = Path(
    "reports/significance_testing/paired_bootstrap/table1_trec_covid_flan_t5_large"
)
experiment.RAW_PATH = experiment.OUT_DIR / "raw_runs.csv"
experiment.QUERY_PATH = experiment.OUT_DIR / "query_ndcg.csv"
experiment.SIG_PATH = experiment.OUT_DIR / "paired_bootstrap_pairs.csv"


if __name__ == "__main__":
    experiment.main()

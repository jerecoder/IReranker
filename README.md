# IReranker

RAG reranking with IR.

## Structure

```
├─ LICENSE
├─ Makefile
├─ README.md
├─ pyproject.toml
├─ config/
│  ├─ beir_eval.json       # Config de evaluación (datasets, k_values, output_dir, etc.)
│  └─ beir_loader.json     # Config del loader BEIR (base_url, cache_subdir)
├─ data/
│  └─ external/            # Cache de datasets BEIR
├─ docs/
├─ notebooks/
├─ references/
├─ reports/
│  └─ beir-metrics/        # Salidas por dataset (CSV), o según config/output_dir
├─ ireranker/
│  ├─ __init__.py
│  ├─ config.py            # Rutas y logging
│  ├─ types.py             # RankingTask, RankingDataset
│  ├─ plots.py
│  ├─ run_beir_eval.py     # CLI/runner para evaluación BEIR
│  ├─ rankers/
│  │  ├─ Ranker.py
│  │  ├─ BubbleRanker.py
│  │  ├─ MohajerRanker.py
│  │  ├─ RandomRanker.py
│  │  └─ registry.py
│  ├─ data/
│  │  └─ loaders.py        # Loader BEIR
│  └─ evaluation/
│     └─ beir.py           # Evaluación con BEIR
└─ tests/
   ├─ test_beir_eval_module.py
   ├─ test_metrics.py
   └─ ...
```

## Installation with conda (recommended)

  - `make create_environment` (creates `IReranker` env with Python 3.10)
  - `conda activate IReranker`

Useful commands inside the `IReranker` env:
- Install project dependencies for development (lint/tests): `make requirements` (installs `.[dev]`)
- Lint/format: `make lint` / `make format`
- Tests: `make test`
- BEIR evaluation (uses configs): `make beir-eval`
- Runtime-only install (no dev tools): `pip install .`

## Configuration

- `config/beir_eval.json`
  - `datasets`: list of BEIR datasets to evaluate (canonical names)
  - `split`: BEIR split (e.g., "test")
  - `max_queries`: query cap (or null)
  - `light_exclude`: datasets to skip when running with `--light`
  - `seed`: RNG seed
  - `k_values`: evaluation cutoffs, e.g., [1,3,5,10,100]
  - `output_dir`: results destination (absolute or relative to REPORTS_DIR)
  - `rankers`: ["all"] or a list of names
  - Current example (`config/beir_eval.json`):

    ```json
    {
      "datasets": [
        "webis-touche2020",
        "nfcorpus",
        "trec-covid",
        "scifact",
        "fiqa",
        "dbpedia-entity"
      ],
      "split": "test",
      "max_queries": null,
      "seed": 69420,
      "k_values": [1, 3, 5, 10, 100],
      "light_exclude": ["dbpedia-entity", "fiqa"],
      "output_dir": "beir-metrics",
      "rankers": ["all"]
    }
    ```

- `config/beir_loader.json`
  - `base_url`: BEIR base URL
  - `cache_subdir`: sub-folder under `data/external/`

## Usage

- View/adjust datasets via `config/beir_eval.json`.

- Run evaluation (uses config):
  - `make beir-eval`
  - or `python -m ireranker.run_beir_eval`

- Overrides:
  - `make beir-eval ARGS="--dataset webis-touche2020"`
  - `make beir-eval ARGS="--light"` to skip datasets from `light_exclude`
  - `python -m ireranker.run_beir_eval --dataset trec-covid --max-queries 200`
  - `python -m ireranker.run_beir_eval --config /path/to/custom.json`

## Output

Per dataset: CSV `summary.csv` with one row per ranker & k
- Columns: `ranker,k,NDCG,MAP,Recall,Precision,Comparisons,NDCG_per_comp` (`NDCG_per_comp` is NDCG divided by comparisons)
- Directory: `reports/beir-metrics/<dataset>/` or `output_dir`

## Notes

- Existing dataset directories are reused (no re-download).
- After extraction the ZIP is removed to save space.
- Download errors are logged and the dataset creates an `ERROR.txt` so the run continues.
- `BubbleRanker` is implemented but intentionally not registered by default because it is expensive; import and instantiate it directly if you need it.


## BEIR results

Auto-updated after each BEIR evaluation:
- Average NDCG@10 per comparison for each ranker (scientific notation; higher is better).
- NDCG@10 per dataset/ranker grid. `n/a` when results are missing.

<!-- BEGIN_BEIR_RESULTS -->
| Ranker | Avg NDCG@10/Comparisons |
| --- | --- |
| mohajer | 1.931e-05 |
| random | 0.000e+00 |

| Dataset | mohajer | random |
| --- | --- | --- |
| dbpedia-entity | 0.5400 | 0.0950 |
| fiqa | 0.4248 | 0.0634 |
| nfcorpus | 0.6035 | 0.2596 |
| scifact | 0.6789 | 0.0440 |
| trec-covid | 0.7713 | 0.3915 |
| webis-touche2020 | 0.2763 | 0.1128 |
<!-- END_BEIR_RESULTS -->

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
├─ notebooks/
├─ references/
├─ reports/
│  └─ beir-metrics/        # Ejemplos de salidas por dataset (CSV); se regeneran
├─ ireranker/
│  ├─ __init__.py
│  ├─ config.py            # Rutas y logging
│  ├─ types.py             # RankingTask, RankingDataset
│  ├─ plots.py             # Plot de NDCG@k a partir de summary.csv
│  ├─ run_beir_eval.py     # CLI/runner para evaluación BEIR
│  ├─ rankers/
│  │  ├─ ranker.py
│  │  ├─ bubble_ranker.py
│  │  ├─ mohajer_ranker.py
│  │  ├─ random_ranker.py
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
  - `light_exclude`: datasets to skip when running with `--light`
  - `seed`: RNG seed
  - `k_values`: evaluation cutoffs, e.g., [1,3,5,10,100]
  - `output_dir`: results destination (absolute or relative to REPORTS_DIR)
  - `matrix_models`: list of rerank matrix model filters (matches rerank filename/path, e.g., `"flan-t5-large"`). Each model is evaluated separately and always gets its own output folder/README tables. If omitted, models are auto-discovered from the available rerank matrices.
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
      "seed": 42,
      "k_values": [1, 3, 5, 10, 100],
      "light_exclude": ["dbpedia-entity", "fiqa"],
      "output_dir": "beir-metrics",
      "matrix_models": ["flan-t5-large", "flan-t5-xl"],
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
  - `make beir-eval ARGS="--max-queries 20 --rankers random"` for a quick/safe run
  - `make beir-eval ARGS="--profile-out reports/profiles/beir_eval.prof --skip-readme-update"` to profile without touching docs
  - `make beir-eval ARGS="--matrix-models flan-t5-large,flan-t5-xl"` to evaluate specific rerank matrix models (one output folder/table per model); omit only if models are set in config or auto-discovery suffices
  - `python -m ireranker.run_beir_eval --config /path/to/custom.json`

## Output

Per dataset: CSV `summary.csv` with one row per ranker & k
- Columns: `ranker,k,NDCG,MAP,Recall,Precision,Comparisons,NDCG_per_comp` (`NDCG_per_comp` is NDCG divided by comparisons)
- Directory: always `reports/beir-metrics/<model>/<dataset>/` (or `<output_dir>/<model>/<dataset>/`) so results are separated per rerank model

## Notes

- Existing dataset directories are reused (no re-download).
- After extraction the ZIP is removed to save space.
- Download errors are logged and the dataset creates an `ERROR.txt` so the run continues.
- Low-memory runs: pass `--max-queries <N>` (and optionally `--dataset ...`/`--light`) to prune rerank matrices to that many queries and avoid loading multiple copies in RAM.
- `BubbleRanker` is implemented but intentionally not registered by default because it is expensive; import and instantiate it directly if you need it.
- `reports/beir-metrics/` contiene ejemplos de salidas; puedes borrarlas o regenerarlas con `make beir-eval`.
- Graficar NDCG@k desde un `summary.csv`: `python -m ireranker.plots ndcg-bar nfcorpus --k 10`.


## BEIR results

Auto-updated after each BEIR evaluation:
- Average NDCG@10 per comparison for each ranker (scientific notation; higher is better).
- NDCG@10 per dataset/ranker grid. `n/a` when results are missing.
- Tables are grouped by rerank matrix model (one section per model in `matrix_models` or auto-discovered models).

<!-- BEGIN_BEIR_RESULTS -->
### flan-t5-large
| Ranker | Avg NDCG@10/Comparisons |
| --- | --- |
| mohajer (ir) | 2.653e-05 |
| nothing | 0.000e+00 |
| quick sort (classic) | 1.078e-05 |
| random | 0.000e+00 |
| sliding window prp (classic) | 1.965e-05 |

| Dataset | mohajer (ir) | nothing | quick sort (classic) | random | sliding window prp (classic) |
| --- | --- | --- | --- | --- | --- |
| nfcorpus | 0.5272 | 0.5453 | 0.5384 | 0.2611 | 0.5922 |
| scifact | 0.5726 | 0.7302 | 0.6338 | 0.0436 | 0.7455 |
| trec-covid | 0.7555 | 0.6026 | 0.7650 | 0.4280 | 0.7311 |
| webis-touche2020 | 0.2517 | 0.4769 | 0.2392 | 0.0940 | 0.4805 |

### flan-t5-xl
| Ranker | Avg NDCG@10/Comparisons |
| --- | --- |
| mohajer (ir) | 2.814e-05 |
| nothing | 0.000e+00 |
| quick sort (classic) | 1.152e-05 |
| random | 0.000e+00 |
| sliding window prp (classic) | 1.776e-05 |

| Dataset | mohajer (ir) | nothing | quick sort (classic) | random | sliding window prp (classic) |
| --- | --- | --- | --- | --- | --- |
| nfcorpus | 0.5944 | 0.5453 | 0.5933 | 0.2611 | 0.6089 |
| scifact | 0.6450 | 0.7302 | 0.6568 | 0.0436 | 0.7679 |
| trec-covid | 0.7853 | 0.6026 | 0.7783 | 0.4280 | 0.7581 |
| webis-touche2020 | 0.2686 | 0.4769 | 0.2992 | 0.0940 | 0.4610 |
<!-- END_BEIR_RESULTS -->

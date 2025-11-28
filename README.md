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
- Single per-model table: NDCG@10 per dataset, a rightmost average column (excluding datasets flagged for missing comparisons), plus trailing columns for average #inference calls and average cache hits per ranker. `n/a` when results are missing.
- Tables are grouped by rerank matrix model (one section per model in `matrix_models` or auto-discovered models).
- Results are split by ranker + oracle (see `oracle` in `summary.csv`; headers include the oracle label).
- Dataset headers turn red (and include "missing comparisons") when the rerank matrix is missing at least one pairwise comparison for that dataset.


## BEIR results

Tables auto-updated after each BEIR evaluation.

<!-- BEGIN_BEIR_RESULTS -->
### flan-t5-large
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | 0.0910 | 0.3816 | 0.7302 | 0.6026 | 0.4769 | 0.4565 | **0** | **0** |
| mohajer (ir) [bidirectional] | 0.5494 | 0.4268 | 0.6003 | 0.7625 | 0.2338 | 0.5146 | 142824 | 14986 |
| mohajer (ir) [sampling] | 0.5428 | 0.4293 | 0.6173 | 0.7525 | 0.2515 | 0.5187 | 67405 | **0** |
| prp sort (classic) [bidirectional] | 0.5426 | 0.5199 | 0.7267 | 0.7667 | 0.3560 | 0.5824 | 351974 | 8001 |
| prp sort (classic) [sampling] | 0.5537 | 0.4643 | 0.6490 | 0.7527 | 0.3014 | 0.5442 | 245000 | **0** |
| quick sort (classic) [bidirectional] | **0.5755** | 0.4495 | 0.6469 | 0.7784 | 0.2937 | 0.5488 | 591455 | **0** |
| quick sort (classic) [sampling] | 0.5437 | 0.4289 | 0.6204 | 0.7438 | 0.2656 | 0.5205 | 159784 | **0** |
| sliding window prp (classic) [bidirectional] | 0.3779 | 0.4817 | **0.7455** | 0.7311 | **0.4805** | 0.5633 | 525359 | 172463 |
| sliding window prp (classic) [sampling] | 0.5363 | **0.5258** | 0.6971 | **0.7863** | 0.4052 | **0.5901** | 286243 | **0** |

### flan-t5-xl
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | n/a | n/a | 0.7302 | 0.6026 | **0.4769** | 0.6033 | **0** | **0** |
| mohajer (ir) [bidirectional] | n/a | n/a | 0.6695 | 0.7843 | 0.2714 | 0.5751 | 64961 | 6757 |
| mohajer (ir) [sampling] | n/a | n/a | 0.6653 | 0.7702 | 0.2784 | 0.5713 | 30968 | **0** |
| prp sort (classic) [bidirectional] | n/a | n/a | 0.7607 | **0.7959** | 0.3057 | 0.6208 | 153899 | 3174 |
| prp sort (classic) [sampling] | n/a | n/a | 0.7134 | 0.7836 | 0.2933 | 0.5968 | 109496 | **0** |
| quick sort (classic) [bidirectional] | n/a | n/a | 0.6615 | 0.7845 | 0.2788 | 0.5749 | 249733 | **0** |
| quick sort (classic) [sampling] | n/a | n/a | 0.6833 | 0.7763 | 0.2846 | 0.5814 | 73002 | **0** |
| sliding window prp (classic) [bidirectional] | n/a | n/a | **0.7679** | 0.7581 | 0.4610 | **0.6624** | 219095 | 70749 |
| sliding window prp (classic) [sampling] | n/a | n/a | 0.7487 | 0.7881 | 0.3669 | 0.6345 | 131557 | **0** |

Notes:
- fiqa: skipped (missing rerank matrix)
- dbpedia-entity: skipped (missing rerank matrix)
<!-- END_BEIR_RESULTS -->

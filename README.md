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
- Install light dev deps for lint/tests: `make requirements` or `make requirements-test` (installs `.[tests,lint]`)
- Install full stack (tests/lint/notebooks + BEIR extras): `make requirements-dev` (installs `.[dev,beir]`)
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

Tables auto-updated after each BEIR evaluation.

<!-- BEGIN_BEIR_RESULTS -->
### flan-t5-large
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | 0.1037 | 0.2361 | 0.6789 | 0.5947 | 0.4422 | 0.4111 | **0** | **0** |
| mohajer (ir) [bidirectional] | 0.2439 | 0.2509 | 0.5578 | 0.7506 | 0.2181 | 0.4043 | 112227 | 15277 |
| mohajer (ir) [sampling] | 0.2417 | 0.2619 | 0.5688 | 0.7601 | 0.2420 | 0.4149 | 67783 | **0** |
| prp sort (classic) [bidirectional] | 0.1830 | 0.3127 | 0.6747 | 0.7530 | 0.3321 | 0.4511 | 292132 | 6198 |
| prp sort (classic) [sampling] | 0.1884 | 0.2770 | 0.6174 | 0.7497 | 0.2993 | 0.4263 | 210097 | **0** |
| quick sort (classic) [bidirectional] | **0.3180** | 0.2684 | 0.6008 | **0.7661** | 0.2736 | 0.4454 | 755133 | **0** |
| quick sort (classic) [sampling] | 0.3043 | 0.2457 | 0.5873 | 0.7470 | 0.2506 | 0.4270 | 232551 | **0** |
| sliding window prp (classic) [bidirectional] | 0.1185 | 0.2945 | **0.6925** | 0.7177 | **0.4473** | **0.4541** | 135980 | 162085 |
| sliding window prp (classic) [sampling] | 0.1223 | **0.3138** | 0.6519 | 0.7597 | 0.3994 | 0.4494 | 284387 | **0** |

### flan-t5-xl
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | n/a | n/a | 0.6789 | 0.5947 | **0.4422** | 0.5719 | **0** | **0** |
| mohajer (ir) [bidirectional] | n/a | n/a | 0.6216 | 0.7696 | 0.2528 | 0.5480 | 51447 | 6757 |
| mohajer (ir) [sampling] | n/a | n/a | 0.6195 | 0.7629 | 0.2743 | 0.5522 | 30937 | **0** |
| prp sort (classic) [bidirectional] | n/a | n/a | 0.7057 | **0.7825** | 0.2843 | 0.5908 | 147551 | 3174 |
| prp sort (classic) [sampling] | n/a | n/a | 0.6598 | 0.7644 | 0.2824 | 0.5689 | 109384 | **0** |
| quick sort (classic) [bidirectional] | n/a | n/a | 0.6139 | 0.7717 | 0.2581 | 0.5479 | 249733 | **0** |
| quick sort (classic) [sampling] | n/a | n/a | 0.6353 | 0.7645 | 0.2657 | 0.5552 | 72960 | **0** |
| sliding window prp (classic) [bidirectional] | n/a | n/a | **0.7131** | 0.7482 | 0.4282 | **0.6298** | 77597 | 70749 |
| sliding window prp (classic) [sampling] | n/a | n/a | 0.6861 | 0.7791 | 0.3486 | 0.6046 | 131557 | **0** |

Notes:
- fiqa: skipped (missing rerank matrix)
- dbpedia-entity: skipped (missing rerank matrix)
<!-- END_BEIR_RESULTS -->

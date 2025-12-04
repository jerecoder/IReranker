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
| Ranker | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | **0.6925** | 0.7177 | **0.4473** | **0.6192** | 95248 | 145286 |
| bubble sort (classic) [cached-sampling] | 0.6747 | 0.7273 | 0.4249 | 0.6090 | 158547 | 466320 |
| bubble sort (classic) [sampling] | 0.6725 | 0.7473 | 0.4119 | 0.6106 | 528005 | **0** |
| bubble sort (classic) [weird $(1.5)$] | 0.6826 | 0.7320 | 0.4344 | 0.6164 | 376825 | **0** |
| mohajer (ir) [bidirectional] | 0.5578 | 0.7506 | 0.2181 | 0.5088 | 51447 | 6725 |
| mohajer (ir) [cached-sampling] | 0.5582 | 0.7350 | 0.2419 | 0.5117 | 48575 | 6545 |
| mohajer (ir) [sampling] | 0.5688 | 0.7601 | 0.2420 | 0.5236 | **30989** | **0** |
| mohajer (ir) [weird $(1.5)$] | 0.5790 | 0.7416 | 0.2410 | 0.5206 | 31687 | **0** |
| mohajer + bubble [bidirectional] | 0.5730 | 0.7399 | 0.2306 | 0.5145 | 96101 | 13839 |
| mohajer + bubble [cached-sampling] | 0.5818 | 0.7293 | 0.2535 | 0.5216 | 93723 | 28265 |
| mohajer + bubble [sampling] | 0.5991 | **0.7638** | 0.2607 | 0.5412 | 77353 | **0** |
| mohajer + bubble [weird $(1.5)$] | 0.5854 | 0.7594 | 0.2620 | 0.5356 | 71580 | **0** |

### flan-t5-xl
| Ranker | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | 0.7131 | 0.7482 | **0.4282** | **0.6298** | 87338 | 94329 |
| bubble sort (classic) [cached-sampling] | 0.7026 | 0.7638 | 0.3978 | 0.6214 | 145125 | 491235 |
| bubble sort (classic) [sampling] | 0.7019 | **0.7809** | 0.3453 | 0.6094 | 530596 | **0** |
| bubble sort (classic) [weird $(1.5)$] | **0.7182** | 0.7746 | 0.3957 | 0.6295 | 316458 | **0** |
| mohajer (ir) [bidirectional] | 0.6216 | 0.7696 | 0.2528 | 0.5480 | 51447 | 6757 |
| mohajer (ir) [cached-sampling] | 0.6105 | 0.7637 | 0.2565 | 0.5435 | 48637 | 6521 |
| mohajer (ir) [sampling] | 0.6195 | 0.7629 | 0.2743 | 0.5522 | **30937** | **0** |
| mohajer (ir) [weird $(1.5)$] | 0.6172 | 0.7734 | 0.2429 | 0.5445 | 31747 | **0** |
| mohajer + bubble [bidirectional] | 0.6125 | 0.7612 | 0.2448 | 0.5395 | 96079 | 14401 |
| mohajer + bubble [cached-sampling] | 0.6276 | 0.7758 | 0.2760 | 0.5598 | 93535 | 27864 |
| mohajer + bubble [sampling] | 0.6429 | 0.7635 | 0.2639 | 0.5568 | 77697 | **0** |
| mohajer + bubble [weird $(1.5)$] | 0.6364 | 0.7755 | 0.2467 | 0.5529 | 71188 | **0** |
<!-- END_BEIR_RESULTS -->


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
- Results are split by ranker + oracle (see `oracle` in `summary.csv`; headers include the oracle label).


## BEIR results

Tables auto-updated after each BEIR evaluation.


## BEIR results

Tables auto-updated after each BEIR evaluation.

<!-- BEGIN_BEIR_RESULTS -->
### flan-t5-large
| Ranker | Avg Comparisons |
| --- | --- |
| bm25 [bidirectional] | **0** |
| mohajer (ir) [bidirectional] | 68491 |
| mohajer (ir) [sampling] | 65376 |
| prp sort (classic) [bidirectional] | 293096 |
| prp sort (classic) [sampling] | 394880 |
| quick sort (classic) [bidirectional] | 699998 |
| quick sort (classic) [sampling] | 442089 |
| sliding window prp (classic) [bidirectional] | 139685 |
| sliding window prp (classic) [sampling] | 220570 |

| Ranker | dbpedia-entity NDCG@10 | dbpedia-entity Comparisons | fiqa NDCG@10 | fiqa Comparisons | nfcorpus NDCG@10 | nfcorpus Comparisons | scifact NDCG@10 | scifact Comparisons | trec-covid NDCG@10 | trec-covid Comparisons | webis-touche2020 NDCG@10 | webis-touche2020 Comparisons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | 0.1037 | 0 | 0.2361 | 0 | 0.3982 | 0 | 0.6789 | 0 | 0.5947 | 0 | 0.4422 | 0 |
| mohajer (ir) [bidirectional] | 0.2461 | 97263 | 0.2400 | 160717 | 0.4884 | 56994 | 0.5537 | 71993 | 0.7439 | 12032 | 0.2417 | 11949 |
| mohajer (ir) [sampling] | 0.2503 | 94607 | 0.2452 | 150555 | 0.4860 | 55228 | 0.5323 | 68986 | 0.7427 | 11561 | 0.2332 | 11321 |
| prp sort (classic) [bidirectional] | 0.1830 | 296216 | **0.3127** | 644742 | **0.4997** | 297916 | 0.6747 | 383026 | 0.7530 | 71932 | 0.3321 | 64744 |
| prp sort (classic) [sampling] | 0.1887 | 345172 | 0.2771 | 997204 | 0.4852 | 367870 | 0.6299 | 490922 | 0.7631 | 86402 | 0.2773 | 81708 |
| quick sort (classic) [bidirectional] | **0.3180** | 1621662 | 0.2684 | 1542130 | 0.4167 | 424322 | 0.6008 | 442690 | **0.7661** | 80198 | 0.2736 | 88986 |
| quick sort (classic) [sampling] | 0.3042 | 1176752 | 0.2537 | 701522 | 0.4222 | 328186 | 0.5763 | 333424 | 0.7303 | 57708 | 0.2460 | 54940 |
| sliding window prp (classic) [bidirectional] | 0.1185 | 117738 | 0.2945 | 307414 | 0.4212 | 158212 | **0.6925** | 177982 | 0.7177 | 44096 | **0.4473** | 32670 |
| sliding window prp (classic) [sampling] | 0.1229 | 152024 | 0.3056 | 553216 | 0.4134 | 227802 | 0.6810 | 280716 | 0.7372 | 60228 | 0.4396 | 49436 |

### flan-t5-xl
| Ranker | Avg Comparisons |
| --- | --- |
| bm25 [bidirectional] | **0** |
| mohajer (ir) [bidirectional] | 38182 |
| mohajer (ir) [sampling] | 36764 |
| prp sort (classic) [bidirectional] | 169888 |
| prp sort (classic) [sampling] | 241248 |
| quick sort (classic) [bidirectional] | 318460 |
| quick sort (classic) [sampling] | 189291 |
| sliding window prp (classic) [bidirectional] | 89108 |
| sliding window prp (classic) [sampling] | 140215 |

| Ranker | dbpedia-entity NDCG@10 | dbpedia-entity Comparisons | fiqa NDCG@10 | fiqa Comparisons | nfcorpus NDCG@10 | nfcorpus Comparisons | scifact NDCG@10 | scifact Comparisons | trec-covid NDCG@10 | trec-covid Comparisons | webis-touche2020 NDCG@10 | webis-touche2020 Comparisons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | n/a | n/a | n/a | n/a | 0.3982 | 0 | 0.6789 | 0 | 0.5947 | 0 | **0.4422** | 0 |
| mohajer (ir) [bidirectional] | n/a | n/a | n/a | n/a | 0.5161 | 57071 | 0.6201 | 72038 | 0.7514 | 11960 | 0.2721 | 11661 |
| mohajer (ir) [sampling] | n/a | n/a | n/a | n/a | 0.5148 | 55065 | 0.5974 | 69122 | 0.7727 | 11567 | 0.2510 | 11303 |
| prp sort (classic) [bidirectional] | n/a | n/a | n/a | n/a | **0.5185** | 236900 | 0.7057 | 294252 | **0.7825** | 74434 | 0.2843 | 73966 |
| prp sort (classic) [sampling] | n/a | n/a | n/a | n/a | 0.5093 | 341342 | 0.6580 | 451204 | 0.7644 | 87290 | 0.2579 | 85156 |
| quick sort (classic) [bidirectional] | n/a | n/a | n/a | n/a | 0.4570 | 524638 | 0.6139 | 605524 | 0.7717 | 74088 | 0.2581 | 69588 |
| quick sort (classic) [sampling] | n/a | n/a | n/a | n/a | 0.4502 | 319154 | 0.6344 | 322394 | 0.7629 | 58346 | 0.2656 | 57270 |
| sliding window prp (classic) [bidirectional] | n/a | n/a | n/a | n/a | 0.4300 | 123640 | **0.7131** | 135484 | 0.7482 | 51366 | 0.4282 | 45940 |
| sliding window prp (classic) [sampling] | n/a | n/a | n/a | n/a | 0.4331 | 197184 | 0.7001 | 239078 | 0.7594 | 64578 | 0.3951 | 60020 |
<!-- END_BEIR_RESULTS -->

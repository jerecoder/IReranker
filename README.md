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

### flan-t5-large
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | 0.1037 | 0.2361 | 0.6789 | 0.5947 | 0.4422 | 0.4111 | **0** | **0** |
| bubble sort (classic) [bidirectional] | 0.1185 | 0.2945 | **0.6925** | 0.7177 | **0.4473** | 0.4541 | 144780 | 192167 |
| bubble sort (classic) [cached-sampling] | 0.1190 | 0.2970 | 0.6747 | 0.7273 | 0.4249 | 0.4486 | 247678 | 1014767 |
| bubble sort (classic) [sampling] | 0.1211 | **0.3231** | 0.6725 | 0.7473 | 0.4119 | 0.4552 | 995151 | **0** |
| bubble sort (classic) [weird $(1.5)$] | 0.1217 | 0.3179 | 0.6826 | 0.7320 | 0.4344 | **0.4577** | 598184 | **0** |
| mohajer (ir) [bidirectional] | 0.2391 | 0.2587 | 0.5385 | 0.7574 | 0.2641 | 0.4116 | 113824 | 14300 |
| mohajer (ir) [cached-sampling] | 0.2396 | 0.2492 | 0.5799 | 0.7611 | 0.2335 | 0.4127 | 108393 | 13245 |
| mohajer (ir) [sampling] | 0.2412 | 0.2485 | 0.5754 | 0.7611 | 0.2445 | 0.4141 | 67660 | **0** |
| mohajer (ir) [weird $(1.5)$] | 0.2419 | 0.2520 | 0.5648 | 0.7693 | 0.2558 | 0.4168 | 69352 | **0** |
| mohajer + bubble [bidirectional] | 0.2411 | 0.2623 | 0.5487 | 0.7556 | 0.2665 | 0.4149 | 120218 | 22681 |
| mohajer + bubble [cached-sampling] | 0.2442 | 0.2683 | 0.5694 | 0.7437 | 0.2390 | 0.4129 | 122975 | 36135 |
| mohajer + bubble [sampling] | 0.2457 | 0.2619 | 0.5880 | **0.7708** | 0.2570 | 0.4247 | 99555 | **0** |
| mohajer + bubble [weird $(1.5)$] | 0.2453 | 0.2770 | 0.5746 | 0.7674 | 0.2522 | 0.4233 | 90619 | **0** |
| prp sort (classic) [bidirectional] | 0.1830 | 0.3127 | 0.6747 | 0.7530 | 0.3321 | 0.4511 | 292132 | 6198 |
| prp sort (classic) [cached-sampling] | 0.1879 | 0.2827 | 0.6233 | 0.7589 | 0.2949 | 0.4295 | 400552 | 11179 |
| prp sort (classic) [sampling] | 0.1884 | 0.2770 | 0.6174 | 0.7497 | 0.2993 | 0.4263 | 210097 | **0** |
| prp sort (classic) [weird $(1.5)$] | 0.1848 | 0.2981 | 0.6411 | 0.7421 | 0.3019 | 0.4336 | 183841 | **0** |
| quick sort (classic) [bidirectional] | **0.3180** | 0.2684 | 0.6008 | 0.7661 | 0.2736 | 0.4454 | 755133 | **0** |
| quick sort (classic) [cached-sampling] | 0.3043 | 0.2457 | 0.5873 | 0.7470 | 0.2506 | 0.4270 | 465101 | **0** |
| quick sort (classic) [sampling] | 0.3043 | 0.2457 | 0.5873 | 0.7470 | 0.2506 | 0.4270 | 232551 | **0** |
| quick sort (classic) [weird $(1.5)$] | 0.3101 | 0.2578 | 0.5752 | 0.7458 | 0.2442 | 0.4266 | 249228 | **0** |
| sliding window prp (classic) [bidirectional] | 0.1185 | 0.2945 | **0.6925** | 0.7177 | **0.4473** | 0.4541 | 135980 | 162085 |
| sliding window prp (classic) [cached-sampling] | 0.1208 | 0.3021 | 0.6920 | 0.7421 | 0.4256 | 0.4565 | 218071 | 174876 |
| sliding window prp (classic) [sampling] | 0.1223 | 0.3138 | 0.6519 | 0.7597 | 0.3994 | 0.4494 | 284387 | **0** |
| sliding window prp (classic) [weird $(1.5)$] | 0.1214 | 0.3127 | 0.6652 | 0.7522 | 0.4147 | 0.4533 | 281239 | **0** |

### flan-t5-xl
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 [bidirectional] | n/a | n/a | 0.6789 | 0.5947 | **0.4422** | 0.5719 | **0** | **0** |
| bubble sort (classic) [bidirectional] | n/a | n/a | 0.7131 | 0.7482 | 0.4282 | **0.6298** | 87338 | 94329 |
| bubble sort (classic) [cached-sampling] | n/a | n/a | 0.7026 | 0.7638 | 0.3978 | 0.6214 | 145125 | 491235 |
| bubble sort (classic) [sampling] | n/a | n/a | 0.7019 | 0.7809 | 0.3453 | 0.6094 | 530596 | **0** |
| bubble sort (classic) [weird $(1.5)$] | n/a | n/a | **0.7182** | 0.7746 | 0.3957 | 0.6295 | 316458 | **0** |
| mohajer (ir) [bidirectional] | n/a | n/a | 0.6122 | 0.7598 | 0.2545 | 0.5422 | 53775 | 5537 |
| mohajer (ir) [cached-sampling] | n/a | n/a | 0.6181 | 0.7659 | 0.2756 | 0.5532 | 51028 | 5282 |
| mohajer (ir) [sampling] | n/a | n/a | 0.6276 | 0.7761 | 0.2720 | 0.5586 | 30891 | **0** |
| mohajer (ir) [weird $(1.5)$] | n/a | n/a | 0.6227 | 0.7708 | 0.2735 | 0.5556 | 31599 | **0** |
| mohajer + bubble [bidirectional] | n/a | n/a | 0.6188 | 0.7598 | 0.2571 | 0.5452 | 57175 | 9527 |
| mohajer + bubble [cached-sampling] | n/a | n/a | 0.6107 | 0.7663 | 0.2772 | 0.5514 | 57912 | 16439 |
| mohajer + bubble [sampling] | n/a | n/a | 0.6354 | **0.7855** | 0.2794 | 0.5668 | 46831 | **0** |
| mohajer + bubble [weird $(1.5)$] | n/a | n/a | 0.6411 | 0.7776 | 0.2653 | 0.5613 | 42171 | **0** |
| prp sort (classic) [bidirectional] | n/a | n/a | 0.7057 | 0.7825 | 0.2843 | 0.5908 | 147551 | 3174 |
| prp sort (classic) [cached-sampling] | n/a | n/a | 0.6627 | 0.7822 | 0.2732 | 0.5727 | 207726 | 6170 |
| prp sort (classic) [sampling] | n/a | n/a | 0.6598 | 0.7644 | 0.2824 | 0.5689 | 109384 | **0** |
| prp sort (classic) [weird $(1.5)$] | n/a | n/a | 0.6861 | 0.7794 | 0.3037 | 0.5897 | 94922 | **0** |
| quick sort (classic) [bidirectional] | n/a | n/a | 0.6139 | 0.7717 | 0.2581 | 0.5479 | 249733 | **0** |
| quick sort (classic) [cached-sampling] | n/a | n/a | 0.6353 | 0.7645 | 0.2657 | 0.5552 | 145921 | **0** |
| quick sort (classic) [sampling] | n/a | n/a | 0.6353 | 0.7645 | 0.2657 | 0.5552 | 72960 | **0** |
| quick sort (classic) [weird $(1.5)$] | n/a | n/a | 0.6113 | 0.7572 | 0.2699 | 0.5461 | 76506 | **0** |
| sliding window prp (classic) [bidirectional] | n/a | n/a | 0.7131 | 0.7482 | 0.4282 | **0.6298** | 77597 | 70749 |
| sliding window prp (classic) [cached-sampling] | n/a | n/a | 0.7094 | 0.7662 | 0.3907 | 0.6221 | 121327 | 70893 |
| sliding window prp (classic) [sampling] | n/a | n/a | 0.6861 | 0.7791 | 0.3486 | 0.6046 | 131557 | **0** |
| sliding window prp (classic) [weird $(1.5)$] | n/a | n/a | 0.6942 | 0.7638 | 0.3898 | 0.6159 | 131557 | **0** |

## Limit Comparisons Experiment

### trec-covid
| Ranker | 50 | 100 | 150 | 200 | 250 | 300 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.5947 | 0.5947 | 0.5947 | 0.6380 | 0.6380 | 0.6411 | 0.6806 |
| Bubble Sort (Classic) [Sampling] | 0.5947 | 0.6350 | 0.6454 | 0.6767 | 0.6668 | 0.7054 | 0.7203 |
| Mohajer (IR) [Bidirectional] | 0.4898 | 0.4898 | 0.4898 | **0.7359** | 0.7285 | 0.7333 | 0.7574 |
| Mohajer (IR) [Sampling] | 0.4898 | **0.7297** | **0.7503** | 0.7328 | **0.7611** | 0.7611 | 0.7611 |
| Mohajer + Bubble [Bidirectional] | 0.4898 | 0.4898 | 0.4898 | **0.7359** | 0.7285 | 0.7333 | 0.7556 |
| Mohajer + Bubble [Sampling] | 0.4898 | **0.7297** | **0.7503** | 0.7328 | 0.7577 | **0.7633** | **0.7708** |
| PAC + Bubble [Bidirectional] | 0.5947 | 0.5947 | 0.5947 | 0.5947 | 0.5947 | 0.6520 | 0.6729 |
| PAC + Bubble [Sampling] | 0.5947 | 0.5947 | 0.6786 | 0.7002 | 0.7002 | 0.7002 | 0.7002 |
| Quick Sort (Classic) [Bidirectional] | **0.6471** | 0.6602 | 0.6705 | 0.6752 | 0.6722 | 0.6744 | 0.6897 |
| Quick Sort (Classic) [Sampling] | 0.6338 | 0.6466 | 0.6446 | 0.6575 | 0.6649 | 0.6791 | 0.7167 |

### dbpedia-entity
| Ranker | 50 | 100 | 150 | 200 | 250 | 300 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.1037 | 0.1037 | 0.1037 | 0.1134 | 0.1138 | 0.1157 | 0.1178 |
| Bubble Sort (Classic) [Sampling] | 0.1037 | 0.1148 | 0.1144 | 0.1172 | 0.1184 | 0.1190 | 0.1211 |
| Mohajer (IR) [Bidirectional] | 0.1157 | 0.1157 | 0.1157 | 0.2097 | 0.2198 | 0.2333 | 0.2391 |
| Mohajer (IR) [Sampling] | 0.1152 | **0.2173** | **0.2327** | 0.2371 | 0.2431 | 0.2412 | 0.2412 |
| Mohajer + Bubble [Bidirectional] | 0.1157 | 0.1157 | 0.1157 | 0.2097 | 0.2198 | 0.2333 | 0.2411 |
| Mohajer + Bubble [Sampling] | 0.1152 | **0.2173** | **0.2327** | **0.2374** | **0.2445** | **0.2456** | **0.2457** |
| PAC + Bubble [Bidirectional] | 0.1037 | 0.1037 | 0.1037 | 0.1037 | 0.1037 | 0.1242 | 0.1495 |
| PAC + Bubble [Sampling] | 0.1037 | 0.1037 | 0.1208 | 0.1530 | 0.1530 | 0.1530 | 0.1530 |
| Quick Sort (Classic) [Bidirectional] | **0.1302** | 0.1492 | 0.1663 | 0.1877 | 0.1972 | 0.2022 | 0.2147 |
| Quick Sort (Classic) [Sampling] | 0.1240 | 0.1447 | 0.1514 | 0.1606 | 0.1563 | 0.1577 | 0.1718 |

### fiqa
| Ranker | 50 | 100 | 150 | 200 | 250 | 300 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.2361 | 0.2361 | 0.2361 | 0.2802 | 0.2803 | 0.2826 | 0.2920 |
| Bubble Sort (Classic) [Sampling] | 0.2361 | **0.2912** | **0.2795** | **0.2892** | **0.2956** | **0.3035** | **0.3116** |
| Mohajer (IR) [Bidirectional] | 0.1531 | 0.1531 | 0.1531 | 0.2515 | 0.2594 | 0.2531 | 0.2587 |
| Mohajer (IR) [Sampling] | 0.1531 | 0.2085 | 0.2390 | 0.2502 | 0.2485 | 0.2485 | 0.2485 |
| Mohajer + Bubble [Bidirectional] | 0.1531 | 0.1531 | 0.1531 | 0.2515 | 0.2594 | 0.2531 | 0.2623 |
| Mohajer + Bubble [Sampling] | 0.1531 | 0.2085 | 0.2390 | 0.2502 | 0.2557 | 0.2657 | 0.2619 |
| PAC + Bubble [Bidirectional] | 0.2361 | 0.2361 | 0.2361 | 0.2361 | 0.2361 | 0.2642 | 0.2920 |
| PAC + Bubble [Sampling] | 0.2361 | 0.2361 | 0.2528 | 0.2719 | 0.2719 | 0.2719 | 0.2719 |
| Quick Sort (Classic) [Bidirectional] | **0.2635** | 0.2663 | 0.2665 | 0.2647 | 0.2657 | 0.2681 | 0.2676 |
| Quick Sort (Classic) [Sampling] | 0.2415 | 0.2408 | 0.2483 | 0.2372 | 0.2506 | 0.2452 | 0.2437 |

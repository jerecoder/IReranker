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
| bubble sort (classic) [weird $(1.5)$] | 0.1217 | 0.3179 | 0.6826 | 0.7320 | 0.4344 | **0.4577** | 598184 | **0** |
| sliding window prp (classic) [cached-sampling] | 0.1208 | 0.3021 | 0.6920 | 0.7421 | 0.4256 | 0.4565 | 218071 | 174876 |
| bubble sort (classic) [sampling] | 0.1211 | **0.3231** | 0.6725 | 0.7473 | 0.4119 | 0.4552 | 995151 | **0** |
| pac top-k (2-round) [bidirectional] | 0.2759 | 0.2921 | 0.6664 | 0.7403 | 0.2977 | 0.4545 | 589811 | 520 |
| bubble sort (classic) [bidirectional] | 0.1185 | 0.2945 | **0.6925** | 0.7177 | **0.4473** | 0.4541 | 144780 | 192167 |
| sliding window prp (classic) [bidirectional] | 0.1185 | 0.2945 | **0.6925** | 0.7177 | **0.4473** | 0.4541 | 135980 | 162085 |
| sliding window prp (classic) [weird $(1.5)$] | 0.1214 | 0.3127 | 0.6652 | 0.7522 | 0.4147 | 0.4533 | 281239 | **0** |
| prp sort (classic) [bidirectional] | 0.1830 | 0.3127 | 0.6747 | 0.7530 | 0.3321 | 0.4511 | 292132 | 6198 |
| pac top-k (3-round) [bidirectional] | 0.2337 | 0.2877 | 0.6589 | 0.7378 | 0.3311 | 0.4498 | 649122 | 38213 |
| sliding window prp (classic) [sampling] | 0.1223 | 0.3138 | 0.6519 | 0.7597 | 0.3994 | 0.4494 | 284387 | **0** |
| bubble sort (classic) [cached-sampling] | 0.1190 | 0.2970 | 0.6747 | 0.7273 | 0.4249 | 0.4486 | 247678 | 1014767 |
| pac + bubble [weird $(1.5)$] | 0.1515 | 0.2762 | 0.6781 | 0.6889 | 0.4327 | 0.4455 | 50648 | **0** |
| quick sort (classic) [bidirectional] | **0.3180** | 0.2684 | 0.6008 | 0.7661 | 0.2736 | 0.4454 | 755133 | **0** |
| pac + bubble [bidirectional] | 0.1495 | 0.2920 | 0.6845 | 0.6729 | 0.4143 | 0.4426 | 93239 | 1954 |
| pac top-k (2-round) [cached-sampling] | 0.2538 | 0.2461 | 0.6444 | 0.7620 | 0.2944 | 0.4401 | 657860 | 2046 |
| pac top-k (2-round) [weird $(1.5)$] | 0.2526 | 0.2406 | 0.6399 | 0.7483 | 0.3126 | 0.4388 | 306259 | **0** |
| pac top-k (3-round) [sampling] | 0.1953 | 0.2511 | 0.6594 | 0.7219 | 0.3550 | 0.4365 | 347574 | **0** |
| pac + bubble [sampling] | 0.1530 | 0.2719 | 0.6699 | 0.7002 | 0.3823 | 0.4355 | 52867 | **0** |
| spectral mle (refinement) [weird $(1.5)$] | 0.2197 | 0.3006 | 0.6242 | 0.7209 | 0.3114 | 0.4353 | 726208 | **0** |
| pac top-k (2-round) [sampling] | 0.2554 | 0.2630 | 0.6247 | 0.7408 | 0.2880 | 0.4344 | 330719 | **0** |
| prp sort (classic) [weird $(1.5)$] | 0.1848 | 0.2981 | 0.6411 | 0.7421 | 0.3019 | 0.4336 | 183841 | **0** |
| prp sort (classic) [cached-sampling] | 0.1879 | 0.2827 | 0.6233 | 0.7589 | 0.2949 | 0.4295 | 400552 | 11179 |
| pac + bubble [cached-sampling] | 0.1472 | 0.2632 | 0.6567 | 0.6884 | 0.3860 | 0.4283 | 97842 | 3730 |
| quick sort (classic) [cached-sampling] | 0.3043 | 0.2457 | 0.5873 | 0.7470 | 0.2506 | 0.4270 | 465101 | **0** |
| quick sort (classic) [sampling] | 0.3043 | 0.2457 | 0.5873 | 0.7470 | 0.2506 | 0.4270 | 232551 | **0** |
| quick sort (classic) [weird $(1.5)$] | 0.3101 | 0.2578 | 0.5752 | 0.7458 | 0.2442 | 0.4266 | 249228 | **0** |
| prp sort (classic) [sampling] | 0.1884 | 0.2770 | 0.6174 | 0.7497 | 0.2993 | 0.4263 | 210097 | **0** |
| pac top-k (optimized) [bidirectional] | 0.1240 | 0.2640 | 0.6785 | 0.6516 | 0.4084 | 0.4253 | 85643 | **0** |
| pac top-k (optimized) [weird $(1.5)$] | 0.1237 | 0.2505 | 0.6683 | 0.6679 | 0.4143 | 0.4249 | 42821 | **0** |
| mohajer + bubble [sampling] | 0.2457 | 0.2619 | 0.5880 | **0.7708** | 0.2570 | 0.4247 | 99555 | **0** |
| mohajer + bubble [weird $(1.5)$] | 0.2453 | 0.2770 | 0.5746 | 0.7674 | 0.2522 | 0.4233 | 90619 | **0** |
| spectral mle (refinement) [cached-sampling] | 0.2272 | 0.2882 | 0.6155 | 0.7219 | 0.2565 | 0.4219 | 1452416 | **0** |
| spectral mle (refinement) [sampling] | 0.2272 | 0.2882 | 0.6155 | 0.7219 | 0.2565 | 0.4219 | 726208 | **0** |
| pac top-k (3-round) [cached-sampling] | 0.2017 | 0.2378 | 0.6228 | 0.7332 | 0.3035 | 0.4198 | 631628 | 30668 |
| pac top-k (optimized) [cached-sampling] | 0.1199 | 0.2452 | 0.6642 | 0.6718 | 0.3953 | 0.4193 | 85643 | **0** |
| pac top-k (optimized) [sampling] | 0.1199 | 0.2452 | 0.6642 | 0.6718 | 0.3953 | 0.4193 | 42821 | **0** |
| pac top-k (3-round) [weird $(1.5)$] | 0.2025 | 0.2325 | 0.6453 | 0.7177 | 0.2952 | 0.4186 | 347506 | **0** |
| mohajer (ir) [weird $(1.5)$] | 0.2419 | 0.2520 | 0.5648 | 0.7693 | 0.2558 | 0.4168 | 69352 | **0** |
| mohajer + bubble [bidirectional] | 0.2411 | 0.2623 | 0.5487 | 0.7556 | 0.2665 | 0.4149 | 120218 | 22681 |
| mohajer (ir) [sampling] | 0.2412 | 0.2485 | 0.5754 | 0.7611 | 0.2445 | 0.4141 | 67660 | **0** |
| mohajer + bubble [cached-sampling] | 0.2442 | 0.2683 | 0.5694 | 0.7437 | 0.2390 | 0.4129 | 122975 | 36135 |
| mohajer (ir) [cached-sampling] | 0.2396 | 0.2492 | 0.5799 | 0.7611 | 0.2335 | 0.4127 | 108393 | 13245 |
| mohajer (ir) [bidirectional] | 0.2391 | 0.2587 | 0.5385 | 0.7574 | 0.2641 | 0.4116 | 113824 | 14300 |
| bm25 [bidirectional] | 0.1037 | 0.2361 | 0.6789 | 0.5947 | 0.4422 | 0.4111 | **0** | **0** |
| spectral mle (refinement) [bidirectional] | 0.2050 | 0.2034 | 0.5495 | 0.6674 | 0.2943 | 0.3839 | 1452416 | **0** |

### flan-t5-xl
| Ranker | dbpedia-entity | fiqa | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | n/a | n/a | 0.7131 | 0.7482 | 0.4282 | **0.6298** | 87338 | 94329 |
| sliding window prp (classic) [bidirectional] | n/a | n/a | 0.7131 | 0.7482 | 0.4282 | **0.6298** | 77597 | 70749 |
| bubble sort (classic) [weird $(1.5)$] | n/a | n/a | **0.7182** | 0.7746 | 0.3957 | 0.6295 | 316458 | **0** |
| sliding window prp (classic) [cached-sampling] | n/a | n/a | 0.7094 | 0.7662 | 0.3907 | 0.6221 | 121327 | 70893 |
| bubble sort (classic) [cached-sampling] | n/a | n/a | 0.7026 | 0.7638 | 0.3978 | 0.6214 | 145125 | 491235 |
| sliding window prp (classic) [weird $(1.5)$] | n/a | n/a | 0.6942 | 0.7638 | 0.3898 | 0.6159 | 131557 | **0** |
| pac + bubble [bidirectional] | n/a | n/a | 0.7043 | 0.7132 | 0.4110 | 0.6095 | 43447 | 1032 |
| bubble sort (classic) [sampling] | n/a | n/a | 0.7019 | 0.7809 | 0.3453 | 0.6094 | 530596 | **0** |
| sliding window prp (classic) [sampling] | n/a | n/a | 0.6861 | 0.7791 | 0.3486 | 0.6046 | 131557 | **0** |
| pac + bubble [sampling] | n/a | n/a | 0.6857 | 0.7130 | 0.3877 | 0.5955 | 24407 | **0** |
| pac + bubble [weird $(1.5)$] | n/a | n/a | 0.6841 | 0.6997 | 0.3975 | 0.5938 | 23447 | **0** |
| pac top-k (optimized) [bidirectional] | n/a | n/a | 0.6765 | 0.6823 | 0.4207 | 0.5932 | 39367 | **0** |
| pac top-k (optimized) [cached-sampling] | n/a | n/a | 0.6823 | 0.6895 | 0.4073 | 0.5930 | 39367 | **0** |
| pac top-k (optimized) [sampling] | n/a | n/a | 0.6823 | 0.6895 | 0.4073 | 0.5930 | 19684 | **0** |
| prp sort (classic) [bidirectional] | n/a | n/a | 0.7057 | 0.7825 | 0.2843 | 0.5908 | 147551 | 3174 |
| prp sort (classic) [weird $(1.5)$] | n/a | n/a | 0.6861 | 0.7794 | 0.3037 | 0.5897 | 94922 | **0** |
| pac top-k (optimized) [weird $(1.5)$] | n/a | n/a | 0.6732 | 0.6805 | 0.4144 | 0.5894 | 19684 | **0** |
| pac top-k (3-round) [bidirectional] | n/a | n/a | 0.6659 | 0.7732 | 0.3264 | 0.5885 | 282873 | 16007 |
| pac + bubble [cached-sampling] | n/a | n/a | 0.6613 | 0.7003 | 0.3979 | 0.5865 | 45099 | 1706 |
| pac top-k (3-round) [cached-sampling] | n/a | n/a | 0.6537 | 0.7356 | 0.3518 | 0.5804 | 288337 | 12658 |
| pac top-k (2-round) [bidirectional] | n/a | n/a | 0.6647 | 0.7661 | 0.3084 | 0.5797 | 267599 | 161 |
| pac top-k (3-round) [sampling] | n/a | n/a | 0.6414 | 0.7391 | 0.3419 | 0.5742 | 156827 | **0** |
| prp sort (classic) [cached-sampling] | n/a | n/a | 0.6627 | 0.7822 | 0.2732 | 0.5727 | 207726 | 6170 |
| bm25 [bidirectional] | n/a | n/a | 0.6789 | 0.5947 | **0.4422** | 0.5719 | **0** | **0** |
| pac top-k (2-round) [cached-sampling] | n/a | n/a | 0.6581 | 0.7624 | 0.2934 | 0.5713 | 300079 | 999 |
| pac top-k (3-round) [weird $(1.5)$] | n/a | n/a | 0.6507 | 0.7475 | 0.3143 | 0.5708 | 156827 | **0** |
| prp sort (classic) [sampling] | n/a | n/a | 0.6598 | 0.7644 | 0.2824 | 0.5689 | 109384 | **0** |
| pac top-k (2-round) [sampling] | n/a | n/a | 0.6522 | 0.7475 | 0.3049 | 0.5682 | 150986 | **0** |
| mohajer + bubble [sampling] | n/a | n/a | 0.6354 | **0.7855** | 0.2794 | 0.5668 | 46831 | **0** |
| mohajer + bubble [weird $(1.5)$] | n/a | n/a | 0.6411 | 0.7776 | 0.2653 | 0.5613 | 42171 | **0** |
| mohajer (ir) [sampling] | n/a | n/a | 0.6276 | 0.7761 | 0.2720 | 0.5586 | 30891 | **0** |
| mohajer (ir) [weird $(1.5)$] | n/a | n/a | 0.6227 | 0.7708 | 0.2735 | 0.5556 | 31599 | **0** |
| quick sort (classic) [cached-sampling] | n/a | n/a | 0.6353 | 0.7645 | 0.2657 | 0.5552 | 145921 | **0** |
| quick sort (classic) [sampling] | n/a | n/a | 0.6353 | 0.7645 | 0.2657 | 0.5552 | 72960 | **0** |
| spectral mle (refinement) [weird $(1.5)$] | n/a | n/a | 0.6593 | 0.7295 | 0.2759 | 0.5549 | 333622 | **0** |
| pac top-k (2-round) [weird $(1.5)$] | n/a | n/a | 0.6274 | 0.7636 | 0.2735 | 0.5548 | 140596 | **0** |
| mohajer (ir) [cached-sampling] | n/a | n/a | 0.6181 | 0.7659 | 0.2756 | 0.5532 | 51028 | 5282 |
| mohajer + bubble [cached-sampling] | n/a | n/a | 0.6107 | 0.7663 | 0.2772 | 0.5514 | 57912 | 16439 |
| quick sort (classic) [bidirectional] | n/a | n/a | 0.6139 | 0.7717 | 0.2581 | 0.5479 | 249733 | **0** |
| quick sort (classic) [weird $(1.5)$] | n/a | n/a | 0.6113 | 0.7572 | 0.2699 | 0.5461 | 76506 | **0** |
| mohajer + bubble [bidirectional] | n/a | n/a | 0.6188 | 0.7598 | 0.2571 | 0.5452 | 57175 | 9527 |
| spectral mle (refinement) [cached-sampling] | n/a | n/a | 0.6345 | 0.7386 | 0.2570 | 0.5434 | 667245 | **0** |
| spectral mle (refinement) [sampling] | n/a | n/a | 0.6345 | 0.7386 | 0.2570 | 0.5434 | 333622 | **0** |
| mohajer (ir) [bidirectional] | n/a | n/a | 0.6122 | 0.7598 | 0.2545 | 0.5422 | 53775 | 5537 |
| spectral mle (refinement) [bidirectional] | n/a | n/a | 0.5300 | 0.6819 | 0.2693 | 0.4937 | 667245 | **0** |

Notes:
- fiqa: skipped (missing rerank matrix)
- dbpedia-entity: skipped (missing rerank matrix)

## Limit Comparisons Experiment

### trec-covid
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.5947 | 0.5947 | 0.5947 | 0.6380 | 0.6380 | 0.6411 | 0.6563 | 0.6704 | 0.6722 | 0.6806 | 0.6882 | 0.6952 | 0.7002 | 0.7044 | 0.7109 | 0.7151 | 0.7158 |
| Bubble Sort (Classic) [Sampling] | 0.5947 | 0.6350 | 0.6454 | 0.6767 | 0.6668 | 0.7054 | 0.6893 | 0.7188 | 0.7148 | 0.7203 | 0.7197 | 0.7372 | 0.7334 | 0.7386 | 0.7520 | **0.7610** | **0.7620** |
| Jingle Bells [Bidirectional] | 0.6182 | 0.6506 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | 0.6704 | - | - | - |
| Jingle Bells [Sampling] | **0.6631** | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | 0.6688 | - | - | - |
| Mohajer (IR) [Bidirectional] | 0.4898 | 0.4898 | 0.4898 | **0.7359** | 0.7285 | 0.7333 | 0.7544 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | - | - | - |
| Mohajer (IR) [Sampling] | 0.4898 | **0.7297** | **0.7503** | 0.7328 | **0.7611** | **0.7611** | 0.7611 | 0.7611 | **0.7611** | **0.7611** | **0.7611** | **0.7611** | **0.7611** | **0.7611** | - | - | - |
| PAC + Bubble [Bidirectional] | 0.5947 | 0.5947 | 0.5947 | 0.5947 | 0.5947 | 0.6520 | 0.6730 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 |
| PAC + Bubble [Sampling] | 0.5947 | 0.5947 | 0.6786 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 |
| PRP Sort (classic) [Bidirectional] | 0.2637 | 0.2637 | 0.2491 | 0.2478 | 0.2476 | 0.3332 | 0.5207 | 0.6090 | 0.6907 | 0.7366 | 0.7524 | 0.7530 | 0.7530 | 0.7530 | **0.7530** | 0.7530 | 0.7530 |
| PRP Sort (classic) [Sampling] | 0.2716 | 0.2476 | 0.2710 | 0.5546 | 0.7336 | 0.7460 | **0.7669** | **0.7657** | 0.7498 | 0.7536 | 0.7571 | 0.7529 | 0.7393 | 0.7419 | 0.7473 | 0.7587 | 0.7523 |
| Quick Sort (Classic) [Bidirectional] | 0.6471 | 0.6602 | 0.6705 | 0.6752 | 0.6722 | 0.6744 | 0.6786 | 0.6798 | 0.6853 | 0.6897 | 0.6932 | 0.6917 | 0.6974 | 0.7041 | 0.7041 | 0.7138 | 0.7133 |
| Quick Sort (Classic) [Sampling] | 0.6338 | 0.6466 | 0.6446 | 0.6575 | 0.6649 | 0.6791 | 0.6754 | 0.6862 | 0.7093 | 0.7167 | 0.7303 | 0.7367 | 0.7439 | 0.7412 | 0.7470 | 0.7470 | 0.7470 |

### dbpedia-entity
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.1037 | 0.1037 | 0.1037 | 0.1134 | 0.1138 | 0.1157 | 0.1172 | 0.1176 | 0.1176 | 0.1178 | 0.1178 | 0.1178 | 0.1181 | 0.1182 | 0.1182 | 0.1183 | 0.1185 |
| Bubble Sort (Classic) [Sampling] | 0.1037 | 0.1148 | 0.1144 | 0.1172 | 0.1184 | 0.1190 | 0.1176 | 0.1210 | 0.1194 | 0.1211 | 0.1211 | 0.1226 | 0.1201 | 0.1211 | 0.1218 | 0.1221 | 0.1214 |
| Jingle Bells [Bidirectional] | 0.1180 | 0.1468 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | 0.1566 | - | - | - |
| Jingle Bells [Sampling] | **0.1407** | 0.1559 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | 0.1540 | - | - | - |
| Mohajer (IR) [Bidirectional] | 0.1157 | 0.1157 | 0.1157 | 0.2097 | 0.2198 | 0.2333 | 0.2390 | 0.2387 | 0.2391 | 0.2391 | 0.2391 | 0.2391 | 0.2391 | 0.2391 | - | - | - |
| Mohajer (IR) [Sampling] | 0.1152 | **0.2173** | **0.2327** | **0.2371** | **0.2431** | **0.2412** | **0.2412** | **0.2412** | **0.2412** | **0.2412** | **0.2412** | **0.2412** | **0.2412** | **0.2412** | - | - | - |
| PAC + Bubble [Bidirectional] | 0.1037 | 0.1037 | 0.1037 | 0.1037 | 0.1037 | 0.1242 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 | 0.1495 |
| PAC + Bubble [Sampling] | 0.1037 | 0.1037 | 0.1208 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 | 0.1530 |
| PRP Sort (classic) [Bidirectional] | 0.1036 | 0.1032 | 0.0985 | 0.0982 | 0.1090 | 0.1388 | 0.1662 | 0.1756 | 0.1800 | 0.1821 | 0.1828 | 0.1830 | 0.1830 | 0.1830 | 0.1830 | 0.1830 | 0.1830 |
| PRP Sort (classic) [Sampling] | 0.1045 | 0.0944 | 0.1262 | 0.1716 | 0.1847 | 0.1859 | 0.1872 | 0.1858 | 0.1873 | 0.1908 | 0.1880 | 0.1849 | 0.1860 | 0.1848 | 0.1875 | 0.1875 | 0.1884 |
| Quick Sort (Classic) [Bidirectional] | 0.1302 | 0.1492 | 0.1663 | 0.1877 | 0.1972 | 0.2022 | 0.2064 | 0.2124 | 0.2142 | 0.2147 | 0.2159 | 0.2187 | 0.2184 | 0.2195 | **0.2213** | **0.2209** | 0.2217 |
| Quick Sort (Classic) [Sampling] | 0.1240 | 0.1447 | 0.1514 | 0.1606 | 0.1563 | 0.1577 | 0.1578 | 0.1605 | 0.1654 | 0.1718 | 0.1733 | 0.1766 | 0.1852 | 0.1960 | 0.2031 | 0.2123 | **0.2253** |

### fiqa
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.2361 | 0.2361 | 0.2361 | 0.2802 | 0.2803 | 0.2826 | 0.2868 | 0.2906 | 0.2913 | 0.2920 | 0.2924 | 0.2930 | 0.2940 | 0.2940 | 0.2944 | 0.2945 | 0.2945 |
| Bubble Sort (Classic) [Sampling] | 0.2361 | **0.2912** | **0.2795** | **0.2892** | **0.2956** | **0.3035** | **0.3071** | **0.3147** | 0.3086 | 0.3116 | **0.3162** | 0.3125 | 0.3115 | **0.3222** | **0.3145** | **0.3172** | **0.3252** |
| Jingle Bells [Bidirectional] | 0.2106 | 0.2283 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2300 | - | - | - |
| Jingle Bells [Sampling] | 0.2472 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | 0.2478 | - | - | - |
| Mohajer (IR) [Bidirectional] | 0.1531 | 0.1531 | 0.1531 | 0.2515 | 0.2594 | 0.2531 | 0.2569 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | - | - | - |
| Mohajer (IR) [Sampling] | 0.1531 | 0.2085 | 0.2390 | 0.2502 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | - | - | - |
| PAC + Bubble [Bidirectional] | 0.2361 | 0.2361 | 0.2361 | 0.2361 | 0.2361 | 0.2642 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 |
| PAC + Bubble [Sampling] | 0.2361 | 0.2361 | 0.2528 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 |
| PRP Sort (classic) [Bidirectional] | 0.0017 | 0.0017 | 0.0014 | 0.0011 | 0.0843 | 0.2284 | 0.2909 | 0.3058 | **0.3107** | **0.3127** | 0.3127 | **0.3127** | **0.3127** | 0.3127 | 0.3127 | 0.3127 | 0.3127 |
| PRP Sort (classic) [Sampling] | 0.0022 | 0.0022 | 0.0471 | 0.2490 | 0.2718 | 0.2818 | 0.2784 | 0.2869 | 0.2697 | 0.2788 | 0.2767 | 0.2814 | 0.2808 | 0.2767 | 0.2740 | 0.2817 | 0.2770 |
| Quick Sort (Classic) [Bidirectional] | **0.2635** | 0.2663 | 0.2665 | 0.2647 | 0.2657 | 0.2681 | 0.2679 | 0.2682 | 0.2682 | 0.2676 | 0.2680 | 0.2680 | 0.2704 | 0.2717 | 0.2707 | 0.2705 | 0.2727 |
| Quick Sort (Classic) [Sampling] | 0.2415 | 0.2408 | 0.2483 | 0.2372 | 0.2506 | 0.2452 | 0.2400 | 0.2410 | 0.2462 | 0.2437 | 0.2512 | 0.2482 | 0.2511 | 0.2457 | 0.2457 | 0.2457 | 0.2457 |

### webis-touche2020
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | **0.4422** | **0.4422** | **0.4422** | **0.4486** | **0.4486** | **0.4465** | **0.4485** | **0.4533** | **0.4523** | **0.4492** | **0.4472** | **0.4466** | **0.4471** | **0.4458** |
| Bubble Sort (Classic) [Sampling] | **0.4422** | 0.4327 | 0.4218 | 0.4236 | 0.4444 | 0.4328 | 0.4240 | 0.4330 | 0.4219 | 0.4336 | 0.4180 | 0.4267 | 0.4128 | 0.3980 |
| Jingle Bells [Bidirectional] | 0.3631 | 0.3406 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 | 0.3191 |
| Jingle Bells [Sampling] | 0.3718 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 | 0.3784 |
| Mohajer (IR) [Bidirectional] | 0.2110 | 0.2110 | 0.2110 | 0.3427 | 0.3292 | 0.2887 | 0.2703 | 0.2637 | 0.2641 | 0.2641 | 0.2641 | 0.2641 | 0.2641 | 0.2641 |
| Mohajer (IR) [Sampling] | 0.2110 | 0.2296 | 0.2320 | 0.2356 | 0.2445 | 0.2445 | 0.2445 | 0.2445 | 0.2445 | 0.2445 | 0.2445 | 0.2445 | 0.2445 | 0.2445 |
| PAC + Bubble [Bidirectional] | **0.4422** | **0.4422** | **0.4422** | 0.4422 | 0.4422 | 0.4083 | 0.4143 | 0.4143 | 0.4143 | 0.4143 | 0.4143 | 0.4143 | 0.4143 | 0.4143 |
| PAC + Bubble [Sampling] | **0.4422** | **0.4422** | 0.3788 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 | 0.3823 |
| PRP Sort (classic) [Bidirectional] | 0.0236 | 0.0236 | 0.0250 | 0.0316 | 0.0823 | 0.1913 | 0.2660 | 0.3028 | 0.3217 | 0.3321 | 0.3321 | 0.3321 | 0.3321 | 0.3321 |
| PRP Sort (classic) [Sampling] | 0.0219 | 0.0373 | 0.0824 | 0.2186 | 0.2827 | 0.2679 | 0.2840 | 0.2741 | 0.2698 | 0.2806 | 0.2806 | 0.2738 | 0.2604 | 0.2751 |
| Quick Sort (Classic) [Bidirectional] | 0.4183 | 0.3964 | 0.3931 | 0.3763 | 0.3810 | 0.3838 | 0.3838 | 0.3807 | 0.3819 | 0.3819 | 0.3847 | 0.3860 | 0.3863 | 0.3855 |
| Quick Sort (Classic) [Sampling] | 0.3926 | 0.3956 | 0.3968 | 0.3900 | 0.4023 | 0.3824 | 0.3748 | 0.3431 | 0.3255 | 0.3266 | 0.2492 | 0.2367 | 0.2430 | 0.2506 |

### scifact
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | **0.6789** | **0.6789** | 0.6789 | **0.6939** | **0.6939** | **0.6940** | **0.6916** | **0.6898** | **0.6901** | **0.6916** | **0.6925** | **0.6925** | **0.6925** | **0.6925** |
| Bubble Sort (Classic) [Sampling] | **0.6789** | 0.6678 | 0.6786 | 0.6563 | 0.6761 | 0.6652 | 0.6682 | 0.6716 | 0.6823 | 0.6567 | 0.6720 | 0.6812 | 0.6682 | 0.6761 |
| Jingle Bells [Bidirectional] | 0.5672 | 0.5766 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 | 0.5739 |
| Jingle Bells [Sampling] | 0.6360 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 | 0.6452 |
| Mohajer (IR) [Bidirectional] | 0.5527 | 0.5527 | 0.5527 | 0.5525 | 0.5312 | 0.5310 | 0.5365 | 0.5389 | 0.5385 | 0.5385 | 0.5385 | 0.5385 | 0.5385 | 0.5385 |
| Mohajer (IR) [Sampling] | 0.5527 | 0.5200 | 0.5651 | 0.5832 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 |
| PAC + Bubble [Bidirectional] | **0.6789** | **0.6789** | 0.6789 | 0.6789 | 0.6789 | 0.6786 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 |
| PAC + Bubble [Sampling] | **0.6789** | **0.6789** | 0.6802 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 |
| PRP Sort (classic) [Bidirectional] | 0.0015 | 0.0015 | 0.0025 | 0.0025 | 0.1184 | 0.5449 | 0.6414 | 0.6657 | 0.6747 | 0.6747 | 0.6747 | 0.6747 | 0.6747 | 0.6747 |
| PRP Sort (classic) [Sampling] | 0.0004 | 0.0015 | 0.2432 | 0.5722 | 0.6207 | 0.6174 | 0.6140 | 0.6177 | 0.6128 | 0.6196 | 0.6229 | 0.6168 | 0.6312 | 0.6201 |
| Quick Sort (Classic) [Bidirectional] | 0.6652 | 0.6641 | 0.6629 | 0.6585 | 0.6589 | 0.6568 | 0.6568 | 0.6546 | 0.6551 | 0.6551 | 0.6551 | 0.6564 | 0.6570 | 0.6508 |
| Quick Sort (Classic) [Sampling] | 0.6724 | 0.6723 | **0.6824** | 0.6745 | 0.6675 | 0.6565 | 0.6606 | 0.6445 | 0.6407 | 0.6292 | 0.5761 | 0.5789 | 0.5664 | 0.5873 |

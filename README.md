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
| Ranker | dbpedia-entity | dl-2019 | dl-2020 | fiqa | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | **0.4158** | 0.6342 | 0.5859 | 0.2945 | **0.4416** | **0.6925** | 0.7177 | **0.4473** | **0.5287** | 158736 | 199027 |
| prp sort (classic) [bidirectional] | 0.4128 | **0.6501** | **0.6264** | **0.3127** | 0.4043 | 0.6747 | 0.7530 | 0.3321 | 0.5208 | 275580 | 6332 |
| pac + bubble [sampling] | 0.3794 | 0.5998 | 0.5730 | 0.2719 | 0.4104 | 0.6699 | 0.7002 | 0.3823 | 0.4984 | 44610 | **0** |
| quick sort (classic) [bidirectional] | 0.4114 | 0.6454 | 0.5847 | 0.2684 | 0.4100 | 0.6008 | 0.7661 | 0.2736 | 0.4951 | 510656 | **0** |
| mohajer + bubble [sampling] | 0.3995 | 0.6175 | 0.5882 | 0.2619 | 0.3779 | 0.5880 | **0.7708** | 0.2570 | 0.4826 | 86132 | **0** |
| jingle bells [sampling] | 0.3694 | 0.5892 | 0.5398 | 0.2478 | 0.4050 | 0.6452 | 0.6688 | 0.3784 | 0.4804 | 16447 | **0** |
| mohajer (ir) [sampling] | 0.3837 | 0.6064 | 0.5723 | 0.2485 | 0.3624 | 0.5754 | 0.7611 | 0.2445 | 0.4693 | 56343 | **0** |
| christmas tree [sampling] | 0.3347 | 0.5411 | 0.5165 | 0.2463 | 0.4052 | 0.6579 | 0.6180 | 0.3878 | 0.4634 | 50168 | **0** |
| bm25 [bidirectional] | 0.3185 | 0.5058 | 0.4796 | 0.2361 | 0.4070 | 0.6789 | 0.5947 | 0.4422 | 0.4578 | **0** | **0** |

### flan-t5-xl
| Ranker | dbpedia-entity | dl-2019 | dl-2020 | fiqa | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | n/a | 0.6844 | 0.6697 | n/a | n/a | **0.7131** | 0.7482 | 0.4282 | **0.6487** | 131182 | 128504 |
| prp sort (classic) [bidirectional] | n/a | **0.7058** | **0.6892** | n/a | n/a | 0.7057 | 0.7825 | 0.2843 | 0.6335 | 165520 | 3994 |
| mohajer + bubble [sampling] | n/a | 0.6947 | 0.6630 | n/a | n/a | 0.6354 | **0.7855** | 0.2794 | 0.6116 | 44556 | **0** |
| mohajer (ir) [sampling] | n/a | 0.6873 | 0.6763 | n/a | n/a | 0.6276 | 0.7761 | 0.2720 | 0.6079 | 29798 | **0** |
| quick sort (classic) [bidirectional] | n/a | 0.7036 | 0.6719 | n/a | n/a | 0.6139 | 0.7717 | 0.2581 | 0.6038 | 220001 | **0** |
| pac + bubble [sampling] | n/a | 0.6105 | 0.5851 | n/a | n/a | 0.6857 | 0.7130 | 0.3877 | 0.5964 | 23577 | **0** |
| jingle bells [sampling] | n/a | 0.6226 | 0.5777 | n/a | n/a | 0.6523 | 0.6886 | 0.3908 | 0.5864 | 8595 | **0** |
| christmas tree [sampling] | n/a | 0.5481 | 0.5287 | n/a | n/a | 0.6520 | 0.6259 | 0.3953 | 0.5500 | 26452 | **0** |
| bm25 [bidirectional] | n/a | 0.5058 | 0.4796 | n/a | n/a | 0.6789 | 0.5947 | **0.4422** | 0.5403 | **0** | **0** |

Notes:
- fiqa: skipped (missing rerank matrix)
- dbpedia-entity: skipped (missing rerank matrix)
- robust04: skipped (missing rerank matrix)

### flan-t5-xxl
| Ranker | dbpedia-entity | dl-2019 | dl-2020 | fiqa | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prp sort (classic) [bidirectional] | n/a | **0.7021** | 0.6882 | n/a | n/a | n/a | n/a | n/a | **0.6951** | 180390 | 4764 |
| mohajer + bubble [sampling] | n/a | 0.6854 | **0.6896** | n/a | n/a | n/a | n/a | n/a | 0.6875 | 41766 | **0** |
| bubble sort (classic) [bidirectional] | n/a | 0.6785 | 0.6798 | n/a | n/a | n/a | n/a | n/a | 0.6792 | 157092 | 143453 |
| quick sort (classic) [bidirectional] | n/a | 0.6902 | 0.6657 | n/a | n/a | n/a | n/a | n/a | 0.6779 | 185644 | **0** |
| mohajer (ir) [sampling] | n/a | 0.6816 | 0.6673 | n/a | n/a | n/a | n/a | n/a | 0.6745 | 28097 | **0** |
| pac + bubble [sampling] | n/a | 0.6093 | 0.6072 | n/a | n/a | n/a | n/a | n/a | 0.6082 | 22304 | **0** |
| jingle bells [sampling] | n/a | 0.6052 | 0.5807 | n/a | n/a | n/a | n/a | n/a | 0.5929 | 8070 | **0** |
| christmas tree [sampling] | n/a | 0.5384 | 0.5374 | n/a | n/a | n/a | n/a | n/a | 0.5379 | 24907 | **0** |
| bm25 [bidirectional] | n/a | 0.5058 | 0.4796 | n/a | n/a | n/a | n/a | n/a | 0.4927 | **0** | **0** |

Notes:
- webis-touche2020: skipped (missing rerank matrix)
- trec-covid: skipped (missing rerank matrix)
- scifact: skipped (missing rerank matrix)
- fiqa: skipped (missing rerank matrix)
- dbpedia-entity: skipped (missing rerank matrix)
- robust04: skipped (missing rerank matrix)
<!-- END_BEIR_RESULTS -->

## Limit Comparisons Experiment

### trec-covid
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [Bidirectional] | **0.5947** | 0.5947 | 0.5947 | 0.6380 | 0.6380 | 0.6411 | 0.6563 | 0.6704 | 0.6722 | 0.6806 | 0.6882 | 0.6952 | 0.7002 | 0.7044 |
| bubble sort (classic) [Sampling] | **0.5947** | 0.6350 | 0.6454 | 0.6767 | 0.6668 | 0.7054 | 0.6893 | 0.7188 | 0.7148 | 0.7203 | 0.7197 | 0.7372 | 0.7334 | 0.7386 |
| mohajer (ir) [Bidirectional] | 0.4898 | 0.4898 | 0.4898 | **0.7359** | 0.7285 | 0.7333 | 0.7544 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | 0.7574 | 0.7574 |
| mohajer (ir) [Sampling] | 0.4898 | **0.7297** | **0.7503** | 0.7328 | **0.7611** | **0.7611** | 0.7611 | 0.7611 | **0.7611** | **0.7611** | **0.7611** | **0.7611** | **0.7611** | **0.7611** |
| pac + bubble [Bidirectional] | **0.5947** | 0.5947 | 0.5947 | 0.5947 | 0.5947 | 0.6520 | 0.6730 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 | 0.6729 |
| pac + bubble [Sampling] | **0.5947** | 0.5947 | 0.6786 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 | 0.7002 |
| prp sort (classic) [Bidirectional] | 0.2637 | 0.2637 | 0.2491 | 0.2478 | 0.2476 | 0.3332 | 0.5207 | 0.6090 | 0.6907 | 0.7366 | 0.7524 | 0.7530 | 0.7530 | 0.7530 |
| prp sort (classic) [Sampling] | 0.2716 | 0.2476 | 0.2710 | 0.5546 | 0.7336 | 0.7460 | **0.7669** | **0.7657** | 0.7498 | 0.7536 | 0.7571 | 0.7529 | 0.7393 | 0.7419 |

### scifact
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [Bidirectional] | **0.6789** | **0.6789** | 0.6789 | **0.6939** | **0.6939** | **0.6940** | **0.6916** | **0.6898** | **0.6901** | **0.6916** | **0.6925** | **0.6925** | **0.6925** | **0.6925** |
| bubble sort (classic) [Sampling] | **0.6789** | 0.6678 | 0.6786 | 0.6563 | 0.6761 | 0.6652 | 0.6682 | 0.6716 | 0.6823 | 0.6567 | 0.6720 | 0.6812 | 0.6682 | 0.6761 |
| mohajer (ir) [Bidirectional] | 0.5527 | 0.5527 | 0.5527 | 0.5525 | 0.5312 | 0.5310 | 0.5365 | 0.5389 | 0.5385 | 0.5385 | 0.5385 | 0.5385 | 0.5385 | 0.5385 |
| mohajer (ir) [Sampling] | 0.5527 | 0.5200 | 0.5651 | 0.5832 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 | 0.5754 |
| pac + bubble [Bidirectional] | **0.6789** | **0.6789** | 0.6789 | 0.6789 | 0.6789 | 0.6786 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 | 0.6845 |
| pac + bubble [Sampling] | **0.6789** | **0.6789** | **0.6802** | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6699 |
| prp sort (classic) [Bidirectional] | 0.0015 | 0.0015 | 0.0025 | 0.0025 | 0.1184 | 0.5449 | 0.6414 | 0.6657 | 0.6747 | 0.6747 | 0.6747 | 0.6747 | 0.6747 | 0.6747 |
| prp sort (classic) [Sampling] | 0.0004 | 0.0015 | 0.2432 | 0.5722 | 0.6207 | 0.6174 | 0.6140 | 0.6177 | 0.6128 | 0.6196 | 0.6229 | 0.6168 | 0.6312 | 0.6201 |

### fiqa
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [Bidirectional] | **0.2361** | 0.2361 | 0.2361 | 0.2802 | 0.2803 | 0.2826 | 0.2868 | 0.2906 | 0.2913 | 0.2920 | 0.2924 | 0.2930 | 0.2940 | 0.2940 |
| bubble sort (classic) [Sampling] | **0.2361** | **0.2912** | **0.2795** | **0.2892** | **0.2956** | **0.3035** | **0.3071** | **0.3147** | 0.3086 | 0.3116 | **0.3162** | 0.3125 | 0.3115 | **0.3222** |
| mohajer (ir) [Bidirectional] | 0.1531 | 0.1531 | 0.1531 | 0.2515 | 0.2594 | 0.2531 | 0.2569 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | 0.2587 | 0.2587 |
| mohajer (ir) [Sampling] | 0.1531 | 0.2085 | 0.2390 | 0.2502 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 | 0.2485 |
| pac + bubble [Bidirectional] | **0.2361** | 0.2361 | 0.2361 | 0.2361 | 0.2361 | 0.2642 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 | 0.2920 |
| pac + bubble [Sampling] | **0.2361** | 0.2361 | 0.2528 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 | 0.2719 |
| prp sort (classic) [Bidirectional] | 0.0017 | 0.0017 | 0.0014 | 0.0011 | 0.0843 | 0.2284 | 0.2909 | 0.3058 | **0.3107** | **0.3127** | 0.3127 | **0.3127** | **0.3127** | 0.3127 |
| prp sort (classic) [Sampling] | 0.0022 | 0.0022 | 0.0471 | 0.2490 | 0.2718 | 0.2818 | 0.2784 | 0.2869 | 0.2697 | 0.2788 | 0.2767 | 0.2814 | 0.2808 | 0.2767 |

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
| Ranker | dl-2019 | dl-2020 | fiqa | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | 0.6342 | 0.5859 | 0.2945 | **0.4416** | **0.6925** | 0.7177 | **0.4473** | **0.5448** | 130532 | 168375 |
| prp sort (classic) [bidirectional] | **0.6501** | **0.6264** | **0.3127** | 0.4043 | 0.6747 | 0.7530 | 0.3321 | 0.5362 | 242303 | 5493 |
| pac + bubble [sampling] | 0.5998 | 0.5730 | 0.2719 | 0.4104 | 0.6699 | 0.7002 | 0.3823 | 0.5154 | 40447 | **0** |
| quick sort (classic) [bidirectional] | 0.6454 | 0.5847 | 0.2684 | 0.4100 | 0.6008 | 0.7661 | 0.2736 | 0.5070 | 471467 | **0** |
| jingle bells [sampling] | 0.5892 | 0.5398 | 0.2478 | 0.4050 | 0.6452 | 0.6688 | 0.3784 | 0.4963 | 14926 | **0** |
| mohajer + bubble [sampling] | 0.6175 | 0.5882 | 0.2619 | 0.3779 | 0.5880 | **0.7708** | 0.2570 | 0.4945 | 78292 | **0** |
| christmas tree [sampling] | 0.5411 | 0.5165 | 0.2463 | 0.4052 | 0.6579 | 0.6180 | 0.3878 | 0.4818 | 45512 | **0** |
| mohajer (ir) [sampling] | 0.6064 | 0.5723 | 0.2485 | 0.3624 | 0.5754 | 0.7611 | 0.2445 | 0.4815 | 51155 | **0** |
| bm25 [bidirectional] | 0.5058 | 0.4796 | 0.2361 | 0.4070 | 0.6789 | 0.5947 | 0.4422 | 0.4778 | **0** | **0** |

### flan-t5-xl
| Ranker | dl-2019 | dl-2020 | fiqa | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | 0.6844 | 0.6697 | n/a | n/a | **0.7131** | 0.7482 | 0.4282 | **0.6487** | 131182 | 128504 |
| prp sort (classic) [bidirectional] | **0.7058** | **0.6892** | n/a | n/a | 0.7057 | 0.7825 | 0.2843 | 0.6335 | 165520 | 3994 |
| mohajer + bubble [sampling] | 0.6947 | 0.6630 | n/a | n/a | 0.6354 | **0.7855** | 0.2794 | 0.6116 | 44556 | **0** |
| mohajer (ir) [sampling] | 0.6873 | 0.6763 | n/a | n/a | 0.6276 | 0.7761 | 0.2720 | 0.6079 | 29798 | **0** |
| quick sort (classic) [bidirectional] | 0.7036 | 0.6719 | n/a | n/a | 0.6139 | 0.7717 | 0.2581 | 0.6038 | 220001 | **0** |
| pac + bubble [sampling] | 0.6105 | 0.5851 | n/a | n/a | 0.6857 | 0.7130 | 0.3877 | 0.5964 | 23577 | **0** |
| jingle bells [sampling] | 0.6226 | 0.5777 | n/a | n/a | 0.6523 | 0.6886 | 0.3908 | 0.5864 | 8595 | **0** |
| christmas tree [sampling] | 0.5481 | 0.5287 | n/a | n/a | 0.6520 | 0.6259 | 0.3953 | 0.5500 | 26452 | **0** |
| bm25 [bidirectional] | 0.5058 | 0.4796 | n/a | n/a | 0.6789 | 0.5947 | **0.4422** | 0.5403 | **0** | **0** |

Notes:
- fiqa: skipped (missing rerank matrix)
- robust04: skipped (missing rerank matrix)
<!-- END_BEIR_RESULTS -->


## Limit Comparisons Experiment

### trec-covid
| DisplayRanker | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 800 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bubble Sort (Classic) [Bidirectional] | 0.5947 | 0.5947 | 0.5947 | 0.6380 | 0.6380 | 0.6411 | 0.6563 | 0.6704 | 0.6722 | 0.6806 | 0.6882 | 0.6952 | 0.7002 | 0.7044 | 0.7109 | 0.7151 | 0.7158 |
| Bubble Sort (Classic) [Sampling] | 0.5947 | 0.6350 | 0.6454 | 0.6767 | 0.6668 | 0.7054 | 0.6893 | 0.7188 | 0.7148 | 0.7203 | 0.7197 | 0.7372 | 0.7334 | 0.7386 | 0.7520 | **0.7610** | **0.7620** |
| Christmas Tree [Bidirectional] | 0.5518 | 0.5698 | 0.5950 | 0.6024 | 0.6064 | 0.5993 | 0.6062 | 0.6049 | 0.6049 | 0.6049 | 0.6049 | 0.6049 | 0.6049 | 0.6049 | - | - | - |
| Christmas Tree [Sampling] | 0.5729 | 0.5917 | 0.5946 | 0.6077 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | 0.6180 | - | - | - |
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
| Christmas Tree [Bidirectional] | 0.1177 | 0.1161 | 0.1158 | 0.1191 | 0.1228 | 0.1282 | 0.1316 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | - | - | - |
| Christmas Tree [Sampling] | 0.1153 | 0.1172 | 0.1205 | 0.1269 | 0.1323 | 0.1349 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | - | - | - |
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
| Christmas Tree [Bidirectional] | 0.2048 | 0.2191 | 0.2221 | 0.2244 | 0.2291 | 0.2338 | 0.2371 | 0.2408 | 0.2405 | 0.2405 | 0.2405 | 0.2405 | 0.2405 | 0.2405 | - | - | - |
| Christmas Tree [Sampling] | 0.2181 | 0.2304 | 0.2330 | 0.2435 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | 0.2463 | - | - | - |
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
| Christmas Tree [Bidirectional] | 0.3319 | 0.3652 | 0.3691 | 0.3852 | 0.3941 | 0.4045 | 0.4004 | 0.4007 | 0.4007 | 0.4007 | 0.4007 | 0.4007 | 0.4007 | 0.4007 |
| Christmas Tree [Sampling] | 0.3605 | 0.3870 | 0.3960 | 0.4012 | 0.3878 | 0.3878 | 0.3878 | 0.3878 | 0.3878 | 0.3878 | 0.3878 | 0.3878 | 0.3878 | 0.3878 |
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
| Christmas Tree [Bidirectional] | 0.5999 | 0.6234 | 0.6371 | 0.6459 | 0.6459 | 0.6612 | 0.6601 | 0.6585 | 0.6585 | 0.6585 | 0.6585 | 0.6585 | 0.6585 | 0.6585 |
| Christmas Tree [Sampling] | 0.6251 | 0.6485 | 0.6342 | 0.6615 | 0.6579 | 0.6579 | 0.6579 | 0.6579 | 0.6579 | 0.6579 | 0.6579 | 0.6579 | 0.6579 | 0.6579 |
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

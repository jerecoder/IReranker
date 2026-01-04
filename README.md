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
| Ranker | dbpedia-entity | dl-2019 | dl-2020 | fiqa | nfcorpus | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | **0.4158** | 0.6342 | 0.5859 | 0.2945 | n/a | **0.4416** | **0.6925** | 0.7177 | **0.4473** | **0.5287** | 158736 | 199027 |
| prp sort (classic) [bidirectional] | 0.4128 | **0.6501** | **0.6264** | **0.3127** | n/a | 0.4043 | 0.6747 | 0.7530 | 0.3321 | 0.5208 | 275580 | 6332 |
| pac + bubble [sampling] | 0.3794 | 0.5998 | 0.5730 | 0.2719 | n/a | 0.4104 | 0.6699 | 0.7002 | 0.3823 | 0.4984 | 44610 | **0** |
| quick sort (classic) [bidirectional] | 0.4114 | 0.6454 | 0.5847 | 0.2684 | n/a | 0.4100 | 0.6008 | 0.7661 | 0.2736 | 0.4951 | 510656 | **0** |
| mohajer + bubble [sampling] | 0.3995 | 0.6175 | 0.5882 | 0.2619 | n/a | 0.3779 | 0.5880 | **0.7708** | 0.2570 | 0.4826 | 86132 | **0** |
| jingle bells [sampling] | 0.3694 | 0.5892 | 0.5398 | 0.2478 | n/a | 0.4050 | 0.6452 | 0.6688 | 0.3784 | 0.4804 | 16447 | **0** |
| mohajer (ir) [sampling] | 0.3837 | 0.6064 | 0.5723 | 0.2485 | n/a | 0.3624 | 0.5754 | 0.7611 | 0.2445 | 0.4693 | 56343 | **0** |
| christmas tree [sampling] | 0.3347 | 0.5411 | 0.5165 | 0.2463 | n/a | 0.4052 | 0.6579 | 0.6180 | 0.3878 | 0.4634 | 50168 | **0** |
| bm25 [bidirectional] | 0.3185 | 0.5058 | 0.4796 | 0.2361 | n/a | 0.4070 | 0.6789 | 0.5947 | 0.4422 | 0.4578 | **0** | **0** |
| sliding window prp (classic) | n/a | n/a | n/a | n/a | **0.4146** | n/a | n/a | n/a | n/a | 0.4146 | 105810 | **0** |
| nothing | n/a | n/a | n/a | n/a | 0.3982 | n/a | n/a | n/a | n/a | 0.3982 | **0** | **0** |
| random | n/a | n/a | n/a | n/a | 0.3702 | n/a | n/a | n/a | n/a | 0.3702 | **0** | **0** |

### flan-t5-xl
| Ranker | dbpedia-entity | dl-2019 | dl-2020 | fiqa | nfcorpus | robust04 | scifact | trec-covid | webis-touche2020 | Average | Avg #Inference | Avg Cache Hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [bidirectional] | n/a | 0.6844 | 0.6697 | n/a | n/a | n/a | **0.7131** | 0.7482 | 0.4282 | **0.6487** | 131182 | 128504 |
| prp sort (classic) [bidirectional] | n/a | **0.7058** | **0.6892** | n/a | n/a | n/a | 0.7057 | 0.7825 | 0.2843 | 0.6335 | 165520 | 3994 |
| mohajer + bubble [sampling] | n/a | 0.6947 | 0.6630 | n/a | n/a | n/a | 0.6354 | **0.7855** | 0.2794 | 0.6116 | 44556 | **0** |
| mohajer (ir) [sampling] | n/a | 0.6873 | 0.6763 | n/a | n/a | n/a | 0.6276 | 0.7761 | 0.2720 | 0.6079 | 29798 | **0** |
| quick sort (classic) [bidirectional] | n/a | 0.7036 | 0.6719 | n/a | n/a | n/a | 0.6139 | 0.7717 | 0.2581 | 0.6038 | 220001 | **0** |
| pac + bubble [sampling] | n/a | 0.6105 | 0.5851 | n/a | n/a | n/a | 0.6857 | 0.7130 | 0.3877 | 0.5964 | 23577 | **0** |
| jingle bells [sampling] | n/a | 0.6226 | 0.5777 | n/a | n/a | n/a | 0.6523 | 0.6886 | 0.3908 | 0.5864 | 8595 | **0** |
| christmas tree [sampling] | n/a | 0.5481 | 0.5287 | n/a | n/a | n/a | 0.6520 | 0.6259 | 0.3953 | 0.5500 | 26452 | **0** |
| bm25 [bidirectional] | n/a | 0.5058 | 0.4796 | n/a | n/a | n/a | 0.6789 | 0.5947 | **0.4422** | 0.5403 | **0** | **0** |
| sliding window prp (classic) | n/a | n/a | n/a | n/a | **0.4294** | n/a | n/a | n/a | n/a | 0.4294 | 86975 | **0** |
| nothing | n/a | n/a | n/a | n/a | 0.3982 | n/a | n/a | n/a | n/a | 0.3982 | **0** | **0** |
| random | n/a | n/a | n/a | n/a | 0.3702 | n/a | n/a | n/a | n/a | 0.3702 | **0** | **0** |

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
<!-- END_BEIR_RESULTS -->

## Limit Comparisons Experiment

### dl-2019
| DisplayRanker | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [Bidirectional] | 0.5058 | 0.5058 | 0.5713 | 0.5713 | 0.5713 | 0.5778 | 0.6064 | 0.6069 | 0.6095 |
| bubble sort (classic) [Sampling] | 0.5592 | 0.5688 | 0.5977 | 0.6033 | 0.6198 | 0.6252 | 0.6427 | 0.6437 | 0.6573 |
| mohajer (ir) [Bidirectional] | 0.3276 | 0.3276 | 0.6244 | 0.6535 | 0.6634 | 0.6623 | 0.6664 | 0.6658 | 0.6658 |
| mohajer (ir) [Sampling] | **0.6141** | **0.6779** | **0.6870** | **0.6873** | **0.6873** | 0.6873 | 0.6873 | 0.6873 | 0.6873 |
| mohajer + bubble [Bidirectional] | 0.3276 | 0.3276 | 0.6244 | 0.6535 | 0.6634 | 0.6623 | 0.6664 | 0.6659 | 0.6659 |
| mohajer + bubble [Sampling] | **0.6141** | **0.6779** | **0.6870** | 0.6799 | 0.6853 | **0.6917** | 0.6947 | **0.6947** | **0.6947** |
| pac + bubble [Bidirectional] | 0.5058 | 0.5058 | 0.5058 | 0.5058 | 0.6000 | 0.6259 | 0.6260 | 0.6260 | 0.6260 |
| pac + bubble [Sampling] | 0.5058 | 0.5835 | 0.6105 | 0.6105 | 0.6105 | 0.6105 | 0.6105 | 0.6105 | 0.6105 |
| prp sort (classic) [Bidirectional] | 0.0915 | 0.0849 | 0.0861 | 0.1191 | 0.2397 | 0.4168 | 0.5414 | 0.6358 | 0.6892 |
| prp sort (classic) [Sampling] | 0.0911 | 0.1689 | 0.4927 | 0.6593 | 0.6849 | 0.6903 | **0.6982** | 0.6876 | 0.6884 |
| quick sort (classic) [Bidirectional] | 0.5808 | 0.5808 | 0.5830 | 0.5857 | 0.5857 | 0.5857 | 0.5885 | 0.5908 | 0.5927 |
| quick sort (classic) [Sampling] | 0.5633 | 0.5757 | 0.5698 | 0.5989 | 0.5842 | 0.6144 | 0.6300 | 0.6483 | 0.6437 |

### dl-2020
| DisplayRanker | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bubble sort (classic) [Bidirectional] | 0.4796 | 0.4796 | 0.5572 | 0.5572 | 0.5572 | 0.5618 | 0.5986 | 0.5990 | 0.6007 |
| bubble sort (classic) [Sampling] | 0.5523 | 0.5531 | 0.5903 | 0.5845 | 0.6130 | 0.6178 | 0.6335 | 0.6437 | 0.6588 |
| mohajer (ir) [Bidirectional] | 0.2748 | 0.2748 | 0.6224 | 0.6425 | 0.6583 | 0.6632 | 0.6698 | 0.6735 | 0.6735 |
| mohajer (ir) [Sampling] | **0.5995** | **0.6506** | **0.6766** | **0.6763** | **0.6763** | 0.6763 | **0.6763** | 0.6763 | 0.6763 |
| mohajer + bubble [Bidirectional] | 0.2748 | 0.2748 | 0.6224 | 0.6425 | 0.6583 | 0.6632 | 0.6702 | 0.6745 | 0.6745 |
| mohajer + bubble [Sampling] | **0.5995** | **0.6506** | **0.6766** | 0.6761 | 0.6705 | 0.6760 | 0.6630 | 0.6630 | 0.6630 |
| pac + bubble [Bidirectional] | 0.4796 | 0.4796 | 0.4796 | 0.4796 | 0.5504 | 0.5859 | 0.5862 | 0.5862 | 0.5862 |
| pac + bubble [Sampling] | 0.4796 | 0.5527 | 0.5851 | 0.5851 | 0.5851 | 0.5851 | 0.5851 | 0.5851 | 0.5851 |
| prp sort (classic) [Bidirectional] | 0.0448 | 0.0377 | 0.0365 | 0.0665 | 0.2212 | 0.4200 | 0.5444 | 0.6205 | 0.6750 |
| prp sort (classic) [Sampling] | 0.0459 | 0.1596 | 0.5138 | 0.6488 | 0.6731 | **0.6764** | 0.6727 | **0.6778** | **0.6858** |
| quick sort (classic) [Bidirectional] | 0.5377 | 0.5369 | 0.5344 | 0.5383 | 0.5383 | 0.5383 | 0.5395 | 0.5410 | 0.5410 |
| quick sort (classic) [Sampling] | 0.5151 | 0.5328 | 0.5434 | 0.5467 | 0.5615 | 0.5605 | 0.5725 | 0.5976 | 0.6191 |

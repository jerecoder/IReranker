# IReranker

RAG reranking con IR.

## Estructura

```
├─ LICENSE
├─ Makefile
├─ README.md
├─ config
│  ├─ beir_eval.json       # Config de evaluación (datasets, k_values, output_dir, etc.)
│  └─ beir_loader.json     # Config del loader BEIR (base_url, cache_subdir)
├─ data
│  └─ external             # Cache de datasets BEIR
├─ reports
│  └─ beir-metrics         # Salidas por dataset (CSV), o según config/output_dir
├─ ireranker
│  ├─ __init__.py
│  ├─ config.py            # Rutas y logging
│  ├─ types.py             # RankingTask, RankingDataset
│  ├─ rankers
│  │  ├─ __init__.py
│  │  ├─ Ranker.py
│  │  ├─ registry.py
│  │  ├─ RandomRanker.py
│  │  └─ BubbleRanker.py
│  ├─ data
│  │  └─ loaders.py        # Loader BEIR
│  ├─ evaluation
│  │  └─ beir.py           # Evaluación con BEIR
│  └─ cli
│     └─ beir_eval.py      # CLI de evaluación
└─ pyproject.toml
```

## Instalacion con conda (sugerido)

  - `make environment` (crea `IReranker` con Python 3.10 e instala deps)
  - `conda activate IReranker`

Comandos útiles dentro del env `IReranker`:
- Instalar dependencias del proyecto: `make requirements`
- Lint/format: `make lint` / `make format`
- Tests: `make test`
- Evaluación BEIR (usa configs): `make beir-eval`

## Configuración

- `config/beir_eval.json`
  - `datasets`: lista de datasets BEIR a evaluar (nombres canónicos)
  - `split`: split (p.ej. "test")
  - `max_queries`: límite de queries (o null)
  - `light_exclude`: lista de datasets que se omiten cuando se corre con `--light`
  - `seed`: semilla
  - `k_values`: cortes de evaluación, p.ej. [1,3,5,10,100]
  - `output_dir`: destino de resultados (absoluto o relativo a REPORTS_DIR)
  - `rankers`: ["all"] o lista de nombres
  - Ejemplo actual (config/beir_eval.json):

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
      "max_queries": null,
      "seed": 69420,
      "k_values": [1, 3, 5, 10, 100],
      "light_exclude": ["dbpedia-entity", "trec-covid", "fiqa", "nfcorpus"],
      "output_dir": "beir-metrics",
      "rankers": ["all"]
    }
    ```

- `config/beir_loader.json`
  - `base_url`: URL base de BEIR
  - `cache_subdir`: subcarpeta bajo `data/external/`

## Uso

- Ver datasets del config:
  - `(edita config/beir_eval.json para cambiar datasets)`

- Ejecutar evaluación (usa config):
  - `make beir-eval`
  - o `python -m ireranker.run_beir_eval`

- Overrides:
  - `make beir-eval ARGS="--dataset webis-touche2020"`
  - `make beir-eval ARGS="--light"` para saltar datasets pesados definidos en `light_exclude`
  - `python -m ireranker.run_beir_eval --dataset trec-covid --max-queries 200`
  - `python -m ireranker.run_beir_eval --config /ruta/a/custom.json`

## Salida

Por dataset: CSV `summary.csv` con filas por ranker y k
- Columnas: `ranker,k,NDCG,MAP,Recall,Precision`
- Carpeta: `reports/beir-metrics/<dataset>/` o `output_dir`

## Notas

- Si el directorio del dataset existe, no se descarga de nuevo.
- Tras descomprimir, se elimina el ZIP para ahorrar espacio.
- Errores de descarga se registran y se continúa con el siguiente dataset (se crea `ERROR.txt`).

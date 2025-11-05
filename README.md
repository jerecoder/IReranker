# IReranker

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este repo implementa distintos algoritmos de Information Retrieval para la parte de rerank en RAG.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- (Optional) Artifacts for learned rankers
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ireranker and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ireranker   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ireranker a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- (Optional) Data preparation helpers
    │
    ├── rankers                 <- Ranker interface and implementations
    │   ├── __init__.py
    │   ├── base.py
    │   ├── registry.py
    │   └── baselines.py        <- Simple built-in rankers
    │
    ├── evaluation              <- Metrics, runner, reporting
    │   ├── metrics.py
    │   ├── runner.py
    │   └── reporting.py
    │
    ├── data                    <- Dataset loaders
    │   └── loaders.py
    │
    ├── cli                     <- Typer CLIs
    │   ├── eval.py             <- Evaluate rankers and write reports
    │   └── rank.py             <- Run a single ranker
    │
    └── types.py                <- Typed containers for ranking tasks
```

--------

Quickstart

- Install: `pip install -e .`
- List rankers: `ireranker-rank list`
- Run synthetic eval: `ireranker-eval`

Notes

- Cookiecutter modeling stubs (train/predict) were removed to focus the repo on reranking algorithms and evaluation.


# TREC-COVID cross-paradigm efficiency experiment

This experiment compares four reranking approaches on the same TREC-COVID
BM25 top-100 candidates with `google/flan-t5-large`:

- Mohajer with the randomized-direction (sampling) pairwise oracle;
- PRP pairwise Heap Sort;
- setwise Heap Sort with three documents per prompt;
- RankGPT listwise sliding windows (`w=4`, `s=2`, repeats 1/3/5).

The primary efficiency axes are actual prompt plus completion/decoder tokens
per query and measured inference seconds per query. Raw calls are retained only
as a secondary diagnostic because calls contain different numbers of documents.
Every baseline receives the same 100-token passage truncation by default.

## Fresh inference

The runner deletes method result files before execution. Mohajer uses a separate
new comparison-cache file for every budget and deletes it before each run. The
Hugging Face model-weight cache is intentionally retained: loading a previously
downloaded immutable checkpoint does not skip measured model inference.

The baseline runner also supplies a non-secret placeholder `OPENAI_API_KEY`
because recent Pyserini releases eagerly import an optional OpenAI encoder. No
OpenAI API is used. Java temporary files are placed under the experiment folder
instead of the VM's potentially constrained `/dev/shm`.

## Setup and run

```bash
git clone https://github.com/ielab/llm-rankers.git external/llm-rankers
git -C external/llm-rankers checkout b36517bf17a3956dc56c4c967f972d02390b1cdd
pip install -e external/llm-rankers
pip install -e ".[beir]"
pip install torch transformers pyserini ir-datasets accelerate tiktoken sentencepiece

bash experiments/trec_covid_cross_paradigm/run_all_fresh.sh \
  2>&1 | tee trec-covid-cross-paradigm-live.log
```

The combined result is written to
`metrics/cross_paradigm_summary.csv`, and the final archive is
`trec-covid-cross-paradigm-results.tar.gz`.

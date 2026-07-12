# Ordered Mohajer hybrid probe

This is an exploratory, compute-capped screen for choosing where to run the
confirmatory cross-paradigm experiment. It uses live `google/flan-t5-large`
inference and never reads or writes a model-output cache.

## Pre-registered screen

Datasets are tried in this order:

1. TREC DL 2019
2. TREC DL 2020
3. DBpedia-Entity
4. FiQA
5. NFCorpus
6. TREC-COVID
7. SciFact
8. Webis-Touche 2020

Robust04 and TREC News are intentionally absent because their document
collections are licensed rather than automatically downloadable.

The screen freezes three queries per dataset by ascending
`sha256(dataset + ':' + query_id)` among queries that have query text, qrels,
and exactly 100 unique BM25 candidates. Every method receives the same BM25 top-100,
query text, rendered documents, FLAN-T5 checkpoint, truncation, generation
primitive, and end-to-end model-token budget. NDCG@10 uses the full qrels and
linear gain, matching `trec_eval`/`pytrec_eval`.

The arms are BM25, plain Mohajer with one query-stable randomized PRP direction,
Mohajer+Bubble, setwise, Mohajer->Setwise, listwise, Mohajer->Listwise, and a
bidirectional PRP Bubble baseline. Hybrid arms use one shared token cap: 80% is
available to Mohajer and 20% is reserved for top-20 refinement. Stage A and B
tokens are reported separately.

The ordered budgets are 100,000 then 50,000 model tokens per query. The larger
point is deliberately run first because smaller budgets can stop Mohajer before
its active-learning warmup pays off. A dataset is rejected after only the two
Mohajer arms when both trail BM25 by at least 0.03 on average and win zero of
three queries. Otherwise all arms run at 100k. The 50k arms run only when a
Mohajer-family arm improves at least 0.02 over BM25 and wins at least two
queries. The hybrids run before the standalone controls; if no Mohajer-family
arm meets that quality/win gate, those controls are skipped. Otherwise controls
run to test whether the candidate also lies on the observed quality/token Pareto
frontier. The first dataset to pass that final gate stops the global queue.

Three queries and one seed are selection evidence, not a paper result. The
selected dataset must be rerun on fresh queries with multiple seeds and all
standalone controls before making a claim.

## VM command

Do not switch branches or pull this code while the existing Robust04 process is
still running. Once that run is finished, fetch and check out this branch, then:

```bash
git fetch origin
git switch mohajer-hybrid-probe
git pull --ff-only

source .venv-a1fp/bin/activate
export TMPDIR="$PWD/.vm-cache/tmp"
export HF_HOME="$PWD/.vm-cache/huggingface"
mkdir -p "$TMPDIR" "$HF_HOME"

nohup bash experiments/mohajer_hybrid_probe/run_all_screen.sh \
  > mohajer-hybrid-probe.log 2>&1 &
echo $! > mohajer-hybrid-probe.pid
tail -f mohajer-hybrid-probe.log
```

The snapshot step downloads public BEIR data only when it is missing, extracts
only the 300 BM25 candidate documents needed per dataset, and deletes BEIR ZIPs
after extraction through the repository loader. DL 2019/2020 reuse the public
MS MARCO Pyserini index. The first run therefore has a data-download/setup cost;
the actual live inference screen is bounded separately.

For a disconnected session, resume completed conditions with:

```bash
RESUME=1 nohup bash experiments/mohajer_hybrid_probe/run_all_screen.sh \
  > mohajer-hybrid-probe-resume.log 2>&1 &
```

Completion markers include source, model, snapshot, CSV, and TREC-run hashes.
Resume skips only fully verified conditions. A condition has only three queries,
so an interrupted partial condition is safely rerun rather than mixed.

## Cost ceiling and outputs

The default rejection path costs at most 600k model tokens per dataset (two
Mohajer arms x three queries x 100k). A dataset that reaches every 100k arm uses
at most 2.1M tokens. If it qualifies, its seven 50k arms add at most 1.05M and
then the queue stops. With early stopping disabled, the full two-budget grid is
25.2M tokens.

The launcher produces:

- `results/per_query/`: atomic per-query metrics and completion markers
- `results/runs/`: TREC-format rankings
- `results/metrics/screen_summary.csv`: quality, token use, stage costs, timing,
  wins, and token-Pareto membership
- `results/metrics/recommendation.json`: first qualifying dataset/arm
- `results/screen_decisions.json`: every early-stop decision
- `mohajer-hybrid-probe-results.tar.gz` and its SHA-256 file

To deliberately run the entire grid, set `NO_EARLY_STOP=1`. Dataset and budget
subsets can be supplied as shell variables, for example:

```bash
DATASETS="dl-2019 fiqa" BUDGETS="100000 50000" \
  bash experiments/mohajer_hybrid_probe/run_all_screen.sh
```

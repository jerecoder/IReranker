# Robust04 fair cross-paradigm FLAN-T5-Large experiment

This experiment evaluates BM25, Mohajer, PRP, setwise Heap Sort, and RankGPT
listwise reranking on the same frozen Robust04 BM25 top-100 candidates.

## Controlled factors

Every LLM method uses:

- `google/flan-t5-large` revision `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
  in FP16 on the same GPU;
- one shared tokenizer and generation implementation;
- the same 32-token query and 100-token passage truncation defaults;
- a corrected attention mask and a hard 768-token rendered-prompt limit;
- greedy decoding and explicit output bounds;
- a frozen local text snapshot for the exact same 249 x 100 candidates;
- fresh live inference, with no comparison-output cache.

The measured process performs one unmeasured live warm-up for the sampled
pairwise, bidirectional pairwise batch, setwise, and listwise prompt shapes so
CUDA initialization is not charged only to the first method.

Mohajer uses its intended sampling oracle. Each logical comparison chooses one
seeded orientation and executes one instance of the exact PRP prompt. PRP uses
the same primitive but evaluates both orientations atomically in one batch.
Setwise compares up to three documents per prompt. RankGPT uses windows of four
with step two and up to five repeats.

The 768-token prompt ceiling preserves all four 100-token listwise passages and
the ranking instruction instead of silently applying the tokenizer's 512-token
whole-prompt truncation. Preflight verifies the bound against the frozen data.

The primary budgets are total model tokens per query: non-padding encoder tokens
plus every decoder token actually processed. A prospective atomic request is
rejected before inference if its maximum possible cost would exceed the budget.
The ranking algorithm then emits its current best-effort full permutation.

## Metrics

Per query, the experiment records:

- trec_eval-compatible NDCG@10 (linear graded gains) using all official qrels
  for the ideal ranking;
- logical comparisons or choice events;
- prompt instances, document instances, and generation invocations;
- non-padding encoder tokens and padded encoder slots;
- decoder and total model tokens;
- synchronized GPU inference time and query wall time;
- peak GPU memory and invalid outputs.

Mohajer runs seeds `42 43 44 45 46`; deterministic methods run seed 42.
Analysis averages stochastic seeds per query before paired query bootstrap,
uses sign-flip tests with Holm correction, and reports token/GPU-time Pareto
fronts with resource confidence intervals.

## VM setup

```bash
git switch robust04-cross-paradigm
git pull --ff-only
source .venv-a1fp/bin/activate

pip install -e ".[beir]"
pip install torch transformers accelerate sentencepiece pyserini matplotlib
```

Pyserini is used once to materialize text for the already-selected local BM25
document IDs. It does not run BM25. On a VM with a small home filesystem, point
its fixed cache at the large workspace disk:

```bash
mkdir -p "$PWD/.vm-cache/pyserini" "$HOME/.cache"
rm -rf "$HOME/.cache/pyserini"
ln -s "$PWD/.vm-cache/pyserini" "$HOME/.cache/pyserini"
```

## Preflight and full run

The full runner first rebuilds public Robust04 topics/qrels, freezes the local
top-100 candidate snapshot, checks rendered prompt lengths, and performs live
end-to-end four-document smoke rankings through Mohajer, PRP, setwise, and
listwise paths. No measured run starts unless every permutation and output
parser passes this preflight.

```bash
nohup bash experiments/robust04_cross_paradigm/run_all_fresh.sh \
  > robust04-cross-paradigm-live.log 2>&1 &
echo $! > robust04-cross-paradigm.pid
tail -f robust04-cross-paradigm-live.log
```

Resume after interruption. Matching partial conditions continue at the first
missing query; completed conditions are hash-checked and skipped. A changed
model, snapshot, truncation, source implementation, GPU, or query set is rejected
instead of being mixed into old results:

```bash
RESUME=1 nohup bash experiments/robust04_cross_paradigm/run_all_fresh.sh \
  > robust04-cross-paradigm-resume.log 2>&1 &
```

For an inexpensive one-query validation before the complete run:

```bash
python experiments/robust04_cross_paradigm/run_experiment.py \
  --methods bm25 mohajer prp setwise listwise \
  --token-budgets 25000 --seeds 42 --max-queries 1
```

The full output archive is `robust04-cross-paradigm-results.tar.gz` with a
matching `.sha256` file. The archive intentionally excludes the frozen Robust04
document texts because the underlying corpus is licensed. Do not publish or
redistribute `data/external/robust04_cross_paradigm/documents.jsonl`.

# Reviewer-response experiments

This directory runs two frozen, token-normalized experiments on the same 20
previously unseen DBpedia-Entity queries. The three pilot queries are excluded
by manifest and never enter either reported result.

## Experiment 1: cross-paradigm comparison

The primary method is plain Mohajer with a randomized-direction pairwise PRP
oracle. It is compared with standard randomized PRP-Heapsort, standard Setwise,
standard Listwise, and BM25. No active-prefix hybrid belongs to Experiment 1.

The primary pre-registered comparisons at the 100k model-token cap are Mohajer
versus PRP, Setwise, and Listwise. The 50k cap is a secondary efficiency point.

## Experiment 2: active-prefix Setwise/Listwise

Experiment 2 introduces Mohajer-guided active-prefix Setwise and Listwise. A
pairwise Mohajer stage identifies and orders a promising prefix; only its top 20
documents receive the multi-document refinement. These are hybrid active-prefix
methods, not theoretical k-ary versions of the Mohajer algorithm.

**Protocol correction:** this completed cascade study is not the requested true
active-Setwise/Listwise experiment and must not be labeled as such in the paper.
The corrected implementation lives in `active_multiway.py`: it applies
Mohajer's strided group tournaments and top-K champion extraction directly with
a multi-document oracle. `active_setwise.py` uses only randomized-presentation
three-document Setwise prompts; it has no pairwise preprocessing or polishing
stage. The analogous strict Listwise primitive remains gated on the output
validity/equivariance diagnostics below.

Experiment 2 reuses the exact, hash-verified BM25, Mohajer, Setwise, and Listwise
conditions from Experiment 1. It runs only the two new hybrid arms. Its primary
comparison is active-prefix Listwise versus standard Listwise at 100k; active
Setwise is retained even though its pilot was unfavorable.

## Frozen protocol

- Dataset: DBpedia-Entity.
- Queries: 20 eligible queries selected by the next SHA-256 positions after the
  three pilot queries.
- Candidate set: identical BM25 top-100.
- Model: `google/flan-t5-large` at revision
  `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`.
- Token caps: 100k primary, 50k secondary.
- Stochastic seeds: 42, 43, and 44 for Mohajer, randomized PRP, and hybrids.
- Deterministic Setwise/Listwise: one run at seed 42.
- Hybrid allocation: at most 80% for pairwise Mohajer, with the total shared cap
  restored for top-20 refinement.
- Primary quality metric: linear-gain NDCG@10 using full qrels.
- Efficiency: actual model tokens and measured GPU inference time. Raw prompt
  counts are reported descriptively but never compared as equivalent costs.
- Statistics: query-paired bootstrap confidence intervals, paired sign-flip
  tests, and Holm correction within each experiment.
- Model-output cache: disabled.

Twenty queries and shared controls are a deliberate overnight/runtime tradeoff.
Experiment 1 contains 320 LLM query-runs; Experiment 2 adds 240 rather than
rerunning its controls. Based on the observed T5-large VM throughput, each phase
is targeted below four hours, but the launcher prioritizes complete conditions
over killing a nearly finished phase at an arbitrary deadline.
The launcher records actual wall time for Experiment 1 and the incremental
hybrid portion of Experiment 2. It does not terminate a nearly complete method
at an arbitrary wall-clock threshold.

## One overnight VM command

```bash
git fetch origin
git switch reviewer-response-experiments
git pull --ff-only
source .venv-a1fp/bin/activate

nohup bash experiments/reviewer_response/run_both_overnight.sh \
  > reviewer-response-overnight.log 2>&1 &
echo $! > reviewer-response.pid
tail -f reviewer-response-overnight.log
```

The launcher compiles/import-checks the analysis before inference, prepares and
hashes the unseen snapshot, performs live smoke tests for every method family,
runs both experiments with one loaded model, verifies every condition, performs
both analyses, and creates `reviewer-response-results.tar.gz` plus a SHA-256
file. If interrupted, resume without deleting completed verified conditions:

```bash
RESUME=1 nohup bash experiments/reviewer_response/run_both_overnight.sh \
  > reviewer-response-resume.log 2>&1 &
```

Progress and failure information is written atomically to
`experiments/reviewer_response/results/overnight_status.json` after every query.

## Listwise output-validity diagnostic

The first overnight run showed that every Listwise generation failed the strict
complete-permutation requirement. Before rerunning Listwise, use the diagnostic
to capture exact raw generations for three hypotheses: the original prompt with
20 output tokens, the same prompt with 64 output tokens, and a compact prompt
with 32 output tokens. It never accesses qrel values or computes NDCG.

```bash
nohup bash experiments/reviewer_response/run_listwise_diagnostic.sh \
  > listwise-diagnostic.log 2>&1 &
tail -f listwise-diagnostic.log
```

The output contains every prompt, raw model string, strict parse decision, and
the legacy repaired permutation. No Listwise result should be rerun or reported
until a frozen protocol achieves 100% strict validity on this diagnostic and a
separate preflight sample.

If the compact prompt only copies its concrete example, run the follow-up
equivariance test. It removes all concrete output examples and evaluates three
symbolic prompt styles. Every window is presented in its original order, with
documents shuffled, and with identifiers reassigned; outputs are mapped back to
document IDs before measuring agreement.

```bash
nohup bash experiments/reviewer_response/run_listwise_equivariance_test.sh \
  > listwise-equivariance.log 2>&1 &
tail -f listwise-equivariance.log
```

A candidate passes only with 100% strict permutations and top-document
agreement on at least 8/9 cases under both perturbations. The test uses no qrel
values and computes no effectiveness metric.

If ordinary generation fails that gate, test greedy decoding constrained to the
complete permutation trie (24 outputs for four documents):

```bash
nohup bash experiments/reviewer_response/run_listwise_constrained_diagnostic.sh \
  > listwise-constrained-diagnostic.log 2>&1 &
tail -f listwise-constrained-diagnostic.log
```

The shell always archives completed diagnostic records, even when no prompt
passes. An effectiveness run remains blocked until at least one constrained
variant satisfies the same strict validity and equivariance rule.

## Corrected one-seed Active-Setwise quick experiment

The corrected scheduler can be tested immediately with seed 42 on all 20 frozen
queries and both token caps. It compares conventional fixed-presentation
Setwise, the same standard scheduler with randomized presentation, and true
Mohajer-style Active-Setwise with randomized presentation. Every choice event
contains at most three documents. BM25 and the frozen snapshot are reused, and
no pairwise prompt is permitted by a runtime invariant.

```bash
nohup bash experiments/reviewer_response/run_true_active_setwise_quick.sh \
  > true-active-setwise-quick.log 2>&1 &
tail -f true-active-setwise-quick.log
```

This is an exploratory one-seed decision run. If its 100k result is promising,
only seeds 43 and 44 need to be added. The expected runtime is comfortably under
one hour based on the previous Setwise measurements.

To extend a completed corrected run, reuse BM25, fixed Setwise, and seed 42;
run only randomized and Active-Setwise at 50k for seeds 43 and 44:

```bash
nohup bash experiments/reviewer_response/run_true_active_setwise_extra_seeds.sh \
  > true-active-setwise-extra-seeds.log 2>&1 &
tail -f true-active-setwise-extra-seeds.log
```

The extension verifies all reusable signatures and hashes before loading the
model, resumes at condition boundaries, and writes a three-seed aggregate under
`metrics/multiseed`.

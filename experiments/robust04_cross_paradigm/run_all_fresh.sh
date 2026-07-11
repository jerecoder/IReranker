#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="experiments/robust04_cross_paradigm"
RESULTS_DIR="${EXP_DIR}/results"
MODEL="${MODEL:-google/flan-t5-large}"
MODEL_REVISION="${MODEL_REVISION:-0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a}"
DEVICE="${DEVICE:-cuda}"
QUERY_TOKENS="${QUERY_TOKENS:-32}"
PASSAGE_TOKENS="${PASSAGE_TOKENS:-100}"
ENCODER_MAX_TOKENS="${ENCODER_MAX_TOKENS:-768}"
TOKEN_BUDGETS="${TOKEN_BUDGETS:-25000 50000 75000 100000 125000}"
SEEDS="${SEEDS:-42 43 44 45 46}"
RESUME="${RESUME:-0}"

mkdir -p "${EXP_DIR}/java-tmp"
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-local-flan-t5-run}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Djava.io.tmpdir=${PWD}/${EXP_DIR}/java-tmp}"

python "${EXP_DIR}/prepare_snapshot.py"

if [[ "${RESUME}" != "1" ]]; then
  rm -rf "${RESULTS_DIR}"
fi

python "${EXP_DIR}/preflight.py" \
  --model "${MODEL}" \
  --model-revision "${MODEL_REVISION}" \
  --device "${DEVICE}" \
  --query-tokens "${QUERY_TOKENS}" \
  --passage-tokens "${PASSAGE_TOKENS}" \
  --encoder-max-tokens "${ENCODER_MAX_TOKENS}"

resume_arg=()
if [[ "${RESUME}" == "1" ]]; then
  resume_arg=(--resume)
fi

python "${EXP_DIR}/run_experiment.py" \
  --model "${MODEL}" \
  --model-revision "${MODEL_REVISION}" \
  --device "${DEVICE}" \
  --query-tokens "${QUERY_TOKENS}" \
  --passage-tokens "${PASSAGE_TOKENS}" \
  --encoder-max-tokens "${ENCODER_MAX_TOKENS}" \
  --token-budgets ${TOKEN_BUDGETS} \
  --seeds ${SEEDS} \
  "${resume_arg[@]}"

python "${EXP_DIR}/analyze.py" \
  --token-budgets ${TOKEN_BUDGETS} \
  --seeds ${SEEDS}

tar -czf robust04-cross-paradigm-results.tar.gz \
  "${RESULTS_DIR}" \
  data/external/robust04_cross_paradigm/manifest.json \
  data/external/robust04_cross_paradigm/queries.jsonl \
  data/external/robust04_cross_paradigm/bm25.robust04.top100.txt

sha256sum robust04-cross-paradigm-results.tar.gz \
  > robust04-cross-paradigm-results.tar.gz.sha256
echo "Saved robust04-cross-paradigm-results.tar.gz"

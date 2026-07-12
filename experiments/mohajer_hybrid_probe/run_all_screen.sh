#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="experiments/mohajer_hybrid_probe"
RESULTS_DIR="${EXP_DIR}/results"
MODEL="${MODEL:-google/flan-t5-large}"
MODEL_REVISION="${MODEL_REVISION:-0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a}"
DEVICE="${DEVICE:-cuda}"
QUERY_TOKENS="${QUERY_TOKENS:-32}"
PASSAGE_TOKENS="${PASSAGE_TOKENS:-100}"
ENCODER_MAX_TOKENS="${ENCODER_MAX_TOKENS:-768}"
DATASETS="${DATASETS:-dl-2019 dl-2020 dbpedia-entity fiqa nfcorpus trec-covid scifact webis-touche2020}"
BUDGETS="${BUDGETS:-100000 50000}"
SEED="${SEED:-42}"
QUERIES_PER_DATASET="${QUERIES_PER_DATASET:-3}"
RESUME="${RESUME:-0}"
NO_EARLY_STOP="${NO_EARLY_STOP:-0}"

mkdir -p "${EXP_DIR}/java-tmp"
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-local-flan-t5-run}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Djava.io.tmpdir=${PWD}/${EXP_DIR}/java-tmp}"

python "${EXP_DIR}/prepare_snapshots.py" \
  --datasets ${DATASETS} \
  --queries-per-dataset "${QUERIES_PER_DATASET}"

if [[ "${RESUME}" != "1" ]]; then
  rm -rf "${RESULTS_DIR}"
fi

python "${EXP_DIR}/preflight.py" \
  --datasets ${DATASETS} \
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

early_stop_arg=()
if [[ "${NO_EARLY_STOP}" == "1" ]]; then
  early_stop_arg=(--no-early-stop)
fi

python "${EXP_DIR}/run_probe.py" \
  --datasets ${DATASETS} \
  --budgets ${BUDGETS} \
  --seed "${SEED}" \
  --model "${MODEL}" \
  --model-revision "${MODEL_REVISION}" \
  --device "${DEVICE}" \
  --query-tokens "${QUERY_TOKENS}" \
  --passage-tokens "${PASSAGE_TOKENS}" \
  --encoder-max-tokens "${ENCODER_MAX_TOKENS}" \
  "${resume_arg[@]}" \
  "${early_stop_arg[@]}"

python "${EXP_DIR}/analyze_probe.py"

archive="mohajer-hybrid-probe-results.tar.gz"
tar -czf "${archive}" \
  "${RESULTS_DIR}" \
  data/external/mohajer_hybrid_probe
sha256sum "${archive}" > "${archive}.sha256"
echo "Saved ${archive}"

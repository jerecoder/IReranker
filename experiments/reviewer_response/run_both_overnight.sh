#!/usr/bin/env bash
set -Eeuo pipefail

EXP_DIR="experiments/reviewer_response"
RESULTS_DIR="${EXP_DIR}/results"
MODEL="${MODEL:-google/flan-t5-large}"
MODEL_REVISION="${MODEL_REVISION:-0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a}"
DEVICE="${DEVICE:-cuda}"
QUERY_TOKENS="${QUERY_TOKENS:-32}"
PASSAGE_TOKENS="${PASSAGE_TOKENS:-100}"
ENCODER_MAX_TOKENS="${ENCODER_MAX_TOKENS:-768}"
RESUME="${RESUME:-0}"
ARCHIVE="reviewer-response-results.tar.gz"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-local-flan-t5-run}"
mkdir -p "${EXP_DIR}/java-tmp" "${PWD}/.vm-cache/tmp" "${PWD}/.vm-cache/huggingface"
export TMPDIR="${TMPDIR:-${PWD}/.vm-cache/tmp}"
export HF_HOME="${HF_HOME:-${PWD}/.vm-cache/huggingface}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Djava.io.tmpdir=${PWD}/${EXP_DIR}/java-tmp}"

on_exit() {
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "OVERNIGHT RUN FAILED with exit code ${rc}"
    echo "Inspect ${RESULTS_DIR}/overnight_status.json and the current log."
  fi
}
trap on_exit EXIT

python -m compileall -q "${EXP_DIR}"
python -c "import experiments.reviewer_response.analyze"
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

run_inference() {
  python "${EXP_DIR}/run_experiments.py" \
    --model "${MODEL}" \
    --model-revision "${MODEL_REVISION}" \
    --device "${DEVICE}" \
    --query-tokens "${QUERY_TOKENS}" \
    --passage-tokens "${PASSAGE_TOKENS}" \
    --encoder-max-tokens "${ENCODER_MAX_TOKENS}" \
    "$@"
}

if ! run_inference "${resume_arg[@]}"; then
  echo "Inference failed once; retrying from verified completion markers in 15 seconds."
  sleep 15
  run_inference --resume
fi

python "${EXP_DIR}/analyze.py" --experiment both

tar -czf "${ARCHIVE}" \
  "${RESULTS_DIR}" \
  data/external/reviewer_response/dbpedia-entity
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

echo "OVERNIGHT RUN COMPLETE"
echo "Experiment 1: ${RESULTS_DIR}/metrics/experiment_1/summary.csv"
echo "Experiment 2: ${RESULTS_DIR}/metrics/experiment_2/summary.csv"
echo "Archive: ${ARCHIVE}"

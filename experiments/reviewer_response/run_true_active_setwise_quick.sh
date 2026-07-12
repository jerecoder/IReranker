#!/usr/bin/env bash
set -Eeuo pipefail

EXP_DIR="experiments/reviewer_response"
OUTPUT_DIR="${EXP_DIR}/results/true_active_setwise_single_seed"
MODEL="${MODEL:-google/flan-t5-large}"
MODEL_REVISION="${MODEL_REVISION:-0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-0}"
ARCHIVE="true-active-setwise-quick-results.tar.gz"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${EXP_DIR}/java-tmp" "${PWD}/.vm-cache/tmp" "${PWD}/.vm-cache/huggingface"
export TMPDIR="${TMPDIR:-${PWD}/.vm-cache/tmp}"
export HF_HOME="${HF_HOME:-${PWD}/.vm-cache/huggingface}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Djava.io.tmpdir=${PWD}/${EXP_DIR}/java-tmp}"

python -m compileall -q "${EXP_DIR}"
python "${EXP_DIR}/prepare_snapshot.py"
if [[ "${RESUME}" != "1" ]]; then
  rm -rf "${OUTPUT_DIR}"
fi
resume_arg=()
if [[ "${RESUME}" == "1" ]]; then
  resume_arg=(--resume)
fi

run_experiment() {
  python "${EXP_DIR}/run_true_active_setwise_quick.py" \
    --model "${MODEL}" \
    --model-revision "${MODEL_REVISION}" \
    --device "${DEVICE}" \
    "$@"
}

if ! run_experiment "${resume_arg[@]}"; then
  echo "First attempt failed; retrying once from verified condition checkpoints." >&2
  sleep 15
  run_experiment --resume
fi

tar -czf "${ARCHIVE}" "${OUTPUT_DIR}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
echo "Saved ${ARCHIVE}"

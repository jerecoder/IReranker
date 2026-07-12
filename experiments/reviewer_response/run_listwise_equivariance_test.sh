#!/usr/bin/env bash
set -Eeuo pipefail

EXP_DIR="experiments/reviewer_response"
MODEL="${MODEL:-google/flan-t5-large}"
MODEL_REVISION="${MODEL_REVISION:-0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a}"
DEVICE="${DEVICE:-cuda}"
ARCHIVE="listwise-equivariance-results.tar.gz"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${EXP_DIR}/java-tmp" "${PWD}/.vm-cache/tmp" "${PWD}/.vm-cache/huggingface"
export TMPDIR="${TMPDIR:-${PWD}/.vm-cache/tmp}"
export HF_HOME="${HF_HOME:-${PWD}/.vm-cache/huggingface}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Djava.io.tmpdir=${PWD}/${EXP_DIR}/java-tmp}"

python -m compileall -q "${EXP_DIR}"
python "${EXP_DIR}/prepare_snapshot.py"
rm -rf "${EXP_DIR}/results/listwise_equivariance_diagnostic"
python "${EXP_DIR}/diagnose_listwise_equivariance.py" \
  --model "${MODEL}" \
  --model-revision "${MODEL_REVISION}" \
  --device "${DEVICE}"

tar -czf "${ARCHIVE}" "${EXP_DIR}/results/listwise_equivariance_diagnostic"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
echo "Saved ${ARCHIVE}"

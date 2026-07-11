#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="experiments/trec_covid_cross_paradigm"
RUNNER="${LLM_RANKERS_RUNNER:-external/llm-rankers/run.py}"
MODEL="${MODEL:-google/flan-t5-large}"
DEVICE="${DEVICE:-cuda}"
PASSAGE_LENGTH="${PASSAGE_LENGTH:-100}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
BM25="${EXP_DIR}/runs/bm25.trec-covid.txt"
INDEX="beir-v1.0.0-trec-covid"

mkdir -p "${EXP_DIR}/runs" "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/java-tmp"

# Recent Pyserini imports its optional OpenAI encoder eagerly. These experiments
# use only a local Hugging Face model, but the import still requires a non-empty
# value. This placeholder is never sent anywhere. Keep Java temporary files on
# the workspace disk because VM /dev/shm may be too small.
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-local-flan-t5-run}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Djava.io.tmpdir=${PWD}/${EXP_DIR}/java-tmp}"

python "${EXP_DIR}/patch_llm_rankers_compat.py" "${RUNNER}"

common=(run --model_name_or_path "${MODEL}" --tokenizer_name_or_path "${MODEL}"
  --run_path "${BM25}" --pyserini_index "${INDEX}" --hits 100
  --query_length 32 --scoring generation --device "${DEVICE}")

run_one() {
  local name="$1"; shift
  local output="${EXP_DIR}/runs/${name}.txt"
  local log="${EXP_DIR}/logs/${name}.log"
  if [[ "${SKIP_COMPLETED}" == "1" && -s "${output}" && -s "${log}" ]] \
      && grep -q "Avg time per query:" "${log}"; then
    echo "SKIP: completed ${name}"
    return
  fi
  rm -f "${output}" "${log}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${RUNNER}" \
    "${common[@]}" --save_path "${output}" "$@" 2>&1 | tee "${log}"
}

# Every method starts without a prior result or inference cache. Hugging Face
# model weights remain cached, which does not change measured inference cost.
run_one pairwise.prp.heapsort --passage_length "${PASSAGE_LENGTH}" pairwise --method heapsort --k 10
run_one setwise.heapsort.c3 --passage_length "${PASSAGE_LENGTH}" setwise --num_child 2 --method heapsort --k 10
for repeat in 1 3 5; do
  run_one "listwise.rankgpt.w4s2.r${repeat}" --passage_length "${PASSAGE_LENGTH}" \
    listwise --window_size 4 --step_size 2 --num_repeat "${repeat}"
done

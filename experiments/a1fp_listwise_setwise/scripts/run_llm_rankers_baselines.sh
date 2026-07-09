#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
CUDA_ID="${CUDA_VISIBLE_DEVICES:-0}"
EXP_DIR="experiments/a1fp_listwise_setwise"
RUNNER="external/llm-rankers/run.py"
MODEL="google/flan-t5-xl"

mkdir -p "${EXP_DIR}/runs" "${EXP_DIR}/logs" "${EXP_DIR}/metrics"

run_listwise() {
  local dataset="$1"
  local ir_dataset="$2"
  local repeat="$3"
  local run_path="${EXP_DIR}/runs/bm25.${dataset}.txt"
  local save_path="${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r${repeat}.${dataset}.txt"
  local log_path="${EXP_DIR}/logs/listwise.rankgpt.flant5xl.w4s2.r${repeat}.${dataset}.log"

  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" "${RUNNER}" \
    run \
      --model_name_or_path "${MODEL}" \
      --tokenizer_name_or_path "${MODEL}" \
      --run_path "${run_path}" \
      --save_path "${save_path}" \
      --ir_dataset_name "${ir_dataset}" \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
      --device cuda \
    listwise \
      --window_size 4 \
      --step_size 2 \
      --num_repeat "${repeat}" \
    2>&1 | tee "${log_path}"
}

run_setwise() {
  local dataset="$1"
  local ir_dataset="$2"
  local run_path="${EXP_DIR}/runs/bm25.${dataset}.txt"
  local save_path="${EXP_DIR}/runs/setwise.heapsort.flant5xl.c3.${dataset}.txt"
  local log_path="${EXP_DIR}/logs/setwise.heapsort.flant5xl.c3.${dataset}.log"

  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" "${RUNNER}" \
    run \
      --model_name_or_path "${MODEL}" \
      --tokenizer_name_or_path "${MODEL}" \
      --run_path "${run_path}" \
      --save_path "${save_path}" \
      --ir_dataset_name "${ir_dataset}" \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
      --device cuda \
    setwise \
      --num_child 2 \
      --method heapsort \
      --k 10 \
    2>&1 | tee "${log_path}"
}

for repeat in 1 3 5; do
  run_listwise "dl19" "msmarco-passage/trec-dl-2019" "${repeat}"
  run_listwise "dl20" "msmarco-passage/trec-dl-2020" "${repeat}"
done

run_setwise "dl19" "msmarco-passage/trec-dl-2019"
run_setwise "dl20" "msmarco-passage/trec-dl-2020"

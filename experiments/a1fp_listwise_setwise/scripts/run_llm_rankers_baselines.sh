#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
CUDA_ID="${CUDA_VISIBLE_DEVICES:-0}"
EXP_DIR="${EXP_DIR:-experiments/a1fp_listwise_setwise}"
RUNNER="${LLM_RANKERS_RUNNER:-external/llm-rankers/run.py}"
MODEL="${MODEL:-google/flan-t5-xl}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN=0
SKIP_PREFLIGHT=0

usage() {
  cat <<'USAGE'
Usage: run_llm_rankers_baselines.sh [--dry-run] [--skip-preflight]

Runs the A1fp listwise/setwise llm-rankers baselines.

Options:
  --dry-run          Run preflight checks only; do not launch generation.
  --skip-preflight  Launch generation without preflight checks.
  -h, --help        Show this help.

Environment overrides:
  PYTHON                 Python executable to use.
  CUDA_VISIBLE_DEVICES   CUDA device mask, default: 0.
  MODEL                  HF model id or local model path, default: google/flan-t5-xl.
  DEVICE                 llm-rankers device, default: cuda.
  LLM_RANKERS_RUNNER     Path to external/llm-rankers/run.py.
  REQUIRE_LOCAL_MODEL=1  Fail preflight if MODEL config/tokenizer are not cached locally.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

info() {
  echo "OK: $*"
}

warn() {
  echo "WARN: $*" >&2
}

fail() {
  echo "FAIL: $*" >&2
}

require_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    info "found ${path}"
    return 0
  fi
  fail "missing required file: ${path}"
  return 1
}

require_command() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    info "found command ${command_name}"
    return 0
  fi
  fail "missing command: ${command_name}"
  return 1
}

warn_existing_outputs() {
  local output
  for output in \
    "${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r1.dl19.txt" \
    "${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r3.dl19.txt" \
    "${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r5.dl19.txt" \
    "${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r1.dl20.txt" \
    "${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r3.dl20.txt" \
    "${EXP_DIR}/runs/listwise.rankgpt.flant5xl.w4s2.r5.dl20.txt" \
    "${EXP_DIR}/runs/setwise.heapsort.flant5xl.c3.dl19.txt" \
    "${EXP_DIR}/runs/setwise.heapsort.flant5xl.c3.dl20.txt"; do
    if [[ -e "${output}" ]]; then
      warn "output already exists and may be overwritten: ${output}"
    fi
  done
}

python_preflight() {
  A1FP_DEVICE="${DEVICE}" \
  A1FP_MODEL="${MODEL}" \
  A1FP_BM25_DL19="${EXP_DIR}/runs/bm25.dl19.txt" \
  A1FP_BM25_DL20="${EXP_DIR}/runs/bm25.dl20.txt" \
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" \
    "${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

from collections import defaultdict
import importlib
import os
from pathlib import Path
import shutil
import sys


failures: list[str] = []


def ok(message: str) -> None:
    print(f"OK: {message}")


def warn(message: str) -> None:
    print(f"WARN: {message}", file=sys.stderr)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    failures.append(message)


def short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


def import_required_modules() -> dict[str, object]:
    loaded: dict[str, object] = {}
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ is required; current interpreter is Python {version}")
    else:
        ok(f"Python {version}")
    if sys.version_info[:2] != (3, 10):
        warn(f"experiment setup was validated with Python 3.10; current interpreter is {version}")

    for module_name in (
        "torch",
        "transformers",
        "ir_datasets",
        "pyserini",
        "accelerate",
        "sentencepiece",
        "tiktoken",
    ):
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - preflight should report all import failures.
            fail(f"cannot import {module_name}: {short_error(exc)}")
        else:
            ok(f"import {module_name}")
    return loaded


def check_cuda(torch_module: object | None) -> None:
    device = os.environ["A1FP_DEVICE"]
    if device != "cuda":
        warn(f"DEVICE={device}; official baseline commands were validated for DEVICE=cuda")
        return

    if torch_module is None:
        return

    try:
        cuda_available = torch_module.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot query torch CUDA availability: {short_error(exc)}")
        return

    if not cuda_available:
        fail("DEVICE=cuda but torch.cuda.is_available() is false")
        return

    try:
        device_name = torch_module.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        device_name = f"unknown device name ({short_error(exc)})"
    ok(f"CUDA is available through torch; visible device 0 is {device_name}")

    try:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        warn(f"could not query CUDA memory: {short_error(exc)}")
        return

    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    ok(f"CUDA memory: {free_gib:.1f} GiB free / {total_gib:.1f} GiB total")
    if total_gib < 12:
        warn("GPU has less than 12 GiB total memory; Flan-T5-XL generation may OOM")


def check_model(transformers_module: object | None) -> None:
    if transformers_module is None:
        return

    model_name = os.environ["A1FP_MODEL"]
    require_local = os.environ.get("REQUIRE_LOCAL_MODEL") == "1"

    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(model_name, local_files_only=True)
        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        message = (
            f"MODEL={model_name} config/tokenizer are not available locally: "
            f"{short_error(exc)}. The real run will try to download them unless offline mode is set."
        )
        if require_local:
            fail(message)
        else:
            warn(message)
    else:
        ok(f"MODEL={model_name} config/tokenizer are available locally")


def check_ir_datasets(ir_datasets_module: object | None) -> None:
    if ir_datasets_module is None:
        return

    for dataset_name in ("msmarco-passage/trec-dl-2019", "msmarco-passage/trec-dl-2020"):
        try:
            ir_datasets_module.load(dataset_name)
        except Exception as exc:  # noqa: BLE001
            fail(f"ir_datasets cannot resolve {dataset_name}: {short_error(exc)}")
        else:
            ok(f"ir_datasets resolves {dataset_name}")


def check_bm25_run(path: Path, expected_hits: int = 100) -> None:
    if not path.exists():
        return

    counts: dict[str, int] = defaultdict(int)
    malformed: list[int] = []
    with path.open("r", encoding="utf-8") as run_file:
        for line_number, line in enumerate(run_file, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                malformed.append(line_number)
                continue
            counts[parts[0]] += 1

    if malformed:
        fail(f"{path} has malformed TREC lines at {malformed[:5]}")
        return
    if not counts:
        fail(f"{path} has no run rows")
        return

    bad_counts = {qid: count for qid, count in counts.items() if count != expected_hits}
    if bad_counts:
        sample = dict(list(bad_counts.items())[:5])
        fail(f"{path} should have exactly {expected_hits} hits per query; sample bad counts: {sample}")
        return

    ok(f"{path} has {sum(counts.values())} rows across {len(counts)} queries")


def check_disk() -> None:
    free_gib = shutil.disk_usage(Path.cwd()).free / 1024**3
    ok(f"workspace disk free: {free_gib:.1f} GiB")
    if free_gib < 25:
        warn("less than 25 GiB free; first-time model or MSMARCO downloads may fail")


loaded_modules = import_required_modules()
check_cuda(loaded_modules.get("torch"))
check_model(loaded_modules.get("transformers"))
check_ir_datasets(loaded_modules.get("ir_datasets"))
check_bm25_run(Path(os.environ["A1FP_BM25_DL19"]))
check_bm25_run(Path(os.environ["A1FP_BM25_DL20"]))
check_disk()

if failures:
    sys.exit(1)
PY
}

preflight() {
  local failed=0

  echo "== A1fp listwise/setwise preflight =="
  require_command "${PYTHON_BIN}" || failed=1
  require_file "${RUNNER}" || failed=1
  require_file "${EXP_DIR}/runs/bm25.dl19.txt" || failed=1
  require_file "${EXP_DIR}/runs/bm25.dl20.txt" || failed=1

  if [[ ! -d "${EXP_DIR}" ]]; then
    fail "missing experiment directory: ${EXP_DIR}"
    failed=1
  elif [[ ! -w "${EXP_DIR}" ]]; then
    fail "experiment directory is not writable: ${EXP_DIR}"
    failed=1
  else
    info "experiment directory is writable: ${EXP_DIR}"
  fi

  warn_existing_outputs

  if ! python_preflight; then
    failed=1
  fi

  if [[ "${failed}" -ne 0 ]]; then
    echo
    fail "preflight failed; fix the issues above before launching generation"
    return 1
  fi

  echo "Preflight passed."
}

if [[ "${SKIP_PREFLIGHT}" -eq 0 || "${DRY_RUN}" -eq 1 ]]; then
  preflight
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete. No baseline generation was launched."
  exit 0
fi

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
      --device "${DEVICE}" \
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
      --device "${DEVICE}" \
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

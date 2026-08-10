#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/evaluation/gemma4_bakeoff.json}"
RUN_ID="${RUN_ID:-gemma4-bakeoff-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$HOME/kinyalm-bakeoffs/$RUN_ID}"
RUNTIME_DIR="${KINYALM_BAKEOFF_RUNTIME:-$HOME/.cache/kinyalm/gemma4-bakeoff}"
VENV_DIR="$RUNTIME_DIR/venv"
STATUS_FILE="$RUN_ROOT/status"
SYSTEM_INFO="$RUN_ROOT/system-info.txt"
RUN_LOG="$RUN_ROOT/bakeoff.log"
PUBLISH_LOG="$RUN_ROOT/publish.log"
PUBLISH_REPO="${PUBLISH_REPO:-kinyalm/kinyalm-data-lake}"
PUBLISH_PATH="${PUBLISH_PATH:-evaluation/model-bakeoffs/$RUN_ID}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-1}"
TASK_LIMIT="${TASK_LIMIT:-}"
CANDIDATE_ID="${CANDIDATE_ID:-}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-76000}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"
ACCELERATE_VERSION="${ACCELERATE_VERSION:-1.14.0}"
PEFT_VERSION="${PEFT_VERSION:-0.20.0}"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-1.24.0}"
SAFETENSORS_VERSION="${SAFETENSORS_VERSION:-0.8.0}"
PILLOW_VERSION="${PILLOW_VERSION:-12.0.0}"
SENTENCEPIECE_VERSION="${SENTENCEPIECE_VERSION:-0.2.2}"

if [[ -n "${KINYALM_HF_TOKEN_FILE:-}" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(<"$KINYALM_HF_TOKEN_FILE")"
fi
if [[ "$PUBLISH_RESULTS" != "0" && "$PUBLISH_RESULTS" != "1" ]]; then
  echo "PUBLISH_RESULTS must be 0 or 1." >&2
  exit 2
fi
if [[ -n "$TASK_LIMIT" && ! "$TASK_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "TASK_LIMIT must be a positive integer when provided." >&2
  exit 2
fi
if [[ "$PUBLISH_RESULTS" == "1" && -n "${KINYALM_HF_PUBLISH_TOKEN_FILE:-}" ]]; then
  export HF_PUBLISH_TOKEN
  HF_PUBLISH_TOKEN="$(<"$KINYALM_HF_PUBLISH_TOKEN_FILE")"
elif [[ "$PUBLISH_RESULTS" == "1" ]]; then
  export HF_PUBLISH_TOKEN="${HF_PUBLISH_TOKEN:-${HF_TOKEN:-}}"
fi

: "${HF_TOKEN:?Set HF_TOKEN or KINYALM_HF_TOKEN_FILE.}"
if [[ "$PUBLISH_RESULTS" == "1" ]]; then
  : "${HF_PUBLISH_TOKEN:?Set HF_PUBLISH_TOKEN or its token file.}"
fi

mkdir -p "$RUN_ROOT" "$RUNTIME_DIR"
printf 'RUNNING\n' >"$STATUS_FILE"

finish() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    printf 'COMPLETED\n' >"$STATUS_FILE"
  else
    printf 'FAILED exit_code=%s\n' "$exit_code" >"$STATUS_FILE"
  fi
  if [[ -n "${KINYALM_HF_TOKEN_FILE:-}" ]]; then
    rm -f "$KINYALM_HF_TOKEN_FILE"
  fi
  if [[ -n "${KINYALM_HF_PUBLISH_TOKEN_FILE:-}" ]]; then
    rm -f "$KINYALM_HF_PUBLISH_TOKEN_FILE"
  fi
}
trap finish EXIT

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; the bake-off requires NVIDIA GPUs." >&2
  exit 1
fi

total_gpu_memory_mib="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
    awk '{ total += $1 } END { print total + 0 }'
)"
if (( total_gpu_memory_mib < MIN_GPU_MEMORY_MIB )); then
  echo "The bake-off needs at least ${MIN_GPU_MEMORY_MIB} MiB total GPU memory; found ${total_gpu_memory_mib} MiB." >&2
  exit 1
fi

available_disk_kib="$(df -Pk "$HOME" | awk 'NR == 2 { print $4 }')"
if (( available_disk_kib < 125829120 )); then
  echo "The bake-off needs at least 120 GiB of free local disk." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
  "transformers==$TRANSFORMERS_VERSION" \
  "accelerate==$ACCELERATE_VERSION" \
  "peft==$PEFT_VERSION" \
  "huggingface-hub==$HUGGINGFACE_HUB_VERSION" \
  "safetensors==$SAFETENSORS_VERSION" \
  "pillow==$PILLOW_VERSION" \
  "sentencepiece==$SENTENCEPIECE_VERSION"

"$VENV_DIR/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "PyTorch cannot see CUDA"'

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'config=%s\n' "$CONFIG_PATH"
  printf 'publish_repo=%s\n' "$PUBLISH_REPO"
  printf 'publish_path=%s\n' "$PUBLISH_PATH"
  printf 'publish_results=%s\n' "$PUBLISH_RESULTS"
  printf 'task_limit=%s\n' "$TASK_LIMIT"
  printf 'candidate_id=%s\n' "$CANDIDATE_ID"
  uname -a
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader
  df -h "$HOME"
  "$VENV_DIR/bin/python" -m pip freeze
} >"$SYSTEM_INFO"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

runner_args=(
  --config "$CONFIG_PATH"
  --output-dir "$RUN_ROOT"
)
if [[ -n "$TASK_LIMIT" ]]; then
  runner_args+=(--limit "$TASK_LIMIT")
fi
if [[ -n "$CANDIDATE_ID" ]]; then
  runner_args+=(--candidate "$CANDIDATE_ID")
fi

set +e
"$VENV_DIR/bin/python" scripts/run_multilingual_bakeoff.py \
  "${runner_args[@]}" 2>&1 | tee "$RUN_LOG"
run_exit="${PIPESTATUS[0]}"
set -e

publish_exit=0
if [[ "$PUBLISH_RESULTS" == "1" ]] && compgen -G "$RUN_ROOT/raw/*.jsonl" >/dev/null; then
  set +e
  "$VENV_DIR/bin/python" scripts/publish_bakeoff_run.py \
    --run-dir "$RUN_ROOT" \
    --repo-id "$PUBLISH_REPO" \
    --path-in-repo "$PUBLISH_PATH" 2>&1 | tee "$PUBLISH_LOG"
  publish_exit="${PIPESTATUS[0]}"
  set -e
fi

if (( run_exit != 0 )); then
  exit "$run_exit"
fi
if (( publish_exit != 0 )); then
  exit "$publish_exit"
fi

if [[ "$PUBLISH_RESULTS" == "1" ]]; then
  echo "Bake-off completed and published."
else
  echo "Bake-off completed; publication was intentionally deferred for blind review."
fi

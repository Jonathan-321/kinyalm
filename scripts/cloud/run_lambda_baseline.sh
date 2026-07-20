#!/usr/bin/env bash
set -Eeuo pipefail

DATA_REPO="${DATA_REPO:-kinyalm/kinyalm-data-lake}"
DATA_REVISION="${DATA_REVISION:-754a58b021cfe1e505f432df0de45ce2f63a3b21}"
MODEL_PROFILE="${MODEL_PROFILE:-qwen}"
case "$MODEL_PROFILE" in
  gemma)
    PROFILE_MODEL_ID="google/gemma-2-9b-it"
    PROFILE_MODEL_REVISION="11c9b309abf73637e4b6f9a3fa1e92e615547819"
    PROFILE_OUTPUT_REPO="kinyalm/kinyalm-gemma-2-9b-track2-baseline-a"
    PROFILE_ATTN_IMPLEMENTATION="eager"
    PROFILE_RUN_SLUG="gemma2-9b-baseline-a"
    ;;
  qwen)
    PROFILE_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
    PROFILE_MODEL_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
    PROFILE_OUTPUT_REPO="kinyalm/kinyalm-qwen2.5-7b-track2-baseline-a"
    PROFILE_ATTN_IMPLEMENTATION="sdpa"
    PROFILE_RUN_SLUG="qwen25-7b-baseline-a"
    ;;
  *)
    echo "MODEL_PROFILE must be gemma or qwen" >&2
    exit 2
    ;;
esac
MODEL_ID="${MODEL_ID:-$PROFILE_MODEL_ID}"
MODEL_REVISION="${MODEL_REVISION:-$PROFILE_MODEL_REVISION}"
OUTPUT_REPO="${OUTPUT_REPO:-$PROFILE_OUTPUT_REPO}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-$PROFILE_ATTN_IMPLEMENTATION}"
MAX_STEPS="${MAX_STEPS:--1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RUN_ID="${RUN_ID:-$PROFILE_RUN_SLUG-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$HOME/kinyalm-runs/$RUN_ID}"
DATA_DIR="$RUN_ROOT/data"
ADAPTER_DIR="$RUN_ROOT/adapter"
STATUS_FILE="$RUN_ROOT/status"
SYSTEM_INFO="$RUN_ROOT/system-info.txt"
TRAIN_LOG="$RUN_ROOT/train.log"

if [[ "$PREFLIGHT_ONLY" != "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
  echo "PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "$PREFLIGHT_ONLY" == "0" ]]; then
  : "${HF_TOKEN:?Set HF_TOKEN to a token that can read the selected model.}"
  HF_PUBLISH_TOKEN="${HF_PUBLISH_TOKEN:-$HF_TOKEN}"
  export HF_PUBLISH_TOKEN
fi

mkdir -p "$RUN_ROOT"
printf 'RUNNING\n' > "$STATUS_FILE"

finish() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    printf 'COMPLETED\n' > "$STATUS_FILE"
  else
    printf 'FAILED exit_code=%s\n' "$exit_code" > "$STATUS_FILE"
  fi
  if [[ -n "${KINYALM_HF_TOKEN_FILE:-}" ]]; then
    rm -f "$KINYALM_HF_TOKEN_FILE"
  fi
  if [[ -n "${KINYALM_HF_PUBLISH_TOKEN_FILE:-}" ]]; then
    rm -f "$KINYALM_HF_PUBLISH_TOKEN_FILE"
  fi
}
trap finish EXIT

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$PREFLIGHT_ONLY" == "0" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this profile requires an NVIDIA GPU." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'model=%s@%s\n' "$MODEL_ID" "$MODEL_REVISION"
  printf 'dataset=%s@%s\n' "$DATA_REPO" "$DATA_REVISION"
  printf 'uv=%s\n' "$(uv --version)"
  uname -a
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
      --format=csv,noheader
  else
    printf 'gpu=not checked in preflight-only mode\n'
  fi
} > "$SYSTEM_INFO"

uv sync --extra train --frozen

uv run python scripts/prepare_hf_sft_baseline.py \
  --repo-id "$DATA_REPO" \
  --revision "$DATA_REVISION" \
  --mode critic-accepted \
  --output-dir "$DATA_DIR" \
  --acknowledge-experimental

training_args=(
  --model "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --train-file "$DATA_DIR/train.jsonl"
  --eval-file "$DATA_DIR/validation.jsonl"
  --dataset-manifest "$DATA_DIR/dataset-manifest.json"
  --output-dir "$ADAPTER_DIR"
  --sample-prompts-file configs/training/track2-baseline-prompts.txt
  --experimental
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --max-steps "$MAX_STEPS"
)

uv run python scripts/train_qlora.py "${training_args[@]}" --dry-run
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Lambda profile preflight complete; no model was loaded or published."
  exit 0
fi
uv run python scripts/train_qlora.py "${training_args[@]}" 2>&1 | tee "$TRAIN_LOG"

uv run python scripts/publish_training_run.py \
  --adapter-dir "$ADAPTER_DIR" \
  --dataset-manifest "$DATA_DIR/dataset-manifest.json" \
  --training-log "$TRAIN_LOG" \
  --system-info "$SYSTEM_INFO" \
  --repo-id "$OUTPUT_REPO" \
  --run-id "$RUN_ID" \
  --base-model "$MODEL_ID" \
  --base-model-revision "$MODEL_REVISION" \
  --dataset-repo "$DATA_REPO" \
  --dataset-revision "$DATA_REVISION"

echo "Run completed and published to https://huggingface.co/$OUTPUT_REPO"

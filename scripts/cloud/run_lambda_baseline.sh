#!/usr/bin/env bash
set -Eeuo pipefail

DATA_REPO="${DATA_REPO:-kinyalm/kinyalm-data-lake}"
DATA_REVISION="${DATA_REVISION:-754a58b021cfe1e505f432df0de45ce2f63a3b21}"
MODEL_PROFILE="${MODEL_PROFILE:-gemma4}"
case "$MODEL_PROFILE" in
  gemma)
    PROFILE_MODEL_ID="google/gemma-2-9b-it"
    PROFILE_MODEL_REVISION="11c9b309abf73637e4b6f9a3fa1e92e615547819"
    PROFILE_OUTPUT_REPO="kinyalm/kinyalm-gemma-2-9b-track2-baseline-a"
    PROFILE_ATTN_IMPLEMENTATION="eager"
    PROFILE_RUN_SLUG="gemma2-9b-baseline-a"
    ;;
  gemma4)
    # Gemma 4 is public under Apache-2.0. The data lake and output repository
    # still require the appropriate Hugging Face organization permissions.
    PROFILE_MODEL_ID="google/gemma-4-12B-it"
    PROFILE_MODEL_REVISION="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    PROFILE_OUTPUT_REPO="kinyalm/kinyalm-gemma-4-12b-experimental"
    PROFILE_ATTN_IMPLEMENTATION="eager"
    PROFILE_RUN_SLUG="gemma4-12b-experimental"
    ;;
  qwen)
    PROFILE_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
    PROFILE_MODEL_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
    PROFILE_OUTPUT_REPO="kinyalm/kinyalm-qwen2.5-7b-track2-baseline-a"
    PROFILE_ATTN_IMPLEMENTATION="sdpa"
    PROFILE_RUN_SLUG="qwen25-7b-baseline-a"
    ;;
  *)
    echo "MODEL_PROFILE must be gemma4, gemma, or qwen" >&2
    exit 2
    ;;
esac
MODEL_ID="${MODEL_ID:-$PROFILE_MODEL_ID}"
MODEL_REVISION="${MODEL_REVISION:-$PROFILE_MODEL_REVISION}"
OUTPUT_REPO="${OUTPUT_REPO:-$PROFILE_OUTPUT_REPO}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-$PROFILE_ATTN_IMPLEMENTATION}"
MAX_STEPS="${MAX_STEPS:--1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PROFILE_ONLY="${PROFILE_ONLY:-0}"
ALLOW_EXPERIMENTAL_FULL_RUN="${ALLOW_EXPERIMENTAL_FULL_RUN:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
EPOCHS="${EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-25}"
SAMPLE_PROMPTS_FILE="${SAMPLE_PROMPTS_FILE:-configs/training/track2-baseline-prompts.txt}"
SAMPLES_ENABLED=1
if [[ "$MAX_STEPS" == "1" ]]; then
  WARMUP_RATIO=0
  SAMPLE_PROMPTS_FILE=""
  SAMPLES_ENABLED=0
fi
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
if [[ "$PROFILE_ONLY" != "0" && "$PROFILE_ONLY" != "1" ]]; then
  echo "PROFILE_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "$ALLOW_EXPERIMENTAL_FULL_RUN" != "0" && "$ALLOW_EXPERIMENTAL_FULL_RUN" != "1" ]]; then
  echo "ALLOW_EXPERIMENTAL_FULL_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "$PROFILE_ONLY" == "1" ]]; then
  printf 'model_profile=%s\n' "$MODEL_PROFILE"
  printf 'model_id=%s\n' "$MODEL_ID"
  printf 'model_revision=%s\n' "$MODEL_REVISION"
  printf 'output_repo=%s\n' "$OUTPUT_REPO"
  printf 'attention_implementation=%s\n' "$ATTN_IMPLEMENTATION"
  printf 'max_steps=%s\n' "$MAX_STEPS"
  printf 'warmup_ratio=%s\n' "$WARMUP_RATIO"
  printf 'learning_rate=%s\n' "$LEARNING_RATE"
  printf 'epochs=%s\n' "$EPOCHS"
  printf 'save_steps=%s\n' "$SAVE_STEPS"
  printf 'eval_steps=%s\n' "$EVAL_STEPS"
  printf 'samples_enabled=%s\n' "$SAMPLES_ENABLED"
  exit 0
fi
if [[ "$PREFLIGHT_ONLY" == "0" && "$MODEL_PROFILE" == "gemma4" \
  && "$MAX_STEPS" != "1" && "$ALLOW_EXPERIMENTAL_FULL_RUN" != "1" ]]; then
  echo "Gemma 4 is limited to MAX_STEPS=1 until the smoke gate passes." >&2
  echo "After review, set ALLOW_EXPERIMENTAL_FULL_RUN=1 explicitly." >&2
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
uv run python -c \
  'import peft, torch, transformers, trl; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"trl={trl.__version__}"); print(f"peft={peft.__version__}")' \
  >> "$SYSTEM_INFO"

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
  --experimental
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --warmup-ratio "$WARMUP_RATIO"
  --learning-rate "$LEARNING_RATE"
  --epochs "$EPOCHS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --max-steps "$MAX_STEPS"
)
if [[ -n "$SAMPLE_PROMPTS_FILE" ]]; then
  training_args+=(--sample-prompts-file "$SAMPLE_PROMPTS_FILE")
fi

uv run python scripts/train_qlora.py \
  "${training_args[@]}" \
  --dry-run \
  --verify-model-metadata
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

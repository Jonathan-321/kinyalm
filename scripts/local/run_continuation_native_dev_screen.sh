#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${KINYALM_MLX_PYTHON:-$HOME/.cache/kinyalm/gemma4-12b-bakeoff/venv/bin/python}"
CONFIG_PATH="$ROOT_DIR/configs/evaluation/gemma4_recovery_bakeoff.json"
BASE_RAW="$ROOT_DIR/outputs/evaluation/gemma4-recovery-native-base-v1/raw/gemma4-12b-it-qat-4bit-mlx.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/evaluation/gemma4-continuation-native-dev-screen-v1-20260811}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "MLX evaluation Python is missing: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BASE_RAW" ]]; then
  echo "Pinned 150-prompt base output is missing: $BASE_RAW" >&2
  exit 1
fi

runtimes=(
  "source780|$HOME/.cache/kinyalm/gemma4-12b-longform-step-780/runtime.json"
  "control|$HOME/.cache/kinyalm/gemma4-12b-continuation-control-lr2e6/runtime.json"
  "targeted|$HOME/.cache/kinyalm/gemma4-12b-continuation-targeted-lr2e6/runtime.json"
)

mkdir -p "$OUTPUT_ROOT"
printf 'RUNNING\n' >"$OUTPUT_ROOT/status"
trap 'printf "FAILED\n" >"$OUTPUT_ROOT/status"' ERR

cd "$ROOT_DIR"
for spec in "${runtimes[@]}"; do
  name="${spec%%|*}"
  runtime="${spec#*|}"
  output_dir="$OUTPUT_ROOT/$name"

  if [[ ! -f "$runtime" ]]; then
    echo "Prepared adapter runtime is missing: $runtime" >&2
    exit 1
  fi

  mkdir -p "$output_dir/raw"
  cp "$BASE_RAW" "$output_dir/raw/gemma4-12b-it-qat-4bit-mlx.jsonl"

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/run_multilingual_bakeoff.py \
    --config "$CONFIG_PATH" \
    --backend mlx \
    --candidate gemma4-12b-it \
    --adapter-runtime "$runtime" \
    --output-dir "$output_dir" \
    2>&1 | tee "$output_dir/run.log"

  "$PYTHON_BIN" scripts/summarize_bakeoff_run.py \
    --run-dir "$output_dir" \
    2>&1 | tee -a "$output_dir/run.log"
done

"$PYTHON_BIN" scripts/build_combined_continuation_review.py \
  --run-root "$OUTPUT_ROOT"
"$PYTHON_BIN" scripts/summarize_bakeoff_run.py \
  --run-dir "$OUTPUT_ROOT/combined"

trap - ERR
printf 'COMPLETED\n' >"$OUTPUT_ROOT/status"

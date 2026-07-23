#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KINYALM_GEMMA4_LOCAL_DIR:-$HOME/.cache/kinyalm/gemma4-12b-bakeoff}"
VENV_DIR="$RUNTIME_DIR/venv"
PYTHON_BIN="${KINYALM_PYTHON:-python3}"
MLX_LM_VERSION="0.31.3"
RUN_ID="${RUN_ID:-gemma4-12b-mlx-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/model-bakeoffs/$RUN_ID}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The local Gemma 4 runner requires an Apple-silicon Mac." >&2
  exit 1
fi

available_kib="$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')"
if (( available_kib < 15728640 )); then
  echo "At least 15 GiB of free disk is required for the model and runtime." >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
  "mlx-lm==$MLX_LM_VERSION"

cd "$ROOT_DIR"
"$VENV_DIR/bin/python" scripts/run_multilingual_bakeoff.py \
  --backend mlx \
  --candidate gemma4-12b-it \
  --output-dir "$OUTPUT_DIR" \
  "$@"

"$VENV_DIR/bin/python" scripts/summarize_bakeoff_run.py \
  --run-dir "$OUTPUT_DIR"

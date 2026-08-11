#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUNTIME_DIR="${KINYALM_GEMMA4_LOCAL_DIR:-$HOME/.cache/kinyalm/gemma4-12b-bakeoff}"
ADAPTER_RUNTIME_DIR="${KINYALM_GEMMA4_STEP500_DIR:-$HOME/.cache/kinyalm/gemma4-12b-longform-step-500}"
VENV_DIR="$BASE_RUNTIME_DIR/venv"
PYTHON_BIN="${KINYALM_PYTHON:-python3}"
MLX_LM_VERSION="0.31.3"
BASE_REPO="mlx-community/gemma-4-12B-it-qat-4bit"
BASE_REVISION="e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6"
ADAPTER_REPO="kinyalm/kinyalm-gemma-4-12b-longform-fullepoch-qv-r8-lr2e5"
ADAPTER_REVISION="44f584225fb3cc215a52be69fcb107da2e743643"
ADAPTER_SUBFOLDER="checkpoints/checkpoint-500"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The local Gemma 4 comparison requires an Apple-silicon Mac." >&2
  exit 1
fi

mkdir -p "$BASE_RUNTIME_DIR" "$ADAPTER_RUNTIME_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
  "mlx-lm==$MLX_LM_VERSION" "huggingface-hub>=1,<2" "safetensors>=0.5,<1"

cd "$ROOT_DIR"
"$VENV_DIR/bin/python" scripts/prepare_local_mlx.py \
  --runtime-dir "$ADAPTER_RUNTIME_DIR" \
  --base-repo "$BASE_REPO" \
  --base-revision "$BASE_REVISION" \
  --adapter-repo "$ADAPTER_REPO" \
  --adapter-revision "$ADAPTER_REVISION" \
  --adapter-subfolder "$ADAPTER_SUBFOLDER"

exec "$VENV_DIR/bin/python" scripts/local/serve_gemma4_chat.py \
  --adapter-path "$ADAPTER_RUNTIME_DIR/adapter-mlx" \
  --adapter-repo "$ADAPTER_REPO/$ADAPTER_SUBFOLDER" \
  --adapter-revision "$ADAPTER_REVISION" \
  "$@"

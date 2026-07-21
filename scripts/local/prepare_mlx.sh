#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KINYALM_LOCAL_DIR:-$HOME/.cache/kinyalm/track2-baseline-a}"
VENV_DIR="$RUNTIME_DIR/venv"
PYTHON_BIN="${KINYALM_PYTHON:-python3}"
MLX_LM_VERSION="${MLX_LM_VERSION:-0.31.3}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Local MLX serving requires an Apple-silicon Mac." >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
  "mlx-lm==$MLX_LM_VERSION" \
  "huggingface-hub>=1,<2" \
  "safetensors>=0.5,<1"

"$VENV_DIR/bin/python" "$ROOT_DIR/scripts/prepare_local_mlx.py" \
  --runtime-dir "$RUNTIME_DIR" >/dev/null

install -m 0644 \
  "$ROOT_DIR/scripts/local/mlx_server.py" \
  "$RUNTIME_DIR/mlx_server.py"

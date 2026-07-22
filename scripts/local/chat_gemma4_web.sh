#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KINYALM_GEMMA4_LOCAL_DIR:-$HOME/.cache/kinyalm/gemma4-12b-bakeoff}"
VENV_DIR="$RUNTIME_DIR/venv"
PYTHON_BIN="${KINYALM_PYTHON:-python3}"
MLX_LM_VERSION="0.31.3"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "The local Gemma 4 chat requires an Apple-silicon Mac." >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
  "mlx-lm==$MLX_LM_VERSION"

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/python" scripts/local/serve_gemma4_chat.py "$@"

#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${KINYALM_KAKUGO_DIR:-$HOME/.cache/kinyalm/kakugo-3b-kin}"
VENV_DIR="$RUNTIME_DIR/venv"
PYTHON_BIN="${KINYALM_PYTHON:-python3}"
MLX_LM_VERSION="${MLX_LM_VERSION:-0.31.3}"
VERSION_FILE="$RUNTIME_DIR/mlx-lm-version"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Kakugo MLX inference requires an Apple-silicon Mac." >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

installed_version=""
if [[ -f "$VERSION_FILE" ]]; then
  installed_version="$(<"$VERSION_FILE")"
fi

if [[ "$installed_version" != "$MLX_LM_VERSION" ]]; then
  echo "Preparing the isolated Kakugo MLX runtime..."
  "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
    "mlx-lm==$MLX_LM_VERSION"
  printf '%s\n' "$MLX_LM_VERSION" >"$VERSION_FILE"
fi

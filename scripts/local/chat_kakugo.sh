#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KINYALM_KAKUGO_DIR:-$HOME/.cache/kinyalm/kakugo-3b-kin}"
VENV_DIR="$RUNTIME_DIR/venv"

bash "$ROOT_DIR/scripts/local/prepare_kakugo.sh"

cd "$ROOT_DIR"
exec "$VENV_DIR/bin/python" scripts/local/chat_kakugo.py "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KINYALM_LOCAL_DIR:-$HOME/.cache/kinyalm/track2-baseline-a}"
VENV_DIR="$RUNTIME_DIR/venv"
HOST="${KINYALM_HOST:-127.0.0.1}"
PORT="${KINYALM_PORT:-8080}"

bash "$ROOT_DIR/scripts/local/prepare_mlx.sh"

BASE_PATH="$("$VENV_DIR/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' \
  "$RUNTIME_DIR/runtime.json")"
ADAPTER_PATH="$("$VENV_DIR/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["adapter"]["path"])' \
  "$RUNTIME_DIR/runtime.json")"

echo "KinyaLM local endpoint: http://$HOST:$PORT/v1/chat/completions"
echo "Press Ctrl-C to stop the server."

exec "$VENV_DIR/bin/python" "$RUNTIME_DIR/mlx_server.py" \
  --model "$BASE_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level INFO

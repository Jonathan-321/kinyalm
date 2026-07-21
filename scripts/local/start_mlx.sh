#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KINYALM_LOCAL_DIR:-$HOME/.cache/kinyalm/track2-baseline-a}"
LOG_FILE="$RUNTIME_DIR/server.log"
HOST="${KINYALM_HOST:-127.0.0.1}"
PORT="${KINYALM_PORT:-8080}"
START_TIMEOUT="${KINYALM_START_TIMEOUT:-300}"
HEALTH_URL="http://$HOST:$PORT/v1/models"
LABEL="ai.kinyalm.local-server"
DOMAIN="gui/$(id -u)"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$LABEL.plist"
PYTHON_BIN="${KINYALM_PYTHON:-python3}"
VENV_PYTHON="$RUNTIME_DIR/venv/bin/python"
SERVER_PATH="$RUNTIME_DIR/mlx_server.py"

mkdir -p "$RUNTIME_DIR" "$PLIST_DIR"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null; then
    echo "KinyaLM is already running."
    echo "Endpoint: http://$HOST:$PORT/v1/chat/completions"
    exit 0
  fi
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
fi

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use by another process." >&2
  exit 1
fi

KINYALM_LOCAL_DIR="$RUNTIME_DIR" \
  KINYALM_PYTHON="$PYTHON_BIN" \
  bash "$ROOT_DIR/scripts/local/prepare_mlx.sh"
BASE_PATH="$("$VENV_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' \
  "$RUNTIME_DIR/runtime.json")"
ADAPTER_PATH="$("$VENV_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["adapter"]["path"])' \
  "$RUNTIME_DIR/runtime.json")"

"$PYTHON_BIN" - \
  "$PLIST_PATH" "$LABEL" "$RUNTIME_DIR" "$HOST" "$PORT" "$LOG_FILE" \
  "$VENV_PYTHON" "$SERVER_PATH" "$BASE_PATH" "$ADAPTER_PATH" <<'PY'
import os
import plistlib
import sys

(
    plist_path,
    label,
    runtime,
    host,
    port,
    log_path,
    python_path,
    server_path,
    base_path,
    adapter_path,
) = sys.argv[1:]
payload = {
    "Label": label,
    "ProgramArguments": [
        python_path,
        server_path,
        "--model",
        base_path,
        "--adapter-path",
        adapter_path,
        "--host",
        host,
        "--port",
        port,
        "--log-level",
        "INFO",
    ],
    "WorkingDirectory": runtime,
    "EnvironmentVariables": {
        "HOME": os.path.expanduser("~"),
        "KINYALM_HOST": host,
        "KINYALM_LOCAL_DIR": runtime,
        "KINYALM_PORT": port,
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    },
    "RunAtLoad": True,
    "KeepAlive": True,
    "ProcessType": "Background",
    "ThrottleInterval": 10,
    "StandardOutPath": log_path,
    "StandardErrorPath": log_path,
}
with open(plist_path, "wb") as handle:
    plistlib.dump(payload, handle, sort_keys=True)
PY
chmod 644 "$PLIST_PATH"

: >"$LOG_FILE"
rm -f "$RUNTIME_DIR/server.pid"
launchctl bootstrap "$DOMAIN" "$PLIST_PATH"

for ((elapsed = 0; elapsed < START_TIMEOUT; elapsed++)); do
  if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null; then
    echo "KinyaLM is running as macOS service $LABEL."
    echo "Endpoint: http://$HOST:$PORT/v1/chat/completions"
    echo "Log: $LOG_FILE"
    exit 0
  fi
  if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "KinyaLM service stopped during startup. Last log lines:" >&2
    tail -n 30 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done

echo "KinyaLM did not become ready within ${START_TIMEOUT}s." >&2
echo "Last log lines:" >&2
tail -n 30 "$LOG_FILE" >&2
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
exit 1

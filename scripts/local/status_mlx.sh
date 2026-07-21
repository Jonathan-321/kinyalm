#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${KINYALM_LOCAL_DIR:-$HOME/.cache/kinyalm/track2-baseline-a}"
HOST="${KINYALM_HOST:-127.0.0.1}"
PORT="${KINYALM_PORT:-8080}"
HEALTH_URL="http://$HOST:$PORT/v1/models"
LABEL="ai.kinyalm.local-server"
DOMAIN="gui/$(id -u)"

if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  echo "KinyaLM is stopped."
  exit 1
fi

if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null; then
  SERVER_PID="$(launchctl print "$DOMAIN/$LABEL" \
    | awk '/pid =/{print $3; exit}')"
  echo "KinyaLM is running (PID ${SERVER_PID:-unknown})."
  echo "Model: Qwen2.5-7B-Instruct base + KinyaLM Track 2 LoRA."
  echo "Endpoint: http://$HOST:$PORT/v1/chat/completions"
  exit 0
fi

echo "KinyaLM is not healthy. Inspect $RUNTIME_DIR/server.log."
exit 1

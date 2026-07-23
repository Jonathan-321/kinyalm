#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${KINYALM_LOCAL_DIR:-$HOME/.cache/kinyalm/track2-baseline-a}"
LABEL="ai.kinyalm.local-server"
DOMAIN="gui/$(id -u)"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"

if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  rm -f "$PLIST_PATH" "$RUNTIME_DIR/server.pid"
  echo "KinyaLM is already stopped."
  exit 0
fi

launchctl bootout "$DOMAIN/$LABEL"
rm -f "$PLIST_PATH" "$RUNTIME_DIR/server.pid"
echo "KinyaLM stopped."

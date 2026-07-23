#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <instance-ip-or-hostname> <run-id> <local-destination>" >&2
  exit 2
fi

HOST="$1"
RUN_ID="$2"
DESTINATION="$3"
SSH_KEY="${LAMBDA_SSH_KEY:-$HOME/.ssh/coolify_key}"

mkdir -p "$DESTINATION"
scp -i "$SSH_KEY" -r \
  "ubuntu@$HOST:kinyalm-bakeoffs/$RUN_ID" \
  "$DESTINATION/"

echo "Fetched $RUN_ID to $DESTINATION."

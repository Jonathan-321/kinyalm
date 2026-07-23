#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <instance-ip-or-hostname> [git-ref]" >&2
  exit 2
fi

HOST="$1"
REPO_REF="${2:-main}"
REMOTE_MAX_STEPS="${MAX_STEPS:--1}"
REMOTE_MODEL_PROFILE="${MODEL_PROFILE:-qwen}"
SSH_KEY="${LAMBDA_SSH_KEY:-$HOME/.ssh/coolify_key}"
HF_MODEL_TOKEN_NAME="${HF_MODEL_TOKEN_NAME:-}"
HF_PUBLISH_TOKEN_NAME="${HF_PUBLISH_TOKEN_NAME:-}"
REMOTE_MODEL_TOKEN_FILE=".config/kinyalm/hf-model-token"
REMOTE_PUBLISH_TOKEN_FILE=".config/kinyalm/hf-publish-token"
REMOTE_LOG="kinyalm-bootstrap.log"

if [[ ! "$REPO_REF" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "invalid git ref: $REPO_REF" >&2
  exit 2
fi
if [[ ! "$REMOTE_MAX_STEPS" =~ ^-?[0-9]+$ ]]; then
  echo "MAX_STEPS must be an integer" >&2
  exit 2
fi
if [[ "$REMOTE_MODEL_PROFILE" != "gemma" && "$REMOTE_MODEL_PROFILE" != "qwen" ]]; then
  echo "MODEL_PROFILE must be gemma or qwen" >&2
  exit 2
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "Lambda SSH private key not found: $SSH_KEY" >&2
  exit 1
fi

read_hf_token() {
  local token_name="$1"
  HF_REQUESTED_TOKEN_NAME="$token_name" uv run python -c '
import os
from huggingface_hub import get_token
from huggingface_hub.utils import get_stored_tokens
name = os.environ["HF_REQUESTED_TOKEN_NAME"]
token = get_stored_tokens().get(name) if name else get_token()
assert token, f"No cached HF token named {name!r}"
print(token, end="")
'
}

HF_MODEL_TOKEN_VALUE="$(read_hf_token "$HF_MODEL_TOKEN_NAME")"
printf '%s' "$HF_MODEL_TOKEN_VALUE" | ssh -i "$SSH_KEY" \
  -o StrictHostKeyChecking=accept-new \
  "ubuntu@$HOST" \
  "umask 077; mkdir -p .config/kinyalm; cat > '$REMOTE_MODEL_TOKEN_FILE'"
unset HF_MODEL_TOKEN_VALUE

HF_PUBLISH_TOKEN_VALUE="$(read_hf_token "$HF_PUBLISH_TOKEN_NAME")"
printf '%s' "$HF_PUBLISH_TOKEN_VALUE" | ssh -i "$SSH_KEY" "ubuntu@$HOST" \
  "umask 077; cat > '$REMOTE_PUBLISH_TOKEN_FILE'"
unset HF_PUBLISH_TOKEN_VALUE

ssh -i "$SSH_KEY" "ubuntu@$HOST" \
  "KINYALM_REPO_REF='$REPO_REF' MAX_STEPS='$REMOTE_MAX_STEPS' MODEL_PROFILE='$REMOTE_MODEL_PROFILE' bash -se" <<'REMOTE_SCRIPT'
if [[ ! -d "$HOME/kinyalm/.git" ]]; then
  git clone --filter=blob:none https://github.com/Jonathan-321/kinyalm.git \
    "$HOME/kinyalm"
fi
git -C "$HOME/kinyalm" fetch origin "$KINYALM_REPO_REF"
git -C "$HOME/kinyalm" checkout --detach FETCH_HEAD
nohup env \
  KINYALM_REPO_REF="$KINYALM_REPO_REF" \
  KINYALM_HF_TOKEN_FILE="$HOME/.config/kinyalm/hf-model-token" \
  KINYALM_HF_PUBLISH_TOKEN_FILE="$HOME/.config/kinyalm/hf-publish-token" \
  MAX_STEPS="$MAX_STEPS" \
  MODEL_PROFILE="$MODEL_PROFILE" \
  bash "$HOME/kinyalm/scripts/cloud/bootstrap_lambda_instance.sh" \
  > "$HOME/kinyalm-bootstrap.log" 2>&1 < /dev/null &
echo "$!"
REMOTE_SCRIPT

echo "Submitted. Follow progress with:"
echo "ssh -i $SSH_KEY ubuntu@$HOST 'tail -f ~/$REMOTE_LOG'"
echo "Terminate the Lambda instance after COMPLETED or FAILED; OS shutdown does not stop billing."

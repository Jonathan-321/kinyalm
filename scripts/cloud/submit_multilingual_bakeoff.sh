#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <instance-ip-or-hostname> [git-ref]" >&2
  exit 2
fi

HOST="$1"
REPO_REF="${2:-main}"
SSH_KEY="${LAMBDA_SSH_KEY:-$HOME/.ssh/coolify_key}"
HF_MODEL_TOKEN_NAME="${HF_MODEL_TOKEN_NAME:-}"
HF_PUBLISH_TOKEN_NAME="${HF_PUBLISH_TOKEN_NAME:-}"
REMOTE_MODEL_TOKEN_FILE=".config/kinyalm/hf-bakeoff-model-token"
REMOTE_PUBLISH_TOKEN_FILE=".config/kinyalm/hf-bakeoff-publish-token"

if [[ ! "$REPO_REF" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "invalid git ref: $REPO_REF" >&2
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

model_token="$(read_hf_token "$HF_MODEL_TOKEN_NAME")"
printf '%s' "$model_token" | ssh -i "$SSH_KEY" \
  -o StrictHostKeyChecking=accept-new "ubuntu@$HOST" \
  "umask 077; mkdir -p .config/kinyalm; cat > '$REMOTE_MODEL_TOKEN_FILE'"
unset model_token

publish_token="$(read_hf_token "$HF_PUBLISH_TOKEN_NAME")"
printf '%s' "$publish_token" | ssh -i "$SSH_KEY" "ubuntu@$HOST" \
  "umask 077; cat > '$REMOTE_PUBLISH_TOKEN_FILE'"
unset publish_token

ssh -i "$SSH_KEY" "ubuntu@$HOST" \
  "KINYALM_REPO_REF='$REPO_REF' bash -se" <<'REMOTE_SCRIPT'
if [[ ! -d "$HOME/kinyalm/.git" ]]; then
  git clone --filter=blob:none https://github.com/Jonathan-321/kinyalm.git \
    "$HOME/kinyalm"
fi
git -C "$HOME/kinyalm" fetch origin "$KINYALM_REPO_REF"
git -C "$HOME/kinyalm" checkout --detach FETCH_HEAD
nohup env \
  KINYALM_REPO_REF="$KINYALM_REPO_REF" \
  KINYALM_HF_TOKEN_FILE="$HOME/.config/kinyalm/hf-bakeoff-model-token" \
  KINYALM_HF_PUBLISH_TOKEN_FILE="$HOME/.config/kinyalm/hf-bakeoff-publish-token" \
  bash "$HOME/kinyalm/scripts/cloud/bootstrap_multilingual_bakeoff.sh" \
  >"$HOME/kinyalm-bakeoff-bootstrap.log" 2>&1 </dev/null &
echo "$!"
REMOTE_SCRIPT

echo "Submitted. Follow progress with:"
echo "ssh -i $SSH_KEY ubuntu@$HOST 'tail -f ~/kinyalm-bakeoff-bootstrap.log'"
echo "Terminate the instance after COMPLETED or FAILED; OS shutdown does not stop billing."

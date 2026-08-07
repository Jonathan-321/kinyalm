#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <instance-ip-or-hostname> [git-ref]" >&2
  exit 2
fi

HOST="$1"
REPO_REF="${2:-main}"
REMOTE_MAX_STEPS="${MAX_STEPS:--1}"
REMOTE_MODEL_PROFILE="${MODEL_PROFILE:-gemma4}"
REMOTE_DATA_PROFILE="${DATA_PROFILE:-legacy-critic-1k}"
REMOTE_ALLOW_EXPERIMENTAL_FULL_RUN="${ALLOW_EXPERIMENTAL_FULL_RUN:-0}"
REMOTE_CANDIDATE_QUALITY_POLICY="${CANDIDATE_QUALITY_POLICY:-unflagged}"
REMOTE_LEARNING_RATE="${LEARNING_RATE:-5e-5}"
REMOTE_WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
REMOTE_EPOCHS="${EPOCHS:-1}"
REMOTE_SAVE_STEPS="${SAVE_STEPS:-}"
REMOTE_EVAL_STEPS="${EVAL_STEPS:-}"
REMOTE_OUTPUT_REPO="${OUTPUT_REPO:-}"
REMOTE_RUN_ID="${RUN_ID:-}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"
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
if [[ "$REMOTE_MODEL_PROFILE" != "gemma4" && "$REMOTE_MODEL_PROFILE" != "gemma" && "$REMOTE_MODEL_PROFILE" != "qwen" ]]; then
  echo "MODEL_PROFILE must be gemma4, gemma, or qwen" >&2
  exit 2
fi
if [[ "$REMOTE_DATA_PROFILE" != "legacy-critic-1k" \
  && "$REMOTE_DATA_PROFILE" != "sft10k-v4" \
  && "$REMOTE_DATA_PROFILE" != "human-reviewed-432" ]]; then
  echo "DATA_PROFILE must be legacy-critic-1k, sft10k-v4, or human-reviewed-432" >&2
  exit 2
fi
if [[ "$REMOTE_ALLOW_EXPERIMENTAL_FULL_RUN" != "0" && "$REMOTE_ALLOW_EXPERIMENTAL_FULL_RUN" != "1" ]]; then
  echo "ALLOW_EXPERIMENTAL_FULL_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "$REMOTE_CANDIDATE_QUALITY_POLICY" != "unflagged" \
  && "$REMOTE_CANDIDATE_QUALITY_POLICY" != "strict-script-clean" \
  && "$REMOTE_CANDIDATE_QUALITY_POLICY" != "core-direct" ]]; then
  echo "CANDIDATE_QUALITY_POLICY must be unflagged, strict-script-clean, or core-direct" >&2
  exit 2
fi
if [[ ! "$REMOTE_LEARNING_RATE" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "LEARNING_RATE must be a non-negative number" >&2
  exit 2
fi
if [[ ! "$REMOTE_WARMUP_RATIO" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "WARMUP_RATIO must be a non-negative number" >&2
  exit 2
fi
if [[ ! "$REMOTE_EPOCHS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "EPOCHS must be a non-negative number" >&2
  exit 2
fi
for step_value in "$REMOTE_SAVE_STEPS" "$REMOTE_EVAL_STEPS"; do
  if [[ -n "$step_value" && ! "$step_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAVE_STEPS and EVAL_STEPS must be positive integers when set" >&2
    exit 2
  fi
done
if [[ -n "$REMOTE_OUTPUT_REPO" \
  && ! "$REMOTE_OUTPUT_REPO" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_REPO must be a Hugging Face namespace/repository ID" >&2
  exit 2
fi
if [[ -n "$REMOTE_RUN_ID" && ! "$REMOTE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters" >&2
  exit 2
fi
if [[ "$SUBMIT_DRY_RUN" != "0" && "$SUBMIT_DRY_RUN" != "1" ]]; then
  echo "SUBMIT_DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "$REMOTE_MODEL_PROFILE" == "gemma4" && "$REMOTE_MAX_STEPS" != "1" \
  && "$REMOTE_ALLOW_EXPERIMENTAL_FULL_RUN" != "1" ]]; then
  echo "Gemma 4 is limited to MAX_STEPS=1 until the smoke gate passes." >&2
  echo "After review, set ALLOW_EXPERIMENTAL_FULL_RUN=1 explicitly." >&2
  exit 2
fi

if [[ "$SUBMIT_DRY_RUN" == "1" ]]; then
  printf 'host=%s\n' "$HOST"
  printf 'git_ref=%s\n' "$REPO_REF"
  printf 'model_profile=%s\n' "$REMOTE_MODEL_PROFILE"
  printf 'data_profile=%s\n' "$REMOTE_DATA_PROFILE"
  printf 'max_steps=%s\n' "$REMOTE_MAX_STEPS"
  printf 'candidate_quality_policy=%s\n' "$REMOTE_CANDIDATE_QUALITY_POLICY"
  printf 'learning_rate=%s\n' "$REMOTE_LEARNING_RATE"
  printf 'warmup_ratio=%s\n' "$REMOTE_WARMUP_RATIO"
  printf 'epochs=%s\n' "$REMOTE_EPOCHS"
  printf 'save_steps=%s\n' "$REMOTE_SAVE_STEPS"
  printf 'eval_steps=%s\n' "$REMOTE_EVAL_STEPS"
  printf 'output_repo=%s\n' "$REMOTE_OUTPUT_REPO"
  printf 'run_id=%s\n' "$REMOTE_RUN_ID"
  exit 0
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
  "KINYALM_REPO_REF='$REPO_REF' MAX_STEPS='$REMOTE_MAX_STEPS' MODEL_PROFILE='$REMOTE_MODEL_PROFILE' DATA_PROFILE='$REMOTE_DATA_PROFILE' ALLOW_EXPERIMENTAL_FULL_RUN='$REMOTE_ALLOW_EXPERIMENTAL_FULL_RUN' CANDIDATE_QUALITY_POLICY='$REMOTE_CANDIDATE_QUALITY_POLICY' LEARNING_RATE='$REMOTE_LEARNING_RATE' WARMUP_RATIO='$REMOTE_WARMUP_RATIO' EPOCHS='$REMOTE_EPOCHS' SAVE_STEPS='$REMOTE_SAVE_STEPS' EVAL_STEPS='$REMOTE_EVAL_STEPS' OUTPUT_REPO='$REMOTE_OUTPUT_REPO' RUN_ID='$REMOTE_RUN_ID' bash -se" <<'REMOTE_SCRIPT'
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
  DATA_PROFILE="$DATA_PROFILE" \
  ALLOW_EXPERIMENTAL_FULL_RUN="$ALLOW_EXPERIMENTAL_FULL_RUN" \
  CANDIDATE_QUALITY_POLICY="$CANDIDATE_QUALITY_POLICY" \
  LEARNING_RATE="$LEARNING_RATE" \
  WARMUP_RATIO="$WARMUP_RATIO" \
  EPOCHS="$EPOCHS" \
  SAVE_STEPS="$SAVE_STEPS" \
  EVAL_STEPS="$EVAL_STEPS" \
  OUTPUT_REPO="$OUTPUT_REPO" \
  RUN_ID="$RUN_ID" \
  bash "$HOME/kinyalm/scripts/cloud/bootstrap_lambda_instance.sh" \
  > "$HOME/kinyalm-bootstrap.log" 2>&1 < /dev/null &
echo "$!"
REMOTE_SCRIPT

echo "Submitted. Follow progress with:"
echo "ssh -i $SSH_KEY ubuntu@$HOST 'tail -f ~/$REMOTE_LOG'"
echo "Terminate the Lambda instance after COMPLETED or FAILED; OS shutdown does not stop billing."

#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${KINYALM_REPO_URL:-https://github.com/Jonathan-321/kinyalm.git}"
REPO_REF="${KINYALM_REPO_REF:-main}"
CHECKOUT_DIR="${KINYALM_CHECKOUT_DIR:-$HOME/kinyalm}"

if [[ -n "${KINYALM_HF_TOKEN_FILE:-}" ]]; then
  if [[ ! -f "$KINYALM_HF_TOKEN_FILE" ]]; then
    echo "Token file does not exist: $KINYALM_HF_TOKEN_FILE" >&2
    exit 1
  fi
  export HF_TOKEN
  HF_TOKEN="$(<"$KINYALM_HF_TOKEN_FILE")"
fi

: "${HF_TOKEN:?Set HF_TOKEN or KINYALM_HF_TOKEN_FILE before bootstrapping.}"

if [[ -n "${KINYALM_HF_PUBLISH_TOKEN_FILE:-}" ]]; then
  if [[ ! -f "$KINYALM_HF_PUBLISH_TOKEN_FILE" ]]; then
    echo "Publish token file does not exist: $KINYALM_HF_PUBLISH_TOKEN_FILE" >&2
    exit 1
  fi
  export HF_PUBLISH_TOKEN
  HF_PUBLISH_TOKEN="$(<"$KINYALM_HF_PUBLISH_TOKEN_FILE")"
else
  export HF_PUBLISH_TOKEN="${HF_PUBLISH_TOKEN:-$HF_TOKEN}"
fi

cleanup_tokens() {
  if [[ -n "${KINYALM_HF_TOKEN_FILE:-}" ]]; then
    rm -f "$KINYALM_HF_TOKEN_FILE"
  fi
  if [[ -n "${KINYALM_HF_PUBLISH_TOKEN_FILE:-}" ]]; then
    rm -f "$KINYALM_HF_PUBLISH_TOKEN_FILE"
  fi
}
trap cleanup_tokens EXIT

if [[ ! -d "$CHECKOUT_DIR/.git" ]]; then
  git clone --filter=blob:none "$REPO_URL" "$CHECKOUT_DIR"
fi

git -C "$CHECKOUT_DIR" fetch origin "$REPO_REF"
git -C "$CHECKOUT_DIR" checkout --detach FETCH_HEAD
cd "$CHECKOUT_DIR"

bash scripts/cloud/run_lambda_baseline.sh

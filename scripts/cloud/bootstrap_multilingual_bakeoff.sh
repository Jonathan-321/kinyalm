#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${KINYALM_REPO_URL:-https://github.com/Jonathan-321/kinyalm.git}"
REPO_REF="${KINYALM_REPO_REF:-main}"
CHECKOUT_DIR="${KINYALM_CHECKOUT_DIR:-$HOME/kinyalm}"

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

bash scripts/cloud/run_multilingual_bakeoff.sh

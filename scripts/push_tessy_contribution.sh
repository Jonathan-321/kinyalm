#!/usr/bin/env bash
# Package Tessy's data-lake contribution and upload it to Hugging Face.
#
# Mirrors Bonheur's layout: incoming/<name>/<clearly-named-subfolders> with a
# descriptive commit message. Shows BOTH the raw review and the training-ready
# conversion, plus a README that explains the work.
#
# Run once from the repo root, AFTER logging in:
#     source .venv/bin/activate
#     hf auth login            # or: huggingface-cli login  (WRITE token)
#     bash scripts/push_tessy_contribution.sh

set -euo pipefail

# Find the Hugging Face CLI: newer installs ship "hf", older ship "huggingface-cli".
if command -v hf >/dev/null 2>&1; then
  HF="hf"; HF_WHOAMI="hf auth whoami"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli"; HF_WHOAMI="huggingface-cli whoami"
else
  echo "ERROR: no Hugging Face CLI found."
  echo "Activate your venv:  source .venv/bin/activate"
  echo "Then install:        pip install -U huggingface_hub"
  exit 1
fi

REPO="kinyalm/kinyalm-data-lake"
CONTRIBUTOR="tessymugisha"
STAGE="$HOME/KinyaLMData/hf_contributions/incoming/${CONTRIBUTOR}"
REMOTE_PATH="incoming/${CONTRIBUTOR}"
COMMIT_MSG="Add Tessy review of distillation queue (326 conversations, all Keep) + SFT-ready conversion (561 rows: 512 train / 49 validation)"

echo "==> Confirming Hugging Face login..."
$HF_WHOAMI

echo "==> Assembling contribution folder..."
rm -rf "$STAGE"
mkdir -p "$STAGE/distillation-review" "$STAGE/sft-ready"
cp data/reviewed/tessy_distillation_queue.jsonl        "$STAGE/distillation-review/"
cp data/sft/tessy-distill-review.train.jsonl           "$STAGE/sft-ready/train.jsonl"
cp data/sft/tessy-distill-review.validation.jsonl      "$STAGE/sft-ready/validation.jsonl"

cat > "$STAGE/README.md" <<'EOF'
# Tessy Mugisha — Distillation Queue Contribution

Reviewer: Tessy Mugisha (HF: tessy17)

## distillation-review/
`tessy_distillation_queue.jsonl` — my 326 assigned distillation-queue
conversations, fluent-speaker reviewed. All flagged **Keep**: I judged the
original responses natural and declined the critic's suggested revisions.
Each row keeps `critic_feedback` and `suggested_revision` as an audit trail.

## sft-ready/
`train.jsonl` (512) + `validation.jsonl` (49) — the same reviewed data
converted into the project's SFT schema and split for training. Multi-turn
conversations are split into (user, assistant) pairs; every row passes
`scripts/validate_sft_jsonl.py`. Reproducible via
`scripts/convert_distillation_review_to_sft.py` in the GitHub repo.
EOF

echo "==> Uploading to ${REPO} under ${REMOTE_PATH} ..."
$HF upload --repo-type dataset "$REPO" "$STAGE" "$REMOTE_PATH" \
  --commit-message "$COMMIT_MSG"

echo ""
echo "Done. View it at:"
echo "  https://huggingface.co/datasets/${REPO}/tree/main/${REMOTE_PATH}"

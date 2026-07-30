#!/usr/bin/env bash
# Package Tessy's audited data-lake contribution and upload it to Hugging Face.
#
# Run once from the repo root, AFTER logging in:
#     source .venv/bin/activate
#     hf auth login            # or: huggingface-cli login  (WRITE token)
#     bash scripts/push_tessy_contribution.sh
#
# Set DRY_RUN=1 to assemble and inspect the upload without contacting the Hub.

set -euo pipefail

REPO="kinyalm/kinyalm-data-lake"
CONTRIBUTOR="tessymugisha"
STAGE="$HOME/KinyaLMData/hf_contributions/incoming/${CONTRIBUTOR}"
REMOTE_PATH="incoming/${CONTRIBUTOR}"
REVIEW_FILE="data/reviewed/tessy_distillation_queue.jsonl"
TRAIN_FILE="data/sft/tessy-distill-review.train.jsonl"
VALIDATION_FILE="data/sft/tessy-distill-review.validation.jsonl"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi

for source_file in "$REVIEW_FILE" "$TRAIN_FILE" "$VALIDATION_FILE"; do
  if [[ ! -f "$source_file" ]]; then
    echo "ERROR: missing source file: $source_file" >&2
    exit 1
  fi
done

REVIEW_COUNT="$(awk 'END { print NR }' "$REVIEW_FILE")"
TRAIN_COUNT="$(awk 'END { print NR }' "$TRAIN_FILE")"
VALIDATION_COUNT="$(awk 'END { print NR }' "$VALIDATION_FILE")"
SFT_COUNT=$((TRAIN_COUNT + VALIDATION_COUNT))
WITHHELD_COUNT=$((REVIEW_COUNT - SFT_COUNT))
COMMIT_MSG="Publish Tessy's critic-agreed SFT set (${SFT_COUNT} conversations: ${TRAIN_COUNT} train / ${VALIDATION_COUNT} validation; ${WITHHELD_COUNT} withheld)"

echo "==> Assembling contribution folder..."
rm -rf "$STAGE"
mkdir -p \
  "$STAGE/distillation-review" \
  "$STAGE/sft-ready-critic-agreed-v1"
cp "$REVIEW_FILE" "$STAGE/distillation-review/"
cp "$TRAIN_FILE" "$STAGE/sft-ready-critic-agreed-v1/train.jsonl"
cp "$VALIDATION_FILE" "$STAGE/sft-ready-critic-agreed-v1/validation.jsonl"

cat > "$STAGE/README.md" <<EOF
# Tessy Mugisha - Distillation Queue Contribution

Reviewer: Tessy Mugisha (HF: tessy17)

## distillation-review/
\`tessy_distillation_queue.jsonl\` contains ${REVIEW_COUNT} fluent-speaker
reviewed conversations. All were marked **Keep** by the reviewer. The file
retains \`critic_feedback\` and \`suggested_revision\` for audit.

## sft-ready-critic-agreed-v1/
\`train.jsonl\` (${TRAIN_COUNT}) and \`validation.jsonl\`
(${VALIDATION_COUNT}) contain ${SFT_COUNT} complete conversations where the
human review and critic both accepted the original response. The remaining
${WITHHELD_COUNT} critic-disputed conversations are not in this training set
and require explicit adjudication.

This version is reproducible from the KinyaLM Git repository with
\`scripts/convert_distillation_review_to_sft.py\`. Multi-turn conversations
remain intact and are assigned to one split to prevent turn leakage.

## Legacy sft-ready/ folder
The older \`sft-ready/\` folder contains a ${REVIEW_COUNT}-conversation export
split into 561 user/assistant pairs. It includes critic-disputed conversations
and is not reproducible with the current audited converter. Keep it only as
historical input; do not use it for the final training run.
EOF

if [[ "$DRY_RUN" == "1" ]]; then
  echo "==> Dry run complete; staged ${STAGE}"
  exit 0
fi

# Find the Hugging Face CLI: newer installs ship "hf", older ship
# "huggingface-cli".
if command -v hf >/dev/null 2>&1; then
  HF=(hf)
  HF_WHOAMI=(hf auth whoami)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli)
  HF_WHOAMI=(huggingface-cli whoami)
else
  echo "ERROR: no Hugging Face CLI found."
  echo "Activate your venv:  source .venv/bin/activate"
  echo "Then install:        pip install -U huggingface_hub"
  exit 1
fi

echo "==> Confirming Hugging Face login..."
"${HF_WHOAMI[@]}"

echo "==> Uploading to ${REPO} under ${REMOTE_PATH} ..."
"${HF[@]}" upload --repo-type dataset "$REPO" "$STAGE" "$REMOTE_PATH" \
  --commit-message "$COMMIT_MSG"

echo ""
echo "Done. View it at:"
echo "  https://huggingface.co/datasets/${REPO}/tree/main/${REMOTE_PATH}"

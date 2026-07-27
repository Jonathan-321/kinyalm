#!/usr/bin/env bash
# Stage Tessy's reviewed distillation slice and upload it to the HF data-lake.
#
# Run once from the repo root, AFTER you have logged in:
#     huggingface-cli login          # paste a WRITE token
#     bash scripts/push_tessy_contribution.sh
#
# It stages the contribution folder (adds CONTRIBUTION.md provenance) and
# uploads it to kinyalm/kinyalm-data-lake under incoming/tessy/.

set -euo pipefail

CONTRIBUTOR="tessy"
BATCH_ID="distillation-review-2026-07"
HF_USERNAME="tessy17"
REVIEW_FILE="data/reviewed/tessy_distillation_queue.jsonl"
STAGE_DIR="$HOME/KinyaLMData/hf_contributions/incoming/${CONTRIBUTOR}/${BATCH_ID}"
REMOTE_PATH="incoming/${CONTRIBUTOR}/${BATCH_ID}"

echo "==> Confirming you are logged in to Hugging Face..."
huggingface-cli whoami

echo "==> Staging contribution..."
python3 scripts/stage_hf_contribution.py "${REVIEW_FILE}" \
  --contributor "${CONTRIBUTOR}" \
  --batch-id "${BATCH_ID}" \
  --hf-username "${HF_USERNAME}" \
  --source-note "reviewed distillation-queue slice, 326 conversations" \
  --training-permission yes \
  --redistribution-permission yes \
  --contains-sensitive-data no \
  --contains-benchmark-rows no

echo "==> Uploading to kinyalm/kinyalm-data-lake under ${REMOTE_PATH} ..."
huggingface-cli upload --repo-type dataset kinyalm/kinyalm-data-lake \
  "${STAGE_DIR}" "${REMOTE_PATH}"

echo ""
echo "Done. View it at:"
echo "  https://huggingface.co/datasets/kinyalm/kinyalm-data-lake/tree/main/${REMOTE_PATH}"

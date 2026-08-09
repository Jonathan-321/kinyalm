# SFT Data

This folder is for small, reviewable supervised fine-tuning JSONL files.

Do not put raw scraped data, model outputs, checkpoints, or large generated
files here.

Before adding `seed_conversations.jsonl`:

1. Follow `docs/data/sft-data-schema.md`.
2. Record every source in `docs/data/source-log.md`.
3. Keep unreviewed examples marked `needs-review`.
4. Use `benchmark-only` for held-out examples that must not train the model.
5. Validate the file:

   ```bash
   python3 scripts/validate_sft_jsonl.py data/sft/seed_conversations.jsonl
   ```

Training rows must have `review_status=approved` before any fine-tune starts.

The first seed gate is 100 reviewed examples. A serious tutor SFT run should
target about 1,000 reviewed examples, following
`docs/data/data-overhaul-plan.md`.

The first committed human-reviewed contribution is:

```text
tessy-distill-review.train.jsonl       258 conversations
tessy-distill-review.validation.jsonl   30 conversations
```

These files preserve complete multi-turn conversations. See
`docs/data/tessy-distillation-contribution.md` for provenance, regeneration,
the 38 withheld critic-disputed rows, and MLX staging instructions.

Generated draft batches live outside Git and are tracked in
`docs/data/sft-draft-batches.md`.

Small generation profiles are tracked under `data/sft/draft-profiles/`. They
contain reviewable source facts and templates, not generated training rows.
Regenerate Batch 002 with:

```bash
python3 scripts/generate_sft_draft_batch.py \
  --batch-id sft-drafts-2026-07-13-batch-002 \
  --profile-file data/sft/draft-profiles/useful-gap-v1.yaml \
  --compare-jsonl ~/KinyaLMData/drafts/sft-drafts-2026-07-13-batch-001.jsonl \
  --review-shards 3
```

## Native-review SFT 1000

Promote the approved Google Sheet export without breaking multi-turn
conversations:

```bash
python3 scripts/build_native_review_sft.py \
  --review-csv outputs/generation/gemini-web-sft1000/kinyalm-sft-1000-native-approved.csv \
  --output-dir outputs/datasets/kinyalm-native-review-sft1000-v1
```

The command requires exactly 1,000 rows that have a named reviewer,
`review_status=approved` (or a populated corrected version), no active failure
flags, and `approved_for_training=true`. It writes hash-pinned `train.jsonl`,
`validation.jsonl`, and `dataset-manifest.json` files for `train_qlora.py`.

The approved v1 package is pinned in the gated Hugging Face data lake at
revision `8b6ec4be68f6e0a0d110334a3778eac19c614072` under
`data/reviewed/kinyalm-native-review-sft1000-v1/`. Downloading through
`scripts/download_reviewed_sft.py` rechecks all row counts and file hashes
before a training process can use it.

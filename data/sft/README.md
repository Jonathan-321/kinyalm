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

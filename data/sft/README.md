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

# Data Manifests

This folder stores small JSON records for large or shared data files.

A manifest is not the dataset. It is the reproducibility record that tells the
team:

- where the file lives,
- who owns it,
- what stage it is in,
- whether it can train,
- how many rows/files it has,
- which source/review status applies,
- what checksum identifies it.

Create manifests with:

```bash
python3 scripts/create_data_manifest.py PATH_TO_FILE_OR_FOLDER \
  --dataset-id sft-drafts-2026-07-13-batch-001 \
  --stage draft \
  --owner Jonathan \
  --storage google-drive \
  --storage-uri "KinyaLM Shared Data/02_generated_drafts/..." \
  --source-status team-authored \
  --review-status needs-review \
  --can-train no \
  --can-redistribute no \
  --out data/manifests/sft-drafts-2026-07-13-batch-001.json
```

Keep manifests in Git even when the dataset itself lives in Drive, Hugging Face,
or local ignored storage.

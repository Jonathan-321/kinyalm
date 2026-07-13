# Hugging Face Data Lake

KinyaLM uses a private Hugging Face Dataset repo as the shared data lake for
review-ready artifacts.

Default repo:

```text
https://huggingface.co/datasets/Jonnyyy/kinyalm-data-lake
```

The repo should remain private while it contains draft examples.

Current status:

| Field | Value |
| --- | --- |
| Visibility | private |
| Latest verified upload | Batch 001 |
| HF commit | `a65a71086805273516fec4edc42b370f253de8fb` |
| Upload manifest | `data/manifests/hf-datalake-sft-drafts-2026-07-13-batch-001.json` |

## Local Staging

The upload script stages a clean HF folder under:

```text
~/KinyaLMData/hf_datalake/kinyalm-data-lake
```

It includes:

- `README.md` dataset card,
- `datalake-index.json`,
- draft JSONL files,
- review TSV files,
- upload package zip files,
- manifests with checksums.

Current uploaded file paths:

```text
README.md
datalake-index.json
data/drafts/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.jsonl
data/drafts/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.summary.md
review/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.review.tsv
packages/sft-drafts-2026-07-13-batch-001-review-package.zip
manifests/sft-drafts-2026-07-13-batch-001.json
manifests/sft-drafts-2026-07-13-batch-001-review-sheet.json
manifests/sft-drafts-2026-07-13-batch-001-review-package.json
```

## Upload Command

First confirm authentication:

```bash
huggingface-cli whoami
```

Then run:

```bash
python3 scripts/upload_hf_datalake.py \
  --repo-id Jonnyyy/kinyalm-data-lake \
  --commit-message "Upload Batch 001 draft review package"
```

The script disables Xet uploads by default because the local `hf_xet` wheel can
be architecture-specific on macOS. The current Batch 001 files are small enough
for regular Hub uploads.

Dry-run without uploading:

```bash
python3 scripts/upload_hf_datalake.py --dry-run
```

## Sharing

For team access, add collaborators to the private Hugging Face Dataset repo or
move the dataset under a shared Hugging Face organization.

## Training Rule

Do not train from the HF datalake `draft` folders. Only train from files created
by `scripts/promote_reviewed_sft.py` after fluent-speaker review.

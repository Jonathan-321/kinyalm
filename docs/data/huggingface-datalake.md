# Hugging Face Data Lake

KinyaLM uses a public-gated Hugging Face Dataset repo as the shared data lake
for review-ready artifacts.

Default repo:

```text
https://huggingface.co/datasets/Jonnyyy/kinyalm-data-lake
```

The repo page is visible, but file access requires a logged-in Hugging Face
account and gated access acceptance. Batch rows are still drafts and are not
approved training data.

Current status:

| Field | Value |
| --- | --- |
| Visibility | public-gated |
| Gated access | automatic approval |
| Latest verified upload | Batch 001 |
| HF commit | `09c902031649f1062b6f2b2837ca713a75af77e8` |
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

For fastest team access, send teammates:

```text
https://huggingface.co/datasets/Jonnyyy/kinyalm-data-lake
```

They should sign in to Hugging Face and accept the gated access prompt. A shared
Hugging Face organization is still better once the team needs shared write/admin
access instead of read/download access.

## Training Rule

Do not train from the HF datalake `draft` folders. Only train from files created
by `scripts/promote_reviewed_sft.py` after fluent-speaker review.

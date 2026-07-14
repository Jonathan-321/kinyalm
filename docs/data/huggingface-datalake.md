# Hugging Face Data Lake

KinyaLM uses a public-gated Hugging Face Dataset repo as the shared data lake
for review-ready artifacts.

Default repo:

```text
https://huggingface.co/datasets/kinyalm/kinyalm-data-lake
```

The repo page is visible, but file access requires a logged-in Hugging Face
account and gated access acceptance. Batch rows are still drafts and are not
approved training data.

Current status:

| Field | Value |
| --- | --- |
| Visibility | public-gated |
| Gated access | automatic approval |
| Latest verified upload | Corrected Batch 001 + Batch 002 |
| Draft rows | 1,000 |
| Remote files | 21, including `.gitattributes` |
| HF commit | `4879dd0878d06a910cd57eccdf543b1ca984c389` |
| Upload manifest | `data/manifests/hf-datalake-sft-drafts-2026-07-13-batches-001-002.json` |

## Local Staging

The upload script stages a clean HF folder under:

```text
~/KinyaLMData/hf_datalake/kinyalm-data-lake
```

It includes:

- `README.md` dataset card,
- `datalake-index.json`,
- draft JSONL files,
- master review TSV files and optional non-overlapping review shards,
- upload package zip files,
- manifests with checksums.

Current uploaded file paths:

```text
README.md
REVIEW_INSTRUCTIONS.md
datalake-index.json
data/drafts/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.jsonl
data/drafts/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.summary.md
review/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.review.tsv
packages/sft-drafts-2026-07-13-batch-001-review-package.zip
manifests/sft-drafts-2026-07-13-batch-001.json
manifests/sft-drafts-2026-07-13-batch-001-review-sheet.json
manifests/sft-drafts-2026-07-13-batch-001-review-package.json
data/drafts/sft-drafts-2026-07-13-batch-002/sft-drafts-2026-07-13-batch-002.jsonl
data/drafts/sft-drafts-2026-07-13-batch-002/sft-drafts-2026-07-13-batch-002.summary.md
review/sft-drafts-2026-07-13-batch-002/sft-drafts-2026-07-13-batch-002.review.tsv
review/sft-drafts-2026-07-13-batch-002/shards/*.tsv
packages/sft-drafts-2026-07-13-batch-002-review-package.zip
manifests/sft-drafts-2026-07-13-batch-002.json
manifests/sft-drafts-2026-07-13-batch-002-review-sheet.json
manifests/sft-drafts-2026-07-13-batch-002-review-package.json
```

## Upload Command

Prerequisites:

- `huggingface_hub` is installed,
- the maintainer is logged in with `hf auth login` or `huggingface-cli login`,
- the token/account has repository write access in the `kinyalm` organization,
- local inputs exist under `~/KinyaLMData/drafts`, `~/KinyaLMData/reviewed`,
  and `~/KinyaLMData/packages`,
- matching manifests exist under `data/manifests`.

First confirm authentication:

```bash
huggingface-cli whoami
```

Then run:

```bash
python3 scripts/upload_hf_datalake.py \
  --repo-id kinyalm/kinyalm-data-lake \
  --batch-id sft-drafts-2026-07-13-batch-001 \
  --batch-id sft-drafts-2026-07-13-batch-002 \
  --commit-message "Publish corrected Batch 001 and Batch 002 review queue"
```

Repeat `--batch-id` for every batch that should appear in the generated dataset
card and `datalake-index.json`. The script disables Xet uploads by default
because the local `hf_xet` wheel can be architecture-specific on macOS.

Dry-run without uploading:

```bash
python3 scripts/upload_hf_datalake.py --dry-run
```

Do not use ungated public visibility for draft rows. The script defaults to
`public-gated` and requires an explicit override before `--visibility public`
can be used.

## Post-Upload Checks

After upload, the script prints the verified HF state:

```text
private=False, gated=auto, sha=<hf-commit>
```

If the commit changes, update the status table above and the upload manifest in
`data/manifests/`.

## Sharing

For fastest team access, send teammates:

```text
https://huggingface.co/datasets/kinyalm/kinyalm-data-lake
```

They should sign in to Hugging Face and accept the gated access prompt. The
dataset is owned by the `kinyalm` organization. Official organization
invitation emails were sent to Tessy and Bonheur on July 13, 2026 with the
organization's default `write` role. They need to accept the invitation before
they can upload directly.

## Adding Data

Public-gated access lets teammates read and download. Organization membership
and repository write permission are separate requirements for direct uploads.

For teammate uploads, use:

```text
docs/team/hf-data-contribution-workflow.md
```

Fast path:

1. teammate stages a package under `incoming/<username>/<batch-id>/`,
2. teammate uploads through the organization or opens an HF pull request,
3. Jonathan or the data owner checks source/review status,
4. accepted files are moved into the appropriate datalake folder.

Helper command:

```bash
python3 scripts/stage_hf_contribution.py ./my-data-file.jsonl \
  --contributor Tessy \
  --batch-id batch-002 \
  --github-username TessyMugisha \
  --training-permission unknown \
  --redistribution-permission unknown
```

Reviewer instructions:

```text
docs/team/hf-datalake-reviewer-onboarding.md
```

Copy-ready invite messages:

```text
docs/team/access-message-templates.md
```

## Training Rule

Do not train from the HF datalake `draft` folders. Only train from files created
by `scripts/promote_reviewed_sft.py` after fluent-speaker review.

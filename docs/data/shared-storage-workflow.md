# Shared Data Storage Workflow

KinyaLM should use Git as the project record, not as the warehouse for every
large file.

Use three storage layers:

| Layer | Location | Purpose | Stored In Git? |
| --- | --- | --- | --- |
| Local data lake | `~/KinyaLMData` or an external drive | Working copies, raw downloads, generated drafts, cleaned files | no |
| Google Drive | shared team folder | Team review, handoff, spreadsheet-style collaboration | manifest only |
| Hugging Face Dataset repo | `Jonathan-321/kinyalm-*` or team org | Clean, licensed, publishable dataset versions | manifest and dataset card |

## Recommended Local Layout

Create one folder outside this repository:

```bash
mkdir -p ~/KinyaLMData/{raw,interim,processed,drafts,reviewed,approved,manifests}
```

Use the same meaning everywhere:

| Folder | Meaning |
| --- | --- |
| `raw/` | Exact downloaded or received files. Never edit in place. |
| `interim/` | Cleaned, decoded, deduplicated, or normalized files. |
| `processed/` | Final prepared files for a run or upload. |
| `drafts/` | Generated candidate examples that are not trainable yet. |
| `reviewed/` | Human-reviewed files that may still need fixes. |
| `approved/` | Trainable files that passed source and language review. |
| `manifests/` | Local copies of manifest JSON files committed to Git. |

Do not put private or sensitive data in shared folders.

## Google Drive Folder Layout

Create a shared folder named:

```text
KinyaLM Shared Data
```

Suggested subfolders:

```text
00_admin/
01_sources_raw/
02_generated_drafts/
03_review_queue/
04_approved_sft/
05_benchmarks_eval_only/
06_hf_release_candidates/
99_archive/
```

Rules:

- `01_sources_raw/`: source owner and license must be recorded before use.
- `02_generated_drafts/`: never train directly from this folder.
- `03_review_queue/`: Tessy/Bonheur review and mark pass, fix, or reject.
- `04_approved_sft/`: only reviewed, trainable data.
- `05_benchmarks_eval_only/`: evaluation data only; never copy into SFT.
- `06_hf_release_candidates/`: cleaned files ready for Hugging Face upload.

## Hugging Face Dataset Layout

Use Hugging Face for clean and licensed releases, not for every draft.
For team-scale review handoff, use the gated data lake workflow in:

```text
docs/data/huggingface-datalake.md
```

Recommended repos:

| Repo | Visibility | Purpose |
| --- | --- | --- |
| `Jonnyyy/kinyalm-data-lake` | public-gated | Shared review datalake for draft batches and manifests. |
| `Jonathan-321/kinyalm-sft-drafts` | private | Optional draft handoff if Drive is not enough. |
| `Jonathan-321/kinyalm-sft-reviewed` | private first, public only when safe | Reviewed SFT files. |
| `Jonathan-321/kinyalm-eval` | private or public | Evaluation prompts and benchmark wrappers. |

Each HF dataset repo should include:

- `README.md` dataset card,
- license field,
- source and attribution notes,
- row counts and splits,
- clear warning for `draft` or `benchmark-only` data,
- no blocked or unclear-source rows.

## Version Naming

Use names that encode stage, date, batch, and status:

```text
sft-drafts-2026-07-13-batch-001.jsonl
sft-review-queue-2026-07-13-batch-001.jsonl
sft-approved-v0.1.jsonl
sft-approved-v0.1.review.tsv
kinyarwanda-mt-clean-v0.1.jsonl
```

Never overwrite an approved version. Make a new version instead.

## Manifest Rule

Every shared or uploaded data file/folder needs a manifest committed under:

```text
data/manifests/
```

The manifest records:

- dataset ID,
- stage,
- owner,
- storage location,
- local source path,
- row count if known,
- file count and bytes,
- SHA-256 checksum for single files,
- source-log status,
- review status,
- whether it can train,
- whether it can be redistributed,
- notes and next action.

Create a manifest with:

```bash
python3 scripts/create_data_manifest.py \
  ~/KinyaLMData/drafts/sft-drafts-2026-07-13-batch-001.jsonl \
  --dataset-id sft-drafts-2026-07-13-batch-001 \
  --stage draft \
  --owner Jonathan \
  --storage google-drive \
  --storage-uri "KinyaLM Shared Data/02_generated_drafts/sft-drafts-2026-07-13-batch-001.jsonl" \
  --source-status team-authored \
  --review-status needs-review \
  --can-train no \
  --can-redistribute no \
  --out data/manifests/sft-drafts-2026-07-13-batch-001.json
```

## Upload Flow

### 1. Local First

Generate or collect data under `~/KinyaLMData`, not inside the repo.

### 2. Manifest Before Sharing

Create the manifest before uploading to Drive or Hugging Face. If the file
changes later, regenerate the manifest.

### 3. Upload To Drive For Review

Use Google Drive web upload, Google Drive for desktop, or `rclone`.

With `rclone`:

```bash
rclone config
rclone copy ~/KinyaLMData/drafts/sft-drafts-2026-07-13-batch-001.jsonl \
  gdrive:"KinyaLM Shared Data/02_generated_drafts/"
```

Use `copy` for one-way uploads. Use `sync` only when you are certain the
destination should mirror the local folder, because sync can delete remote
files that are not present locally.

### 4. Review And Promote

Move files through Drive folders:

```text
02_generated_drafts -> 03_review_queue -> 04_approved_sft
```

Only `04_approved_sft` can feed training.

### 5. Upload Clean Releases To Hugging Face

Use a private dataset repo first:

```bash
python3 -m pip install -U huggingface_hub
huggingface-cli login
```

Upload with Python:

```python
from huggingface_hub import HfApi, create_repo

repo_id = "Jonathan-321/kinyalm-sft-reviewed"
create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

api = HfApi()
api.upload_folder(
    folder_path="~/KinyaLMData/approved/sft-approved-v0.1",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="data/sft-approved-v0.1",
)
```

Keep the Hugging Face dataset private until source permissions, attribution,
and reviewer approval are all clean.

## Minimum Approval Gate

A file can move into `approved/` only if:

1. every training row has `review_status=approved`,
2. every training row has `source_status=approved`, `team-authored`, or
   `manual`,
3. no row has `split=draft`,
4. no row is copied from `benchmark-only` prompts,
5. the manifest says `can_train=true`,
6. the source log has any required attribution,
7. a reviewer has accepted the language quality.

## What The Repo Stores

Commit:

- manifests,
- schemas,
- dataset cards,
- generation prompts,
- review rubrics,
- tiny reviewed sample files if safe.

Do not commit:

- raw corpora,
- unreviewed generated bulk files,
- private Drive exports,
- model checkpoints,
- datasets with unclear permission.

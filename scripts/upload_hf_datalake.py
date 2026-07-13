"""Stage and upload the KinyaLM shared datalake to Hugging Face.

The script creates a local staging folder from approved manifest entries and
uploads it to a public-gated Hugging Face Dataset repository. Draft rows stay
marked as draft; this is a review datalake, not approved training data.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_REPO_ID = "Jonnyyy/kinyalm-data-lake"
DEFAULT_STAGE_DIR = Path("~/KinyaLMData/hf_datalake/kinyalm-data-lake").expanduser()
DEFAULT_BATCH_ID = "sft-drafts-2026-07-13-batch-001"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument(
        "--visibility",
        choices=("public-gated", "private", "public"),
        default="public-gated",
        help="HF repo visibility/access mode to enforce before upload.",
    )
    parser.add_argument(
        "--allow-ungated-public",
        action="store_true",
        help="Required with --visibility public because drafts should stay gated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit-message", default="Upload KinyaLM datalake batch")
    args = parser.parse_args()

    if args.visibility == "public" and not args.allow_ungated_public:
        parser.error("--visibility public requires --allow-ungated-public")

    staged_files = stage_batch(args.batch_id, args.stage_dir.expanduser())
    print(f"staged {len(staged_files)} files under {args.stage_dir.expanduser()}")
    for path in staged_files:
        print(f"- {path.relative_to(args.stage_dir.expanduser())}")

    if args.dry_run:
        print("dry run complete; no Hugging Face upload attempted")
        return 0

    info = upload_to_hf(
        stage_dir=args.stage_dir.expanduser(),
        repo_id=args.repo_id,
        visibility=args.visibility,
        commit_message=args.commit_message,
    )
    print(f"uploaded datalake to https://huggingface.co/datasets/{args.repo_id}")
    print(
        "verified HF state: "
        f"private={info.private}, "
        f"gated={getattr(info, 'gated', None)}, "
        f"sha={info.sha}"
    )
    return 0


def stage_batch(batch_id: str, stage_dir: Path) -> list[Path]:
    """Create the local folder layout that will be uploaded to HF."""

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    local_data = Path("~/KinyaLMData").expanduser()
    files = [
        (
            local_data / "drafts" / f"{batch_id}.jsonl",
            stage_dir / "data" / "drafts" / batch_id / f"{batch_id}.jsonl",
        ),
        (
            local_data / "drafts" / f"{batch_id}.summary.md",
            stage_dir / "data" / "drafts" / batch_id / f"{batch_id}.summary.md",
        ),
        (
            local_data / "reviewed" / f"{batch_id}.review.tsv",
            stage_dir / "review" / batch_id / f"{batch_id}.review.tsv",
        ),
        (
            local_data / "packages" / f"{batch_id}-review-package.zip",
            stage_dir / "packages" / f"{batch_id}-review-package.zip",
        ),
        (
            ROOT / "data" / "manifests" / f"{batch_id}.json",
            stage_dir / "manifests" / f"{batch_id}.json",
        ),
        (
            ROOT / "data" / "manifests" / f"{batch_id}-review-sheet.json",
            stage_dir / "manifests" / f"{batch_id}-review-sheet.json",
        ),
        (
            ROOT / "data" / "manifests" / f"{batch_id}-review-package.json",
            stage_dir / "manifests" / f"{batch_id}-review-package.json",
        ),
    ]

    copied: list[Path] = []
    for source, destination in files:
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    readme_path = stage_dir / "README.md"
    readme_path.write_text(dataset_card(batch_id=batch_id), encoding="utf-8")
    copied.append(readme_path)

    index_path = stage_dir / "datalake-index.json"
    index_path.write_text(index_json(batch_id=batch_id), encoding="utf-8")
    copied.append(index_path)

    review_instructions_path = stage_dir / "REVIEW_INSTRUCTIONS.md"
    review_instructions_path.write_text(
        review_instructions(batch_id=batch_id),
        encoding="utf-8",
    )
    copied.append(review_instructions_path)

    return sorted(copied)


def upload_to_hf(
    *,
    stage_dir: Path,
    repo_id: str,
    visibility: str,
    commit_message: str,
):
    # The local hf_xet wheel can be architecture-specific. Plain HTTP uploads
    # are enough for these small review packages and avoid that dependency.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import HfApi, create_repo

    private = visibility == "private"
    gated = "auto" if visibility == "public-gated" else False

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api = HfApi()
    api.update_repo_settings(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        gated=gated,
    )
    info = api.dataset_info(repo_id)
    if info.private != private:
        # Some hub versions do not flip personal dataset visibility through
        # update_repo_settings, so keep a compatibility fallback.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            api.update_repo_visibility(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
            )
        api.update_repo_settings(
            repo_id=repo_id,
            repo_type="dataset",
            gated=gated,
        )
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return api.dataset_info(repo_id)


def dataset_card(*, batch_id: str) -> str:
    return f"""---
license: other
pretty_name: KinyaLM Data Lake
language:
- rw
- en
tags:
- kinyarwanda
- sft
- data-lake
- draft
size_categories:
- n<1K
---

# KinyaLM Data Lake

This public-gated dataset repository stores KinyaLM data artifacts for team
review. Visitors can see the dataset page, but file access requires a
logged-in Hugging Face account and gated access acceptance.

Current contents:

- `{batch_id}` draft SFT JSONL
- `{batch_id}` review TSV
- `REVIEW_INSTRUCTIONS.md`
- `{batch_id}` review package zip
- manifests with checksums and review status

## Status

This is not an approved training dataset. Batch rows are draft examples marked
`split=draft` and `review_status=needs-review`.

## Review Rule

Rows can train only after fluent-speaker review promotes them to:

- `review_status=approved`
- `split=train` or `split=validation`

Use the project promotion script after review:

```bash
python3 scripts/promote_reviewed_sft.py \\
  --draft-jsonl ~/KinyaLMData/drafts/{batch_id}.jsonl \\
  --review-tsv ~/KinyaLMData/reviewed/{batch_id}.review.tsv \\
  --out-dir ~/KinyaLMData/approved/sft-approved-batch-001
```

## Source And License Notes

Batch 001 is synthetic draft data generated for review. It is not redistributed
as an approved public training dataset until the team confirms quality and
release terms.
"""


def review_instructions(*, batch_id: str) -> str:
    return f"""# Review Instructions

This datalake contains draft review data for KinyaLM. These rows are not
approved training data.

## What To Review

Review this TSV:

```text
review/{batch_id}/{batch_id}.review.tsv
```

## How To Return Review

Download the TSV, edit it in a spreadsheet editor, save/export it as TSV or
tab-separated text, and send the completed file back to Jonathan. Do not rename
columns.

Only edit these columns:

- `review_status`
- `correctness_1_5`
- `naturalness_1_5`
- `helpfulness_1_5`
- `failure_tags`
- `reviewer`
- `reviewer_notes`

Do not edit:

- `id`
- `task_type`
- `user_prompt`
- `assistant_response`

If an answer needs different wording, mark it `needs-fix` and put the exact
suggested replacement in `reviewer_notes`. The promotion script only promotes
original draft rows that are marked `approved`.

## Approved Row Rule

Use `approved` only when the row can train as written. Approved rows need:

- correctness, naturalness, and helpfulness scores of 4 or 5,
- a filled reviewer name,
- no failure tags.

Use comma-separated failure tags when helpful:

```text
wrong-translation, wrong-grammar, awkward-kinyarwanda, unclear, too-advanced,
too-much-english, too-much-kinyarwanda, unsupported-claim,
hallucinated-culture, overconfident-uncertainty, bad-register, unsafe,
source-risk, benchmark-leak, privacy-risk, format-error, duplicate, off-task
```
"""


def index_json(*, batch_id: str) -> str:
    index = {
        "datalake": "KinyaLM Data Lake",
        "visibility": "public-gated",
        "gated_access": "auto",
        "batches": [
            {
                "batch_id": batch_id,
                "stage": "draft",
                "review_status": "needs-review",
                "can_train": False,
                "paths": {
                    "review_instructions": "REVIEW_INSTRUCTIONS.md",
                    "draft_jsonl": f"data/drafts/{batch_id}/{batch_id}.jsonl",
                    "summary": f"data/drafts/{batch_id}/{batch_id}.summary.md",
                    "review_tsv": f"review/{batch_id}/{batch_id}.review.tsv",
                    "package": f"packages/{batch_id}-review-package.zip",
                    "manifests": "manifests/",
                },
            }
        ],
    }
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

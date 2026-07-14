"""Create manifests and a review package for one local SFT draft batch."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.manifest import create_manifest, write_manifest  # noqa: E402
from kinyalm.data.sft import load_jsonl, validate_sft_records  # noqa: E402

DEFAULT_DATA_ROOT = Path("~/KinyaLMData").expanduser()
DEFAULT_MANIFEST_DIR = ROOT / "data" / "manifests"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--owner", default="Jonathan")
    args = parser.parse_args()

    outputs = package_batch(
        batch_id=args.batch_id,
        data_root=args.data_root.expanduser(),
        manifest_dir=args.manifest_dir.expanduser(),
        owner=args.owner,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


def package_batch(
    *,
    batch_id: str,
    data_root: Path,
    manifest_dir: Path,
    owner: str,
) -> dict[str, Path]:
    """Validate, manifest, and zip one draft batch for human review."""

    draft_path = data_root / "drafts" / f"{batch_id}.jsonl"
    summary_path = data_root / "drafts" / f"{batch_id}.summary.md"
    review_path = data_root / "reviewed" / f"{batch_id}.review.tsv"
    required = (draft_path, summary_path, review_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing batch artifacts: {joined}")

    records = load_jsonl(draft_path)
    failures = [result for result in validate_sft_records(records) if not result.ok]
    if failures:
        raise ValueError(f"draft JSONL has {len(failures)} invalid rows")
    row_count = len(records)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    draft_manifest_path = manifest_dir / f"{batch_id}.json"
    review_manifest_path = manifest_dir / f"{batch_id}-review-sheet.json"
    package_manifest_path = manifest_dir / f"{batch_id}-review-package.json"

    common = {
        "owner": owner,
        "source_status": "team-authored",
        "review_status": "needs-review",
        "can_train": False,
        "can_redistribute": False,
        "contains_benchmark_rows": False,
        "contains_private_or_sensitive_data": False,
        "license": "not-applicable",
        "row_count": row_count,
    }
    draft_manifest = create_manifest(
        draft_path,
        dataset_id=batch_id,
        stage="draft",
        storage="local",
        storage_uri=str(draft_path),
        notes=(
            "Generated candidate SFT examples. Requires fluent-speaker review "
            "before any row can train."
        ),
        next_action="Complete the review TSV and promote only approved rows.",
        **common,
    )
    write_manifest(draft_manifest, draft_manifest_path)

    review_manifest = create_manifest(
        review_path,
        dataset_id=f"{batch_id}-review-sheet",
        stage="review-queue",
        storage="local",
        storage_uri=str(review_path),
        notes=(
            "Reviewer TSV for correctness, naturalness, helpfulness, failure "
            "tags, and approval decisions."
        ),
        next_action="Assign review shards and merge completed decisions.",
        **common,
    )
    write_manifest(review_manifest, review_manifest_path)

    package_dir = data_root / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / f"{batch_id}-review-package.zip"
    shard_paths = sorted(
        review_path.parent.glob(f"{batch_id}.review.part-*-of-*.tsv")
    )
    package_members = [
        (draft_path, draft_path.name),
        (summary_path, summary_path.name),
        (review_path, review_path.name),
        (draft_manifest_path, draft_manifest_path.name),
        (review_manifest_path, review_manifest_path.name),
    ]
    profile_id = records[0].get("generation_profile") if records else None
    if profile_id:
        profile_path = ROOT / "data" / "sft" / "draft-profiles" / f"{profile_id}.yaml"
        if profile_path.exists():
            package_members.append(
                (profile_path, f"generation-profile/{profile_path.name}")
            )
    package_members.extend(
        (path, f"review-shards/{path.name}") for path in shard_paths
    )
    with zipfile.ZipFile(
        package_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for source, archive_name in package_members:
            archive.write(source, arcname=archive_name)

    package_manifest = create_manifest(
        package_path,
        dataset_id=f"{batch_id}-review-package",
        stage="review-queue",
        storage="local",
        storage_uri=str(package_path),
        notes=(
            "Upload-ready zip containing the draft JSONL, review TSV, optional "
            "review shards, summary, and manifests."
        ),
        next_action="Upload this package to the shared Hugging Face datalake.",
        **common,
    )
    write_manifest(package_manifest, package_manifest_path)

    return {
        "draft_manifest": draft_manifest_path,
        "review_manifest": review_manifest_path,
        "review_package": package_path,
        "package_manifest": package_manifest_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())

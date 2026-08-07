#!/usr/bin/env python3
"""Download and verify a pinned human-reviewed SFT package from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from kinyalm.data.sft import load_jsonl, validate_sft_records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_package(
    source_dir: Path,
    *,
    minimum_rows: int = 500,
    maximum_rows: int = 1000,
) -> tuple[Path, Path, Path]:
    manifest_path = source_dir / "dataset-manifest.json"
    train_path = source_dir / "train.jsonl"
    validation_path = source_dir / "validation.jsonl"
    for path in (manifest_path, train_path, validation_path):
        if not path.is_file():
            raise ValueError(f"missing reviewed dataset artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_tier") != "human-reviewed-recovery-sft":
        raise ValueError("dataset tier is not human-reviewed-recovery-sft")
    if manifest.get("human_reviewed") is not True:
        raise ValueError("dataset manifest must state human_reviewed=true")
    if manifest.get("training_eligible") is not True:
        raise ValueError("dataset manifest must state training_eligible=true")

    records = load_jsonl(train_path) + load_jsonl(validation_path)
    if not minimum_rows <= len(records) <= maximum_rows:
        raise ValueError(
            f"reviewed row count must be {minimum_rows}-{maximum_rows}; "
            f"found {len(records)}"
        )
    failures = [result for result in validate_sft_records(records) if not result.ok]
    if failures:
        raise ValueError("downloaded reviewed rows fail the project SFT schema")
    expected_outputs = manifest.get("outputs", {})
    for name, path in (("train", train_path), ("validation", validation_path)):
        expected = expected_outputs.get(name, {})
        if expected.get("rows") != len(load_jsonl(path)):
            raise ValueError(f"{name} row count does not match manifest")
        if expected.get("sha256") != _sha256(path):
            raise ValueError(f"{name} sha256 does not match manifest")
    return manifest_path, train_path, validation_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="kinyalm/kinyalm-data-lake")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--path-in-repo",
        default="data/reviewed/native-recovery-rewrites-v1",
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=500)
    parser.add_argument("--maximum-rows", type=int, default=1000)
    args = parser.parse_args()

    if args.source_dir:
        source_dir = args.source_dir.expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download

        prefix = args.path_in_repo.strip("/")
        snapshot = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=[f"{prefix}/*"],
            )
        )
        source_dir = snapshot / prefix
    try:
        paths = verify_package(
            source_dir,
            minimum_rows=args.minimum_rows,
            maximum_rows=args.maximum_rows,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"reviewed dataset verification failed: {exc}") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, args.output_dir / path.name)
    print(f"Verified reviewed SFT package: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

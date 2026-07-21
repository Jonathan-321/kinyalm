#!/usr/bin/env python3
"""Publish model bake-off artifacts to a versioned Hugging Face dataset path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="kinyalm/kinyalm-data-lake")
    parser.add_argument("--path-in-repo", required=True)
    return parser.parse_args()


def validate_run_dir(path: Path) -> dict[str, object]:
    """Require a manifest and at least one durable raw result file."""

    manifest_path = path / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be a JSON object")

    raw_files = sorted((path / "raw").glob("*.jsonl"))
    if not raw_files:
        raise ValueError(f"no raw result files found under {path / 'raw'}")
    if not any(file_path.stat().st_size > 0 for file_path in raw_files):
        raise ValueError("raw result files are empty")
    return manifest


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = validate_run_dir(run_dir)
    token = os.environ.get("HF_PUBLISH_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_PUBLISH_TOKEN or HF_TOKEN before publishing.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=args.path_in_repo.strip("/"),
        ignore_patterns=[
            "private/**",
            "errors/**",
            "publish.log",
            "status",
        ],
        commit_message=(
            "Publish multilingual bake-off artifacts for "
            f"{manifest.get('run_name', run_dir.name)}"
        ),
    )
    print(
        f"Published to https://huggingface.co/datasets/{args.repo_id}/tree/main/"
        f"{args.path_in_repo.strip('/')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

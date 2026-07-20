#!/usr/bin/env python3
"""Publish a completed experimental adapter and its provenance to HF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--system-info", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Publish publicly. The experimental baseline is private by default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def validate_publication_inputs(
    adapter_dir: Path,
    dataset_manifest: Path,
    training_log: Path,
    system_info: Path,
) -> None:
    required = [
        adapter_dir / "adapter_config.json",
        adapter_dir / "run-preflight.json",
        dataset_manifest,
        training_log,
        system_info,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    adapter_weights = list(adapter_dir.glob("adapter_model.*"))
    if not adapter_weights:
        missing.append(str(adapter_dir / "adapter_model.*"))
    if missing:
        details = ", ".join(missing)
        raise SystemExit(f"cannot publish; missing required artifacts: {details}")


def build_run_metadata(args: argparse.Namespace) -> dict:
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    dataset_manifest = Path(args.dataset_manifest).expanduser().resolve()
    training_log = Path(args.training_log).expanduser().resolve()
    system_info = Path(args.system_info).expanduser().resolve()
    tracked_files = [
        adapter_dir / "adapter_config.json",
        adapter_dir / "run-preflight.json",
        *sorted(adapter_dir.glob("adapter_model.*")),
        dataset_manifest,
        training_log,
        system_info,
    ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "status": "experimental-critic-filtered",
        "human_reviewed": False,
        "production_eligible": False,
        "base_model": {
            "repo_id": args.base_model,
            "revision": args.base_model_revision,
        },
        "dataset": {
            "repo_id": args.dataset_repo,
            "revision": args.dataset_revision,
        },
        "artifacts": [file_metadata(path) for path in tracked_files],
    }


def render_model_card(args: argparse.Namespace, metadata: dict) -> str:
    return f"""---
base_model:
- {args.base_model}
datasets:
- {args.dataset_repo}
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- kinyarwanda
- experimental
---

# KinyaLM Track 2 Baseline A

This repository contains the QLoRA adapter from `{args.run_id}`.

## Status

This is an **experimental critic-filtered baseline**. Its examples have not
completed fluent-human review, so the adapter is not production-eligible and
must not be presented as a released KinyaLM model.

## Provenance

- Base model: `{args.base_model}` at `{args.base_model_revision}`
- Dataset: `{args.dataset_repo}` at `{args.dataset_revision}`
- Dataset tier: `{metadata['status']}`
- Human reviewed: `false`
- Training method: 4-bit NF4 QLoRA

The `run/` directory contains the immutable dataset manifest, preflight
manifest, system information, and training log used to audit this run.
"""


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    dataset_manifest = Path(args.dataset_manifest).expanduser().resolve()
    training_log = Path(args.training_log).expanduser().resolve()
    system_info = Path(args.system_info).expanduser().resolve()
    validate_publication_inputs(
        adapter_dir, dataset_manifest, training_log, system_info
    )

    metadata = build_run_metadata(args)
    metadata_path = adapter_dir / "run-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    card_path = adapter_dir / "README.md"
    card_path.write_text(render_model_card(args, metadata), encoding="utf-8")

    if args.dry_run:
        print(f"publication dry run complete: {adapter_dir}")
        return 0

    api = HfApi(token=os.environ.get("HF_PUBLISH_TOKEN"))
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=[
            "checkpoint-*",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ],
        commit_message=f"Publish experimental run {args.run_id}",
    )
    for source, destination in [
        (dataset_manifest, "run/dataset-manifest.json"),
        (training_log, "run/train.log"),
        (system_info, "run/system-info.txt"),
    ]:
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add provenance for {args.run_id}",
        )
    print(f"published private={not args.public}: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

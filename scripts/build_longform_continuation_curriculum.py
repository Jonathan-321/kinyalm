#!/usr/bin/env python3
"""Build and optionally publish the reviewed continuation curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.human_reviewed import write_jsonl  # noqa: E402
from kinyalm.data.sft import load_jsonl, validate_sft_records  # noqa: E402
from scripts.build_native_review_sft import (  # noqa: E402
    add_held_out_overlap_report,
    prompt_overlap_report,
)
from scripts.download_reviewed_sft import verify_package  # noqa: E402

DATASET_ID = "kinyalm-team-reviewed-longform-curriculum-v1"
SOURCE_DATASET_ID = "kinyalm-team-reviewed-longform-sft3144-v1"
DEFAULT_SOURCE_DIR = ROOT / "outputs/datasets" / SOURCE_DATASET_ID
DEFAULT_OUTPUT_DIR = ROOT / "outputs/datasets" / DATASET_ID
DEFAULT_PATH_IN_REPO = f"data/reviewed/{DATASET_ID}"
DEFAULT_HELD_OUT_CONFIG = ROOT / "configs/evaluation/gemma4_recovery_bakeoff.json"
TARGET_FAMILIES = (
    "formal-informal",
    "sentence-correction",
    "uncertainty-clarification",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assistant_turn_count(record: dict[str, Any]) -> int:
    return sum(message["role"] == "assistant" for message in record["messages"])


def build_curriculum(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    dataset_id: str = DATASET_ID,
    target_families: tuple[str, ...] = TARGET_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Weight approved weak families once without changing their language."""

    base_train = [deepcopy(record) for record in train_records]
    validation = [deepcopy(record) for record in validation_records]
    duplicates: list[dict[str, Any]] = []
    source_ids = {record["id"] for record in base_train + validation}
    if len(source_ids) != len(base_train) + len(validation):
        raise ValueError("source package contains duplicate IDs")

    for source_record in base_train:
        family = source_record.get("task_family", source_record["task_type"])
        if family not in target_families:
            continue
        duplicate = deepcopy(source_record)
        duplicate["id"] = f"{source_record['id']}-curriculum-repeat-1"
        duplicate["source"] = "team-reviewed-curriculum-weighting-v1"
        duplicate["dataset_version"] = dataset_id
        duplicate["curriculum_source_record_id"] = source_record["id"]
        duplicate["curriculum_repeat_index"] = 1
        duplicate["curriculum_reason"] = family
        duplicate["reviewer_notes"] = (
            f"{source_record['reviewer_notes']} Curriculum v1 repeats this "
            "approved conversation without changing its messages."
        )
        duplicates.append(duplicate)

    for record in base_train:
        record["dataset_version"] = dataset_id

    train = sorted(base_train + duplicates, key=lambda record: record["id"])
    validation = sorted(validation, key=lambda record: record["id"])
    results = validate_sft_records(train + validation)
    failures = [result for result in results if not result.ok]
    if failures:
        details = "; ".join(
            f"row {result.line_number}: {', '.join(result.errors)}"
            for result in failures[:5]
        )
        raise ValueError(f"curriculum rows failed validation: {details}")

    overlap = prompt_overlap_report(train, validation)
    if overlap["exact_pairs"] or overlap["near_pairs"]:
        raise ValueError("curriculum introduced train/validation prompt leakage")

    duplicate_family_counts = Counter(
        record["curriculum_reason"] for record in duplicates
    )
    report = {
        "dataset_id": dataset_id,
        "conversation_count": len(train) + len(validation),
        "assistant_turn_count": sum(
            assistant_turn_count(record) for record in train + validation
        ),
        "multi_turn_conversation_count": sum(
            assistant_turn_count(record) > 1 for record in train + validation
        ),
        "split_counts": {"train": len(train), "validation": len(validation)},
        "split_assistant_turn_counts": {
            "train": sum(assistant_turn_count(record) for record in train),
            "validation": sum(
                assistant_turn_count(record) for record in validation
            ),
        },
        "task_family_counts": dict(
            sorted(
                Counter(
                    record.get("task_family", record["task_type"])
                    for record in train + validation
                ).items()
            )
        ),
        "curriculum": {
            "method": "deterministic one-repeat weighting",
            "target_families": list(target_families),
            "duplicate_conversations": len(duplicates),
            "duplicate_assistant_turns": sum(
                assistant_turn_count(record) for record in duplicates
            ),
            "duplicate_family_counts": dict(sorted(duplicate_family_counts.items())),
            "message_content_changed": False,
            "validation_changed": False,
        },
        "train_validation_prompt_overlap": overlap,
        "team_reviewed": True,
        "complete_conversation_split": True,
    }
    return train, validation, report


def write_package(
    *,
    source_dir: Path,
    output_dir: Path,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    report: dict[str, Any],
    source_revision: str,
    dataset_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    source_manifest = json.loads(
        (source_dir / "dataset-manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_id,
        "dataset_tier": "team-reviewed-experimental-sft",
        "human_reviewed": True,
        "team_reviewed": True,
        "training_eligible": True,
        "production_eligible": False,
        "redistribution_eligible": False,
        "source": {
            "repo_id": "kinyalm/kinyalm-data-lake",
            "resolved_revision": source_revision,
            "path_in_repo": f"data/reviewed/{SOURCE_DATASET_ID}",
            "dataset_id": source_manifest["dataset_id"],
            "manifest_sha256": sha256(source_dir / "dataset-manifest.json"),
            "train_sha256": sha256(source_dir / "train.jsonl"),
            "validation_sha256": sha256(source_dir / "validation.jsonl"),
        },
        "build": report,
        "outputs": {
            "train": {
                "path": train_path.name,
                "rows": len(train),
                "sha256": sha256(train_path),
            },
            "validation": {
                "path": validation_path.name,
                "rows": len(validation),
                "sha256": sha256(validation_path),
            },
        },
        "quality_note": (
            "Controlled continuation curriculum derived only from reviewed rows. "
            "Repeated rows change sampling weight; they introduce no new language."
        ),
    }
    (output_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def publish_package(output_dir: Path, repo_id: str, path_in_repo: str) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=path_in_repo.strip("/"),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Publish {output_dir.name} continuation curriculum",
    )
    return api.dataset_info(repo_id).sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--repo-id", default="kinyalm/kinyalm-data-lake")
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument(
        "--held-out-config", type=Path, default=DEFAULT_HELD_OUT_CONFIG
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise SystemExit("source_revision must be a 40-character commit SHA")

    try:
        verify_package(args.source_dir, minimum_rows=3144, maximum_rows=3144)
        train, validation, report = build_curriculum(
            load_jsonl(args.source_dir / "train.jsonl"),
            load_jsonl(args.source_dir / "validation.jsonl"),
            dataset_id=args.dataset_id,
        )
        add_held_out_overlap_report(
            report,
            train + validation,
            args.held_out_config,
        )
        held_out_overlap = report["held_out_prompt_overlap"]
        if held_out_overlap["exact_pairs"] or held_out_overlap["near_pairs"]:
            raise ValueError("curriculum overlaps the held-out benchmark")
        manifest = write_package(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            train=train,
            validation=validation,
            report=report,
            source_revision=args.source_revision,
            dataset_id=args.dataset_id,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"continuation curriculum build failed: {exc}") from exc

    print(json.dumps(manifest["build"], ensure_ascii=False, indent=2))
    if args.publish:
        revision = publish_package(args.output_dir, args.repo_id, args.path_in_repo)
        print(f"published_revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

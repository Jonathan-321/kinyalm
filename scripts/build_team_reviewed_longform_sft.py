#!/usr/bin/env python3
"""Build and optionally publish the complete team-reviewed long-form SFT set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from kinyalm.data.human_reviewed import (  # noqa: E402
    stratified_conversation_split,
    write_jsonl,
)
from kinyalm.data.sft import load_jsonl, validate_sft_records  # noqa: E402
from scripts.build_native_review_sft import (  # noqa: E402
    add_held_out_overlap_report,
    convert_review_row,
    load_review_csv,
    prompt_overlap_report,
)
from scripts.download_reviewed_sft import verify_package  # noqa: E402

DATASET_ID = "kinyalm-team-reviewed-longform-sft3144-v1"
DEFAULT_PATH_IN_REPO = f"data/reviewed/{DATASET_ID}"
DEFAULT_HELD_OUT_CONFIG = ROOT / "configs/evaluation/gemma4_recovery_bakeoff.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine_records(
    native_records: list[dict[str, Any]],
    longform_records: list[dict[str, Any]],
    *,
    dataset_id: str = DATASET_ID,
    train_ratio: float = 0.9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combine reviewed sources, then make one leak-resistant 90/10 split."""

    combined: list[dict[str, Any]] = []
    for source_record in native_records:
        record = deepcopy(source_record)
        record["dataset_version"] = dataset_id
        combined.append(record)
    for source_record in longform_records:
        record = deepcopy(source_record)
        record.update(
            {
                "source": "gemini-3.1-pro-team-reviewed-longform-v1",
                "curation_tier": "team-reviewed-distillation",
                "dataset_version": dataset_id,
            }
        )
        combined.append(record)

    ids = Counter(record["id"] for record in combined)
    duplicate_ids = [row_id for row_id, count in ids.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate IDs: {', '.join(sorted(duplicate_ids)[:5])}")

    fingerprints = Counter(
        json.dumps(
            [
                {
                    "role": message["role"],
                    "content": " ".join(message["content"].casefold().split()),
                }
                for message in record["messages"]
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in combined
    )
    duplicate_conversations = sum(
        count - 1 for count in fingerprints.values() if count > 1
    )
    if duplicate_conversations:
        raise ValueError(
            f"duplicate normalized conversations: {duplicate_conversations}"
        )

    combined = stratified_conversation_split(
        combined,
        train_ratio=train_ratio,
        split_seed=dataset_id,
        dataset_version=dataset_id,
    )
    combined, repaired_overlap_rows = repair_split_overlap(
        combined, split_seed=dataset_id
    )

    failures = [result for result in validate_sft_records(combined) if not result.ok]
    if failures:
        details = "; ".join(
            f"row {failure.line_number}: {', '.join(failure.errors)}"
            for failure in failures[:5]
        )
        raise ValueError(f"combined SFT rows failed validation: {details}")

    train = [record for record in combined if record["split"] == "train"]
    validation = [
        record for record in combined if record["split"] == "validation"
    ]
    if len(train) + len(validation) != len(combined):
        raise ValueError("combined package contains a non-training split")
    split_overlap = prompt_overlap_report(train, validation)
    if split_overlap["exact_pairs"] or split_overlap["near_pairs"]:
        raise ValueError("train/validation prompt leakage detected")

    report = {
        "dataset_id": dataset_id,
        "conversation_count": len(combined),
        "assistant_turn_count": sum(
            sum(message["role"] == "assistant" for message in record["messages"])
            for record in combined
        ),
        "multi_turn_conversation_count": sum(
            sum(message["role"] == "assistant" for message in record["messages"])
            > 1
            for record in combined
        ),
        "split_counts": dict(
            sorted(Counter(record["split"] for record in combined).items())
        ),
        "split_assistant_turn_counts": {
            split: sum(
                sum(message["role"] == "assistant" for message in record["messages"])
                for record in combined
                if record["split"] == split
            )
            for split in ("train", "validation")
        },
        "task_family_counts": dict(
            sorted(
                Counter(
                    record.get("task_family", record["task_type"])
                    for record in combined
                ).items()
            )
        ),
        "source_counts": dict(
            sorted(Counter(record["source"] for record in combined).items())
        ),
        "train_validation_prompt_overlap": split_overlap,
        "split_overlap_rows_reassigned": repaired_overlap_rows,
        "team_reviewed": True,
        "complete_conversation_split": True,
    }
    return sorted(combined, key=lambda row: (row["split"], row["id"])), report


def _prompt_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = set(left["messages"][0]["content"].casefold().split())
    right_tokens = set(right["messages"][0]["content"].casefold().split())
    union = len(left_tokens | right_tokens)
    return len(left_tokens & right_tokens) / union if union else 1.0


def repair_split_overlap(
    records: list[dict[str, Any]],
    *,
    split_seed: str,
    threshold: float = 0.88,
) -> tuple[list[dict[str, Any]], int]:
    """Keep near-identical prompts together while preserving split size."""

    working = [deepcopy(record) for record in records]
    reassigned_ids: set[str] = set()
    for _ in range(10):
        train = [record for record in working if record["split"] == "train"]
        validation = [
            record for record in working if record["split"] == "validation"
        ]
        leaking_validation_ids = {
            validation_row["id"]
            for validation_row in validation
            if any(
                _prompt_similarity(train_row, validation_row) >= threshold
                for train_row in train
            )
        }
        if not leaking_validation_ids:
            return working, len(reassigned_ids)

        deficits = Counter()
        for record in working:
            if record["id"] in leaking_validation_ids:
                deficits[record["task_type"]] += 1
                record["split"] = "train"
                reassigned_ids.add(record["id"])

        for task_type, needed in sorted(deficits.items()):
            for _ in range(needed):
                train = [record for record in working if record["split"] == "train"]
                candidates = []
                for candidate in train:
                    if candidate["task_type"] != task_type:
                        continue
                    if candidate["id"] in reassigned_ids:
                        continue
                    other_train = [
                        record for record in train if record["id"] != candidate["id"]
                    ]
                    if any(
                        _prompt_similarity(candidate, other) >= threshold
                        for other in other_train
                    ):
                        continue
                    digest = hashlib.sha256(
                        f"{split_seed}\0{candidate['id']}".encode()
                    ).hexdigest()
                    candidates.append((digest, candidate))
                if not candidates:
                    raise ValueError(
                        f"cannot repair split overlap for task type {task_type}"
                    )
                replacement = min(candidates, key=lambda item: item[0])[1]
                replacement["split"] = "validation"
                reassigned_ids.add(replacement["id"])
    raise ValueError("split overlap repair did not converge")


def write_package(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    report: dict[str, Any],
    native_dir: Path,
    review_csv: Path,
    source_revision: str,
    dataset_id: str,
) -> dict[str, Any]:
    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

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
            "native_reviewed_package": {
                "dataset_id": "kinyalm-native-review-sft1000-v1.1",
                "conversations": 1000,
                "manifest_sha256": sha256(native_dir / "dataset-manifest.json"),
            },
            "team_reviewed_longform": {
                "conversations": 2144,
                "review_csv_sha256": sha256(review_csv),
                "reviewer": "KinyaLM project team",
                "approval_scope": "controlled-experimental-sft",
            },
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
            "Approved for controlled team experiments. Production promotion "
            "requires the held-out base-versus-adapter evaluation."
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
        commit_message=f"Publish {output_dir.name} training package",
    )
    return api.dataset_info(repo_id).sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--longform-review-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--repo-id", default="kinyalm/kinyalm-data-lake")
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--held-out-config", type=Path, default=DEFAULT_HELD_OUT_CONFIG
    )
    args = parser.parse_args()
    if len(args.source_revision) != 40:
        raise SystemExit("source_revision must be a 40-character commit SHA")

    try:
        verify_package(args.native_dir, minimum_rows=1000, maximum_rows=1000)
        native_records = load_jsonl(args.native_dir / "train.jsonl") + load_jsonl(
            args.native_dir / "validation.jsonl"
        )
        longform_records = [
            convert_review_row(row, dataset_id=args.dataset_id)
            for row in load_review_csv(args.longform_review_csv)
        ]
        if any(record is None for record in longform_records):
            raise ValueError("every long-form review row must be approved")
        if len(longform_records) != 2144:
            raise ValueError("long-form review must contain exactly 2,144 rows")
        records, report = combine_records(
            native_records,
            [record for record in longform_records if record is not None],
            dataset_id=args.dataset_id,
        )
        if report["conversation_count"] != 3144:
            raise ValueError("combined package must contain exactly 3,144 rows")
        if report["assistant_turn_count"] != 6930:
            raise ValueError("combined package must contain exactly 6,930 responses")
        add_held_out_overlap_report(report, records, args.held_out_config)
        manifest = write_package(
            output_dir=args.output_dir,
            records=records,
            report=report,
            native_dir=args.native_dir,
            review_csv=args.longform_review_csv,
            source_revision=args.source_revision,
            dataset_id=args.dataset_id,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"team-reviewed SFT build failed: {exc}") from exc

    print(json.dumps(manifest["build"], ensure_ascii=False, indent=2))
    if args.publish:
        revision = publish_package(args.output_dir, args.repo_id, args.path_in_repo)
        print(f"published_revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

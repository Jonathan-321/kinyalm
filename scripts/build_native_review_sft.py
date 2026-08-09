#!/usr/bin/env python3
"""Promote an approved native-review CSV into a pinned SFT package.

The input is one row per complete conversation. Only rows explicitly marked
approved (or corrected) and approved_for_training are promoted. Corrected rows
use corrected_messages_json; approved rows use messages_json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.human_reviewed import (  # noqa: E402
    stratified_conversation_split,
    write_jsonl,
)
from kinyalm.data.sft import validate_sft_records  # noqa: E402
from kinyalm.evaluation import (  # noqa: E402
    benchmark_tasks,
    load_bakeoff_config,
    load_task_bank,
)

DEFAULT_DATASET_ID = "kinyalm-native-review-sft1000-v1"
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1NA_XGNsz6SY-ksEc_jHiUa8LA-tfmPC8HW6WiRUOQLs/"
    "edit?gid=356553873#gid=356553873"
)
DEFAULT_HELD_OUT_CONFIG = ROOT / "configs/evaluation/gemma4_recovery_bakeoff.json"
TASK_FAMILY_TO_TYPE = {
    "natural-conversation": "dialogue",
    "en-to-rw": "translation-en-rw",
    "sentence-correction": "sentence-correction",
    "uncertainty-clarification": "uncertainty",
    "beginner-tutoring": "grammar-explanation",
    "morphology": "grammar-explanation",
    "rw-to-en": "translation-rw-en",
    "reading-comprehension": "reading-comprehension",
    "code-switching": "code-switching",
    "formal-informal": "culture-register",
}
REQUIRED_COLUMNS = {
    "conversation_id",
    "batch_id",
    "task_family",
    "difficulty",
    "source_language",
    "target_language",
    "user_turns",
    "assistant_turns",
    "is_multi_turn",
    "messages_json",
    "teacher_model",
    "generation_status",
    "reviewer_name",
    "review_status",
    "corrected_messages_json",
    "reviewer_notes",
    "repetition_flag",
    "factual_error_flag",
    "naturalness_score_1_5",
    "correctness_score_1_5",
    "approved_for_training",
}
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"", "0", "false", "no", "n"}


def parse_bool(value: Any, *, field: str, row_id: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{row_id}: {field} must be a boolean, found {value!r}")


def load_review_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or []))
        if missing:
            raise ValueError("review CSV is missing columns: " + ", ".join(missing))
        return list(reader)


def convert_review_row(row: dict[str, str], *, dataset_id: str) -> dict | None:
    source_id = row.get("conversation_id", "").strip()
    if not source_id:
        raise ValueError("review row is missing conversation_id")

    status = row.get("review_status", "").strip().casefold()
    approved_for_training = parse_bool(
        row.get("approved_for_training", ""),
        field="approved_for_training",
        row_id=source_id,
    )
    if status not in {"approved", "corrected"} or not approved_for_training:
        return None
    if parse_bool(
        row.get("repetition_flag", ""),
        field="repetition_flag",
        row_id=source_id,
    ):
        raise ValueError(f"{source_id}: approved row still has repetition_flag=true")
    if parse_bool(
        row.get("factual_error_flag", ""),
        field="factual_error_flag",
        row_id=source_id,
    ):
        raise ValueError(f"{source_id}: approved row still has factual_error_flag=true")

    reviewer = row.get("reviewer_name", "").strip()
    if not reviewer:
        raise ValueError(f"{source_id}: approved row is missing reviewer_name")

    corrected_json = row.get("corrected_messages_json", "").strip()
    if status == "corrected":
        if not corrected_json:
            raise ValueError(
                f"{source_id}: corrected row is missing corrected_messages_json"
            )
        messages_text = corrected_json
        review_basis = "corrected_messages_json"
    else:
        if corrected_json:
            raise ValueError(
                f"{source_id}: approved row has unused corrected_messages_json"
            )
        messages_text = row.get("messages_json", "").strip()
        review_basis = "messages_json"

    try:
        messages = json.loads(messages_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_id}: invalid {review_basis}: {exc}") from exc
    _validate_messages(messages, source_id)

    user_turns = sum(message["role"] == "user" for message in messages)
    assistant_turns = sum(message["role"] == "assistant" for message in messages)
    declared_user_turns = _parse_positive_int(
        row.get("user_turns", ""), "user_turns", source_id
    )
    if declared_user_turns != user_turns:
        raise ValueError(f"{source_id}: user_turns does not match messages_json")
    if (
        _parse_positive_int(
            row.get("assistant_turns", ""), "assistant_turns", source_id
        )
        != assistant_turns
    ):
        raise ValueError(f"{source_id}: assistant_turns does not match messages_json")
    expected_multi = parse_bool(
        row.get("is_multi_turn", ""), field="is_multi_turn", row_id=source_id
    )
    if expected_multi != (assistant_turns > 1):
        raise ValueError(f"{source_id}: is_multi_turn does not match messages_json")

    family = row.get("task_family", "").strip()
    try:
        task_type = TASK_FAMILY_TO_TYPE[family]
    except KeyError as exc:
        raise ValueError(f"{source_id}: unknown task_family {family!r}") from exc

    canonical_id = source_id.casefold()
    notes = row.get("reviewer_notes", "").strip()
    return {
        "id": canonical_id,
        "task_type": task_type,
        "split": "draft",
        "source": "gemini-web-native-review-sft1000-v1",
        "source_status": "team-authored",
        "review_status": "approved",
        "language_mix": _language_mix(row),
        "messages": messages,
        "reviewer_notes": (
            f"Reviewer: {reviewer}. Basis: {review_basis}. "
            f"Original task_family={family}. {notes}"
        ).strip(),
        "dataset_version": dataset_id,
        "curation_tier": "native-reviewed-distillation",
        "reviewers": [reviewer],
        "source_record_id": source_id,
        "batch_id": row.get("batch_id", "").strip(),
        "task_family": family,
        "difficulty": row.get("difficulty", "").strip(),
        "teacher_model": row.get("teacher_model", "").strip(),
        "generation_status": row.get("generation_status", "").strip(),
        "assistant_turn_count": assistant_turns,
    }


def build_records(
    review_rows: list[dict[str, str]],
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    train_ratio: float = 0.9,
    split_seed: str | None = None,
    expected_records: int | None = 1000,
) -> tuple[list[dict], dict[str, Any]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be greater than 0 and less than 1")

    converted: list[dict] = []
    withheld: list[str] = []
    for row in review_rows:
        record = convert_review_row(row, dataset_id=dataset_id)
        if record is None:
            withheld.append(row.get("conversation_id", "<missing id>"))
        else:
            converted.append(record)

    if expected_records is not None and len(converted) != expected_records:
        raise ValueError(
            f"expected exactly {expected_records} training-approved rows; "
            f"found {len(converted)} ({len(withheld)} withheld)"
        )
    _reject_duplicates(converted)

    records = stratified_conversation_split(
        converted,
        train_ratio=train_ratio,
        split_seed=split_seed or dataset_id,
        dataset_version=dataset_id,
    )
    failures = [result for result in validate_sft_records(records) if not result.ok]
    if failures:
        details = "; ".join(
            f"row {failure.line_number}: {', '.join(failure.errors)}"
            for failure in failures[:5]
        )
        raise ValueError(f"converted SFT rows failed schema validation: {details}")

    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    split_overlap = prompt_overlap_report(train, validation)
    if split_overlap["exact_pairs"] or split_overlap["near_pairs"]:
        raise ValueError("train/validation prompt leakage detected")

    report = {
        "conversation_count": len(records),
        "assistant_turn_count": sum(
            record["assistant_turn_count"] for record in records
        ),
        "multi_turn_conversation_count": sum(
            record["assistant_turn_count"] > 1 for record in records
        ),
        "split_counts": dict(sorted(Counter(r["split"] for r in records).items())),
        "task_family_counts": dict(
            sorted(Counter(r["task_family"] for r in records).items())
        ),
        "task_type_counts": dict(
            sorted(Counter(r["task_type"] for r in records).items())
        ),
        "reviewer_counts": dict(
            sorted(Counter(r["reviewers"][0] for r in records).items())
        ),
        "corrected_conversation_count": sum(
            row.get("review_status", "").strip().casefold() == "corrected"
            for row in review_rows
        ),
        "withheld_conversation_ids": withheld,
        "complete_conversation_split": True,
        "train_validation_prompt_overlap": split_overlap,
    }
    return records, report


def prompt_overlap_report(
    left_rows: list[dict],
    right_rows: list[dict],
    *,
    threshold: float = 0.88,
) -> dict[str, Any]:
    """Compare first-user prompts without exposing their text in the report."""

    exact_pairs = 0
    near_pairs = 0
    right = [
        (
            _normalized_prompt(record),
            set(_normalized_prompt(record).split()),
        )
        for record in right_rows
    ]
    for record in left_rows:
        normalized = _normalized_prompt(record)
        tokens = set(normalized.split())
        for other_normalized, other_tokens in right:
            if normalized == other_normalized:
                exact_pairs += 1
            union = len(tokens | other_tokens)
            similarity = len(tokens & other_tokens) / union if union else 1.0
            if similarity >= threshold:
                near_pairs += 1
    return {
        "threshold": threshold,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "exact_pairs": exact_pairs,
        "near_pairs": near_pairs,
    }


def add_held_out_overlap_report(
    report: dict[str, Any],
    records: list[dict],
    config_path: Path,
) -> None:
    config = load_bakeoff_config(config_path)
    tasks = benchmark_tasks(load_task_bank(ROOT / config.task_bank))
    held_out_rows = [
        {"messages": [{"role": "user", "content": task.prompt}]} for task in tasks
    ]
    overlap = prompt_overlap_report(records, held_out_rows)
    if overlap["exact_pairs"] or overlap["near_pairs"]:
        raise ValueError(
            "approved SFT prompts overlap with the held-out evaluation bank"
        )
    try:
        config_label = str(config_path.resolve().relative_to(ROOT))
    except ValueError:
        config_label = str(config_path)
    report["held_out_prompt_overlap"] = {
        **overlap,
        "config": config_label,
    }


def write_package(
    *,
    csv_path: Path,
    output_dir: Path,
    records: list[dict],
    report: dict[str, Any],
    dataset_id: str,
    train_ratio: float,
    split_seed: str,
    sheet_url: str,
) -> dict[str, Any]:
    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "dataset-manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_id,
        "dataset_tier": "human-reviewed-recovery-sft",
        "human_reviewed": True,
        "training_eligible": True,
        "production_eligible": False,
        "redistribution_eligible": False,
        "source": {
            "review_sheet_url": sheet_url,
            "approved_csv": {
                "path": csv_path.name,
                "sha256": _sha256(csv_path),
                "bytes": csv_path.stat().st_size,
            },
            "teacher_models": sorted({r["teacher_model"] for r in records}),
        },
        "build": {
            **report,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
        },
        "outputs": {
            "train": {
                "path": train_path.name,
                "rows": len(train),
                "sha256": _sha256(train_path),
            },
            "validation": {
                "path": validation_path.name,
                "rows": len(validation),
                "sha256": _sha256(validation_path),
            },
        },
        "quality_note": (
            "All promoted rows were explicitly approved for training by the "
            "named native reviewer in the source sheet. Row-level numeric scores "
            "were optional and are not treated as additional evidence. Production "
            "eligibility remains false until base-versus-adapter evaluation passes."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_messages(messages: Any, row_id: str) -> None:
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError(f"{row_id}: messages must contain complete turns")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(f"{row_id}: messages[{index}] must have role={expected}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{row_id}: messages[{index}] has empty content")


def _parse_positive_int(value: str, field: str, row_id: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row_id}: {field} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{row_id}: {field} must be positive")
    return parsed


def _language_mix(row: dict[str, str]) -> str:
    source = row.get("source_language", "").strip().casefold()
    target = row.get("target_language", "").strip().casefold()
    if source == target == "rw":
        return "kinyarwanda"
    if source == target == "en":
        return "english"
    return "kinyarwanda+english"


def _normalized_prompt(record: dict) -> str:
    return " ".join(record["messages"][0]["content"].casefold().split())


def _reject_duplicates(records: list[dict]) -> None:
    ids = Counter(record["id"] for record in records)
    duplicate_ids = sorted(row_id for row_id, count in ids.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate ids: {', '.join(duplicate_ids[:5])}")

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
        for record in records
    )
    duplicate_conversations = sum(
        count - 1 for count in fingerprints.values() if count > 1
    )
    if duplicate_conversations:
        raise ValueError(
            f"duplicate normalized conversations: {duplicate_conversations}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", default=None)
    parser.add_argument("--expected-records", type=int, default=1000)
    parser.add_argument("--sheet-url", default=DEFAULT_SHEET_URL)
    parser.add_argument(
        "--held-out-config",
        type=Path,
        default=DEFAULT_HELD_OUT_CONFIG,
    )
    args = parser.parse_args()

    split_seed = args.split_seed or args.dataset_id
    try:
        rows = load_review_csv(args.review_csv)
        records, report = build_records(
            rows,
            dataset_id=args.dataset_id,
            train_ratio=args.train_ratio,
            split_seed=split_seed,
            expected_records=args.expected_records,
        )
        add_held_out_overlap_report(report, records, args.held_out_config)
        manifest = write_package(
            csv_path=args.review_csv.expanduser(),
            output_dir=args.output_dir.expanduser(),
            records=records,
            report=report,
            dataset_id=args.dataset_id,
            train_ratio=args.train_ratio,
            split_seed=split_seed,
            sheet_url=args.sheet_url,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"native-review promotion failed: {exc}") from exc

    print(f"Conversations: {report['conversation_count']}")
    print(f"Assistant responses: {report['assistant_turn_count']}")
    print(f"Multi-turn conversations: {report['multi_turn_conversation_count']}")
    print(f"Splits: {report['split_counts']}")
    print(f"Output: {args.output_dir.expanduser()}")
    print(f"Manifest training_eligible: {manifest['training_eligible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

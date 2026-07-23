"""Validation helpers for supervised fine-tuning conversation data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_TYPES = {
    "greeting",
    "translation-en-rw",
    "translation-rw-en",
    "grammar-explanation",
    "sentence-correction",
    "vocabulary",
    "quiz-generation",
    "dialogue",
    "uncertainty",
    "culture-register",
    "reading-comprehension",
    "sentence-generation",
    "pronunciation",
    "code-switching",
}

SPLITS = {
    "draft",
    "train",
    "validation",
    "experimental-train",
    "experimental-validation",
    "benchmark-only",
}
SOURCE_STATUSES = {
    "approved",
    "team-authored",
    "manual",
    "reference-only",
    "blocked",
    "investigate",
    "model-generated",
}
REVIEW_STATUSES = {
    "needs-review",
    "approved",
    "needs-fix",
    "rejected",
    "not-sure",
    "critic-accepted",
    "critic-repaired",
}
LANGUAGE_MIXES = {"kinyarwanda", "english", "kinyarwanda+english"}
TRAINABLE_SOURCE_STATUSES = {"approved", "team-authored", "manual"}
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SFTValidationResult:
    """Validation outcome for one JSONL row."""

    line_number: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and attach line-specific context to JSON errors."""

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            records.append(record)
    return records


def validate_sft_record(
    record: dict[str, Any],
    *,
    line_number: int,
    seen_ids: set[str] | None = None,
) -> SFTValidationResult:
    """Validate one SFT JSON object against the project schema."""

    errors: list[str] = []
    required = {
        "id",
        "task_type",
        "split",
        "source",
        "source_status",
        "review_status",
        "language_mix",
        "messages",
        "reviewer_notes",
    }
    missing = sorted(required.difference(record))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    example_id = record.get("id")
    if not isinstance(example_id, str) or not example_id:
        errors.append("id must be a non-empty string")
    elif not ID_PATTERN.fullmatch(example_id):
        errors.append("id must use lowercase letters, numbers, and hyphens")
    elif seen_ids is not None:
        if example_id in seen_ids:
            errors.append(f"duplicate id: {example_id}")
        seen_ids.add(example_id)

    _check_allowed(record, "task_type", TASK_TYPES, errors)
    _check_allowed(record, "split", SPLITS, errors)
    _check_allowed(record, "source_status", SOURCE_STATUSES, errors)
    _check_allowed(record, "review_status", REVIEW_STATUSES, errors)
    _check_allowed(record, "language_mix", LANGUAGE_MIXES, errors)

    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append("source must be a non-empty string")

    if not isinstance(record.get("reviewer_notes"), str):
        errors.append("reviewer_notes must be a string")

    _check_messages(record.get("messages"), errors)
    _check_training_gate(record, errors)

    return SFTValidationResult(line_number=line_number, errors=tuple(errors))


def validate_sft_records(records: list[dict[str, Any]]) -> list[SFTValidationResult]:
    """Validate a list of SFT records while enforcing unique IDs."""

    seen_ids: set[str] = set()
    return [
        validate_sft_record(record, line_number=index, seen_ids=seen_ids)
        for index, record in enumerate(records, start=1)
    ]


def _check_allowed(
    record: dict[str, Any],
    field: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    value = record.get(field)
    if not isinstance(value, str):
        errors.append(f"{field} must be a string")
    elif value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        errors.append(f"{field} must be one of: {allowed_text}")


def _check_messages(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("messages must be a list")
        return
    if len(value) < 2:
        errors.append("messages must contain at least one user/assistant turn")
        return
    if len(value) % 2:
        errors.append("messages must contain complete user/assistant turns")
        return

    for index, message in enumerate(value):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue
        if message.get("role") != expected_role:
            errors.append(f"messages[{index}].role must be {expected_role}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"messages[{index}].content must be a non-empty string")


def _check_training_gate(record: dict[str, Any], errors: list[str]) -> None:
    split = record.get("split")
    if split in {"experimental-train", "experimental-validation"}:
        if record.get("review_status") not in {
            "critic-accepted",
            "critic-repaired",
        }:
            errors.append(
                "experimental rows must have review_status=critic-accepted "
                "or critic-repaired"
            )
        if record.get("source_status") != "model-generated":
            errors.append(
                "experimental rows must have source_status=model-generated"
            )
        return

    if split not in {"train", "validation"}:
        return

    review_status = record.get("review_status")
    source_status = record.get("source_status")
    if review_status != "approved":
        errors.append("train/validation rows must have review_status=approved")
    if source_status not in TRAINABLE_SOURCE_STATUSES:
        errors.append(
            "train/validation rows must have source_status approved, "
            "team-authored, or manual"
        )

"""Build a leak-resistant SFT set from fluent-speaker-reviewed sources."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from kinyalm.data.sft import validate_sft_records

DATASET_ID = "human-reviewed-recovery-432-v1"


def prepare_tessy_rows(
    converted_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the original generation metadata and review-decision tier."""

    review_by_id = _unique_by(review_rows, "conversation_id", "Tessy review")
    draft_by_id = _unique_by(draft_rows, "id", "distillation draft")
    prepared: list[dict[str, Any]] = []
    for converted in converted_rows:
        row = deepcopy(converted)
        row_id = row["id"]
        if row_id not in review_by_id:
            raise ValueError(f"converted Tessy row has no review record: {row_id}")
        if row_id not in draft_by_id:
            raise ValueError(f"converted Tessy row has no source draft: {row_id}")

        review = review_by_id[row_id]
        draft = draft_by_id[row_id]
        priority = str(review.get("priority", "")).strip().casefold()
        if priority == "critic accepted":
            tier = "human-and-critic-agreed"
        elif priority == "repair first":
            tier = "human-approved-critic-disputed"
        else:
            raise ValueError(f"unknown critic priority for {row_id}: {priority!r}")

        row["language_mix"] = draft["language_mix"]
        row["curation_tier"] = tier
        row["reviewers"] = ["Tessy Mugisha"]
        row["source_record_id"] = draft.get("source_record_id", row_id)
        row["task_family"] = draft.get("task_family")
        row["difficulty"] = draft.get("difficulty")
        row["evaluation_categories"] = draft.get("evaluation_categories", [])
        prepared.append(row)
    return prepared


def prepare_existing_rows(
    rows: list[dict[str, Any]],
    *,
    tier: str,
    reviewer: str,
) -> list[dict[str, Any]]:
    """Copy already-promoted rows and add common provenance metadata."""

    prepared = []
    for source_row in rows:
        row = deepcopy(source_row)
        if row.get("review_status") != "approved":
            raise ValueError(f"row is not human-approved: {row.get('id')}")
        row["curation_tier"] = tier
        row["reviewers"] = [reviewer]
        prepared.append(row)
    return prepared


def build_dataset(
    source_groups: list[list[dict[str, Any]]],
    *,
    train_ratio: float = 0.9,
    split_seed: str = DATASET_ID,
    minimum_records: int = 400,
    include_critic_disagreements: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge, gate, deduplicate, and split complete conversations."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be greater than 0 and less than 1")
    if minimum_records < 1:
        raise ValueError("minimum_records must be positive")

    combined = [deepcopy(row) for group in source_groups for row in group]
    if not include_critic_disagreements:
        combined = [
            row
            for row in combined
            if row.get("curation_tier") != "human-approved-critic-disputed"
        ]

    _reject_duplicate_ids(combined)
    _reject_duplicate_conversations(combined)
    if len(combined) < minimum_records:
        raise ValueError(
            f"quality-gated pool has {len(combined)} records; "
            f"minimum requested is {minimum_records}"
        )

    split_rows = stratified_conversation_split(
        combined,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    validation = validate_sft_records(split_rows)
    failures = [result for result in validation if not result.ok]
    if failures:
        details = "; ".join(
            f"row {result.line_number}: {', '.join(result.errors)}"
            for result in failures[:5]
        )
        raise ValueError(f"built dataset failed SFT validation: {details}")

    report = dataset_report(
        split_rows,
        minimum_records=minimum_records,
        include_critic_disagreements=include_critic_disagreements,
    )
    return split_rows, report


def stratified_conversation_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    split_seed: str,
) -> list[dict[str, Any]]:
    """Create an exact, task-stratified split without breaking conversations."""

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)

    validation_target = round(len(rows) * (1 - train_ratio))
    allocations = _validation_allocations(by_task, validation_target)
    output: list[dict[str, Any]] = []
    for task_type, task_rows in sorted(by_task.items()):
        ranked = sorted(
            task_rows,
            key=lambda row: _split_digest(split_seed, row["id"]),
        )
        validation_ids = {
            row["id"] for row in ranked[: allocations[task_type]]
        }
        for source_row in ranked:
            row = deepcopy(source_row)
            row["split"] = (
                "validation" if row["id"] in validation_ids else "train"
            )
            row["dataset_version"] = DATASET_ID
            output.append(row)
    return sorted(output, key=lambda row: (row["split"], row["id"]))


def dataset_report(
    rows: list[dict[str, Any]],
    *,
    minimum_records: int,
    include_critic_disagreements: bool,
) -> dict[str, Any]:
    """Summarize the properties that matter before a training run."""

    split_counts = Counter(row["split"] for row in rows)
    task_counts = Counter(row["task_type"] for row in rows)
    tier_counts = Counter(row.get("curation_tier", "unknown") for row in rows)
    assistant_turns = sum(
        1
        for row in rows
        for message in row["messages"]
        if message["role"] == "assistant"
    )
    multi_turn = sum(len(row["messages"]) > 2 for row in rows)
    return {
        "dataset_id": DATASET_ID,
        "minimum_requested": minimum_records,
        "conversation_count": len(rows),
        "assistant_turn_count": assistant_turns,
        "multi_turn_conversation_count": multi_turn,
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "curation_tier_counts": dict(sorted(tier_counts.items())),
        "human_reviewed": True,
        "complete_conversation_split": True,
        "include_human_approved_critic_disagreements": (
            include_critic_disagreements
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL with stable field ordering from the input objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validation_allocations(
    by_task: dict[str, list[dict[str, Any]]],
    validation_target: int,
) -> dict[str, int]:
    task_types = sorted(by_task)
    if validation_target < len(task_types):
        raise ValueError("validation split is too small to cover every task type")

    total = sum(len(rows) for rows in by_task.values())
    ideals = {
        task: len(by_task[task]) * validation_target / total for task in task_types
    }
    allocations = {
        task: min(len(by_task[task]) - 1, max(1, math.floor(ideals[task])))
        for task in task_types
    }

    while sum(allocations.values()) < validation_target:
        candidates = [
            task
            for task in task_types
            if allocations[task] < len(by_task[task]) - 1
        ]
        if not candidates:
            raise ValueError("cannot satisfy requested validation size")
        task = max(
            candidates,
            key=lambda name: (ideals[name] - allocations[name], name),
        )
        allocations[task] += 1

    while sum(allocations.values()) > validation_target:
        candidates = [task for task in task_types if allocations[task] > 1]
        if not candidates:
            raise ValueError("cannot reduce validation split without losing coverage")
        task = min(
            candidates,
            key=lambda name: (ideals[name] - allocations[name], name),
        )
        allocations[task] -= 1
    return allocations


def _unique_by(
    rows: list[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{label} row is missing {key}")
        if value in output:
            raise ValueError(f"duplicate {label} {key}: {value}")
        output[value] = row
    return output


def _reject_duplicate_ids(rows: list[dict[str, Any]]) -> None:
    duplicates = [
        key
        for key, count in Counter(row["id"] for row in rows).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate record ids: {', '.join(sorted(duplicates)[:5])}")


def _reject_duplicate_conversations(rows: list[dict[str, Any]]) -> None:
    fingerprints = Counter(_conversation_fingerprint(row) for row in rows)
    duplicates = sum(count - 1 for count in fingerprints.values() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate normalized conversations: {duplicates}")


def _conversation_fingerprint(row: dict[str, Any]) -> str:
    normalized = [
        {
            "role": message["role"],
            "content": " ".join(message["content"].casefold().split()),
        }
        for message in row["messages"]
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_digest(seed: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()

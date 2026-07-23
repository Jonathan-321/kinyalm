"""Leakage-resistant grouping helpers for SFT train/validation splits."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

GROUP_FIELDS = (
    "source_family_id",
    "source_group_id",
    "content_key",
    "source_document_id",
    "source_record_id",
    "parent_id",
)


def grouping_keys(record: dict[str, Any]) -> tuple[str, ...]:
    """Return provenance and normalized-content keys connecting related rows."""

    keys: list[str] = []
    for field in GROUP_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            keys.append(f"{field}:{value.strip()}")

    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        for field in GROUP_FIELDS:
            value = provenance.get(field)
            if isinstance(value, str) and value.strip():
                keys.append(f"provenance.{field}:{value.strip()}")

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            normalized = normalize_text(content)
            if normalized:
                digest = sha256(normalized.encode("utf-8")).hexdigest()
                keys.append(f"normalized-{role}-sha256:{digest}")
    return tuple(dict.fromkeys(keys))


def connected_record_groups(
    records: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Join records transitively when any grouping key overlaps."""

    if not records:
        return []

    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    key_owner: dict[str, int] = {}
    record_keys: list[tuple[str, ...]] = []
    for index, record in enumerate(records):
        keys = grouping_keys(record)
        record_keys.append(keys)
        for key in keys:
            owner = key_owner.setdefault(key, index)
            union(index, owner)

    component_indexes: dict[int, list[int]] = {}
    for index in range(len(records)):
        component_indexes.setdefault(find(index), []).append(index)

    result: list[tuple[str, list[dict[str, Any]]]] = []
    for indexes in component_indexes.values():
        keys = sorted({key for index in indexes for key in record_keys[index]})
        component_id = "|".join(keys)
        result.append((component_id, [records[index] for index in indexes]))
    return sorted(result, key=lambda item: item[0])


def assign_grouped_splits(
    records: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: str,
    train_split: str,
    validation_split: str,
) -> None:
    """Assign connected groups deterministically without crossing splits."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    groups = sorted(
        connected_record_groups(records),
        key=lambda item: sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
    )
    target_train_rows = int(len(records) * train_ratio)
    train_rows = 0
    for _, group_records in groups:
        remaining = target_train_rows - train_rows
        use_train = remaining > 0 and len(group_records) <= remaining
        split = train_split if use_train else validation_split
        for record in group_records:
            record["split"] = split
        if use_train:
            train_rows += len(group_records)


def normalize_text(value: str) -> str:
    """Normalize text for exact-content grouping without losing Unicode words."""

    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())

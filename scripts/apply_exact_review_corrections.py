#!/usr/bin/env python3
"""Apply exact, auditable corrections to a native-review CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_corrections(
    rows: list[dict[str, str]],
    specification: dict,
) -> tuple[list[dict[str, str]], list[dict]]:
    by_id = {row["conversation_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("input CSV contains duplicate conversation_id values")

    applied: list[dict] = []
    for correction in specification.get("corrections", []):
        row_id = correction["conversation_id"]
        try:
            row = by_id[row_id]
        except KeyError as exc:
            raise ValueError(f"correction row not found: {row_id}") from exc
        if row.get("review_status", "").casefold() != "approved":
            raise ValueError(f"{row_id}: expected review_status=approved")
        if row.get("corrected_messages_json", "").strip():
            raise ValueError(f"{row_id}: row already has corrected messages")

        messages = json.loads(row["messages_json"])
        message_index = int(correction["message_index"])
        try:
            message = messages[message_index]
        except IndexError as exc:
            raise ValueError(f"{row_id}: message_index is out of range") from exc
        expected = correction["expected_text"]
        if message.get("content") != expected:
            raise ValueError(
                f"{row_id}: expected text does not exactly match message content"
            )
        replacement = correction["replacement_text"]
        if not replacement.strip() or replacement == expected:
            raise ValueError(f"{row_id}: replacement must be non-empty and different")

        message["content"] = replacement
        row["review_status"] = "corrected"
        row["corrected_messages_json"] = json.dumps(messages, ensure_ascii=False)
        note = (
            f"Exact correction authorized by {specification['authorized_by']} on "
            f"{specification['authorized_on']}: {correction['reason']}"
        )
        prior = row.get("reviewer_notes", "").strip()
        row["reviewer_notes"] = f"{prior} {note}".strip()
        applied.append(
            {
                "conversation_id": row_id,
                "message_index": message_index,
                "before": expected,
                "after": replacement,
                "reason": correction["reason"],
            }
        )
    return rows, applied


def main() -> int:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("input CSV is missing a header")
    specification = json.loads(args.corrections.read_text(encoding="utf-8"))
    rows, applied = apply_corrections(rows, specification)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    ledger = {
        "source_dataset_id": specification["source_dataset_id"],
        "target_dataset_id": specification["target_dataset_id"],
        "source_csv_sha256": sha256(args.input),
        "output_csv_sha256": sha256(args.output),
        "row_count": len(rows),
        "applied_corrections": applied,
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ledger, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

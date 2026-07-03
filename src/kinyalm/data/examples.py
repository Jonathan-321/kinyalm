"""Utilities for small, reviewable project example sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class TextExample:
    """A short text example used for tokenizer or tutor evaluation."""

    id: str
    category: str
    text: str
    notes: str
    status: str


def load_tsv_examples(path: str | Path) -> list[TextExample]:
    """Load the tracked tokenizer/tutor example TSV format."""

    rows: list[TextExample] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "category", "text", "notes", "status"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"missing required columns: {missing_text}")

        for line_number, row in enumerate(reader, start=2):
            example = TextExample(
                id=(row.get("id") or "").strip(),
                category=(row.get("category") or "").strip(),
                text=(row.get("text") or "").strip(),
                notes=(row.get("notes") or "").strip(),
                status=(row.get("status") or "").strip(),
            )
            if not example.id:
                raise ValueError(f"missing id on line {line_number}")
            if not example.text:
                raise ValueError(f"missing text on line {line_number}")
            rows.append(example)

    return rows

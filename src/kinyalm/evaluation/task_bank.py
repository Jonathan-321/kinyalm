"""Parse and validate the project-owned tutor task bank."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SPLITS = {"benchmark-only", "train-template"}
EXPECTED_COLUMNS = (
    "ID",
    "Category",
    "Split",
    "Learner Prompt",
    "Review Focus",
)


@dataclass(frozen=True)
class TutorTask:
    """One project-owned tutor prompt and its evaluation metadata."""

    id: str
    category: str
    split: str
    prompt: str
    review_focus: str


def load_task_bank(path: str | Path) -> list[TutorTask]:
    """Load task rows from the Markdown table at ``path``."""

    rows: list[TutorTask] = []
    header_seen = False
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.startswith("|"):
            continue
        values = _markdown_row(line)
        if tuple(values) == EXPECTED_COLUMNS:
            header_seen = True
            continue
        if not values or not values[0].startswith("T"):
            continue
        if len(values) != len(EXPECTED_COLUMNS):
            raise ValueError(
                f"task-bank row {line_number} has {len(values)} columns; "
                f"expected {len(EXPECTED_COLUMNS)}"
            )
        rows.append(
            TutorTask(
                id=values[0],
                category=values[1],
                split=values[2],
                prompt=values[3],
                review_focus=values[4],
            )
        )

    if not header_seen:
        raise ValueError("task bank is missing the expected Markdown table header")
    _validate_tasks(rows)
    return rows


def benchmark_tasks(tasks: list[TutorTask]) -> list[TutorTask]:
    """Return only rows that are permanently held out from training."""

    selected = [task for task in tasks if task.split == "benchmark-only"]
    if not selected:
        raise ValueError("task bank contains no benchmark-only rows")
    return selected


def _markdown_row(line: str) -> list[str]:
    parsed = next(csv.reader([line], delimiter="|", skipinitialspace=True))
    return [value.strip() for value in parsed[1:-1]]


def _validate_tasks(tasks: list[TutorTask]) -> None:
    if not tasks:
        raise ValueError("task bank contains no task rows")

    seen_ids: set[str] = set()
    for task in tasks:
        if task.id in seen_ids:
            raise ValueError(f"duplicate task id: {task.id}")
        seen_ids.add(task.id)
        if not task.id.startswith("T") or not task.id[1:].isdigit():
            raise ValueError(f"invalid task id: {task.id}")
        if task.split not in ALLOWED_SPLITS:
            raise ValueError(f"{task.id}: unsupported split {task.split!r}")
        for field_name in ("category", "prompt", "review_focus"):
            if not getattr(task, field_name):
                raise ValueError(f"{task.id}: {field_name} cannot be empty")

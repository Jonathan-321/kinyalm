#!/usr/bin/env python3
"""Build a blank native rewrite queue from scored baseline failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

QUEUE_COLUMNS = (
    "rewrite_id",
    "source_task_id",
    "category",
    "task_type",
    "language_mix",
    "learning_objective",
    "failure_tags",
    "rewrite_priority",
    "variation_instruction",
    "new_user_prompt",
    "gold_assistant_response",
    "review_status",
    "reviewer",
    "reviewer_notes",
)
VARIATIONS = (
    "Change all names, places, and surface vocabulary while testing the same skill.",
    "Use a different noun class or singular-plural contrast.",
    "Change the tense, polarity, or aspect while keeping the same learning objective.",
    "Write a natural learner mistake and a concise correction task.",
    "Put the skill inside a short two-turn conversational context.",
    "Change the learner level and require an appropriately concise explanation.",
    "Switch the requested explanation language without copying the held-out prompt.",
    "Create a fresh everyday scenario that tests the same underlying distinction.",
)
CATEGORY_TASK_TYPES = {
    "Greeting and introduction": "greeting",
    "Vocabulary and usage": "vocabulary",
    "Translation EN-RW": "translation-en-rw",
    "Translation RW-EN": "translation-rw-en",
    "Sentence correction": "sentence-correction",
    "Morphology and grammar": "grammar-explanation",
    "Orthography and pronunciation": "pronunciation",
    "Dialogue and conversation": "dialogue",
    "Tutoring and exercises": "grammar-explanation",
    "Culture and register": "culture-register",
    "Ambiguity and uncertainty": "uncertainty",
    "Reading and multi-turn consistency": "reading-comprehension",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=750)
    return parser.parse_args()


def build_rewrite_rows(
    review_rows: list[dict[str, str]],
    key: dict,
    *,
    candidate_id: str,
    target_rows: int,
) -> list[dict[str, str]]:
    if not 500 <= target_rows <= 1000:
        raise ValueError("target_rows must be between 500 and 1000")
    candidate_by_blind = {
        str(row["blind_id"]): str(row["candidate_id"])
        for row in key.get("rows", [])
    }
    failures = []
    for row in review_rows:
        blind_id = row.get("blind_id", "").strip()
        if candidate_by_blind.get(blind_id) != candidate_id:
            continue
        if row.get("prompt_validity", "").strip().casefold() != "valid":
            continue
        if row.get("pass_fail", "").strip().casefold() != "fail":
            continue
        if not row.get("reviewer", "").strip():
            raise ValueError(f"{blind_id}: failed row has no named reviewer")
        if not row.get("corrected_response", "").strip():
            raise ValueError(f"{blind_id}: failed row has no corrected_response")
        priority = row.get("rewrite_priority", "").strip().casefold()
        if priority not in {"high", "medium", "low", "none"}:
            raise ValueError(
                f"{blind_id}: rewrite_priority must be high, medium, low, or none"
            )
        if priority == "none":
            continue
        category = row.get("category", "").strip()
        if category not in CATEGORY_TASK_TYPES:
            raise ValueError(f"{blind_id}: unsupported category {category!r}")
        failures.append(
            {
                "blind_id": blind_id,
                "task_id": row.get("task_id", "").strip(),
                "category": category,
                "review_focus": row.get("review_focus", "").strip(),
                "failure_tags": row.get("failure_tags", "").strip(),
                "priority": priority,
            }
        )
    if not failures:
        raise ValueError("no corrected failures are eligible for rewrite expansion")
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    failures.sort(key=lambda row: (priority_rank[row["priority"]], row["blind_id"]))

    rows = []
    for index in range(target_rows):
        source = failures[index % len(failures)]
        variation = VARIATIONS[(index // len(failures)) % len(VARIATIONS)]
        rows.append(
            {
                "rewrite_id": f"recovery-rewrite-{index + 1:04d}",
                "source_task_id": source["task_id"],
                "category": source["category"],
                "task_type": CATEGORY_TASK_TYPES[source["category"]],
                "language_mix": "kinyarwanda+english",
                "learning_objective": source["review_focus"],
                "failure_tags": source["failure_tags"],
                "rewrite_priority": source["priority"],
                "variation_instruction": variation,
                "new_user_prompt": "",
                "gold_assistant_response": "",
                "review_status": "needs-review",
                "reviewer": "",
                "reviewer_notes": "",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    rows = build_rewrite_rows(
        review_rows,
        key,
        candidate_id=args.candidate_id,
        target_rows=args.target_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} blank native rewrite rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate Gemini long-form batches and prepare a native-review CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

TASK_FAMILIES = {
    "natural-conversation",
    "en-to-rw",
    "sentence-correction",
    "uncertainty-clarification",
    "beginner-tutoring",
    "morphology",
    "rw-to-en",
    "reading-comprehension",
    "code-switching",
    "formal-informal",
}
DIFFICULTIES = {"beginner", "intermediate", "advanced"}
LANGUAGES = {"rw", "en", "rw+en"}
REVIEW_FIELDS = [
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
]
STYLE_PATTERNS = {
    "generic-praise": re.compile(
        r"\b(excellent job|perfectly (?:correct|constructed|natural)|"
        r"you are absolutely correct|you are doing great)\b",
        re.IGNORECASE,
    ),
    "repeated-praise": re.compile(r"\bni byiza cyane\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--teacher-model", default="Gemini 3.1 Pro web")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--minimum-assistant-tokens", type=int, default=100)
    parser.add_argument("--maximum-assistant-tokens", type=int, default=300)
    return parser.parse_args()


def load_batches(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"{path}: expected a JSON array")
        rows.extend(parsed)
    return rows


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def validate_and_prepare(
    rows: list[dict],
    *,
    teacher_model: str,
    token_counter: Callable[[str], int],
    minimum_assistant_tokens: int = 100,
    maximum_assistant_tokens: int = 300,
) -> tuple[list[dict[str, str]], dict]:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    prepared: list[dict[str, str]] = []
    all_token_counts: list[int] = []
    family_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        row_id = _required_text(row, "conversation_id", index)
        if row_id in seen_ids:
            raise ValueError(f"duplicate conversation_id: {row_id}")
        seen_ids.add(row_id)
        family = _choice(row, "task_family", TASK_FAMILIES, row_id)
        _choice(row, "difficulty", DIFFICULTIES, row_id)
        _choice(row, "source_language", LANGUAGES, row_id)
        _choice(row, "target_language", LANGUAGES, row_id)
        batch_id = _required_text(row, "batch_id", index)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) not in {2, 4, 6, 8}:
            raise ValueError(f"{row_id}: messages must contain 2, 4, 6, or 8 items")

        assistant_tokens: list[int] = []
        warnings: set[str] = set()
        for message_index, message in enumerate(messages):
            expected = "user" if message_index % 2 == 0 else "assistant"
            if not isinstance(message, dict) or message.get("role") != expected:
                raise ValueError(
                    f"{row_id}: messages[{message_index}] must have role={expected}"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{row_id}: messages[{message_index}] is empty")
            if expected == "assistant":
                count = token_counter(content)
                assistant_tokens.append(count)
                all_token_counts.append(count)
                if count < minimum_assistant_tokens:
                    warnings.add("assistant-under-token-minimum")
                if count > maximum_assistant_tokens:
                    warnings.add("assistant-over-token-maximum")
                for name, pattern in STYLE_PATTERNS.items():
                    if pattern.search(content):
                        warnings.add(name)

        first_prompt = normalize(messages[0]["content"])
        if first_prompt in seen_prompts:
            raise ValueError(f"{row_id}: duplicate first-user prompt")
        seen_prompts.add(first_prompt)
        user_turns = len(messages) // 2
        assistant_turns = len(messages) // 2
        family_counts[family] += 1
        shape_counts[f"{len(messages)}_messages"] += 1
        warning_counts.update(warnings)
        status = "candidate-needs-fix" if warnings else "candidate"
        prepared.append(
            {
                "conversation_id": row_id,
                "batch_id": batch_id,
                "task_family": family,
                "difficulty": row["difficulty"],
                "source_language": row["source_language"],
                "target_language": row["target_language"],
                "user_turns": str(user_turns),
                "assistant_turns": str(assistant_turns),
                "is_multi_turn": "TRUE" if assistant_turns > 1 else "FALSE",
                "messages_json": json.dumps(messages, ensure_ascii=False),
                "teacher_model": teacher_model,
                "generation_status": status,
                "reviewer_name": "",
                "review_status": "needs-review",
                "corrected_messages_json": "",
                "reviewer_notes": "; ".join(sorted(warnings)),
                "repetition_flag": "FALSE",
                "factual_error_flag": "FALSE",
                "naturalness_score_1_5": "",
                "correctness_score_1_5": "",
                "approved_for_training": "FALSE",
            }
        )

    sorted_tokens = sorted(all_token_counts)
    median = (
        sorted_tokens[len(sorted_tokens) // 2]
        if sorted_tokens
        else 0
    )
    report = {
        "conversation_count": len(prepared),
        "assistant_response_count": len(all_token_counts),
        "assistant_tokens_total": sum(all_token_counts),
        "assistant_tokens_min": min(all_token_counts, default=0),
        "assistant_tokens_median": median,
        "assistant_tokens_max": max(all_token_counts, default=0),
        "conversation_shapes": dict(sorted(shape_counts.items())),
        "task_families": dict(sorted(family_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "rows_needing_fix": sum(
            row["generation_status"] == "candidate-needs-fix" for row in prepared
        ),
        "training_eligible_rows": 0,
    }
    return prepared, report


def _required_text(row: dict, field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {index}: {field} must be a non-empty string")
    return value.strip()


def _choice(row: dict, field: str, allowed: set[str], row_id: str) -> str:
    value = row.get(field)
    if value not in allowed:
        raise ValueError(f"{row_id}: unsupported {field}={value!r}")
    return value


def main() -> int:
    args = parse_args()
    rows = load_batches(args.input)
    if args.tokenizer:
        from transformers import AutoTokenizer

        kwargs = (
            {"revision": args.tokenizer_revision}
            if args.tokenizer_revision
            else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **kwargs)
        token_counter = lambda text: len(  # noqa: E731
            tokenizer.encode(text, add_special_tokens=False)
        )
    else:
        token_counter = lambda text: len(text.split())  # noqa: E731

    prepared, report = validate_and_prepare(
        rows,
        teacher_model=args.teacher_model,
        token_counter=token_counter,
        minimum_assistant_tokens=args.minimum_assistant_tokens,
        maximum_assistant_tokens=args.maximum_assistant_tokens,
    )
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(prepared)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

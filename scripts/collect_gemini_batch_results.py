#!/usr/bin/env python3
"""Validate Gemini Batch API results without changing teacher output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONVERSATION_FIELDS = {
    "conversation_id",
    "batch_id",
    "task_family",
    "difficulty",
    "source_language",
    "target_language",
    "messages",
}
MESSAGE_FIELDS = {"role", "content"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_jobs(path: Path) -> dict[str, dict[str, Any]]:
    jobs = read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("teacher job is missing job_id")
        if job_id in indexed:
            raise ValueError(f"duplicate teacher job: {job_id}")
        expected = job.get("expected_conversations")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{job_id}: expected_conversations must be non-empty")
        indexed[job_id] = job
    return indexed


def response_text(row: dict[str, Any], job_id: str) -> str:
    response = row.get("response")
    if not isinstance(response, dict):
        detail = row.get("error") or row.get("status") or "missing response"
        raise ValueError(f"{job_id}: provider error: {detail}")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{job_id}: response contains no candidates")
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise ValueError(f"{job_id}: first candidate contains no content")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"{job_id}: first candidate contains no parts")
    text_parts = [part.get("text") for part in parts if isinstance(part, dict)]
    if not text_parts or not all(isinstance(value, str) for value in text_parts):
        raise ValueError(f"{job_id}: candidate parts contain no text")
    return "".join(text_parts)


def complete_array_prefix(text: str) -> list[Any]:
    """Return complete top-level array items without repairing truncated JSON."""

    decoder = json.JSONDecoder()
    position = 0
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "[":
        return []
    position += 1
    values: list[Any] = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text) and text[position] == ",":
            position += 1
            continue
        if position >= len(text) or text[position] == "]":
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        values.append(value)
    return values


def validate_conversations(
    value: Any,
    expected: list[dict[str, Any]],
    job_id: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{job_id}: candidate output must be a JSON array")
    expected_by_id = {row["conversation_id"]: row for row in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError(f"{job_id}: teacher job contains duplicate conversation IDs")
    actual_by_id: dict[str, dict[str, Any]] = {}
    for conversation in value:
        if not isinstance(conversation, dict):
            raise ValueError(f"{job_id}: conversation must be an object")
        if set(conversation) != CONVERSATION_FIELDS:
            raise ValueError(f"{job_id}: conversation fields do not match schema")
        conversation_id = conversation.get("conversation_id")
        if conversation_id not in expected_by_id:
            raise ValueError(
                f"{job_id}: unexpected conversation_id={conversation_id!r}"
            )
        if conversation_id in actual_by_id:
            raise ValueError(f"{job_id}: duplicate conversation_id={conversation_id}")
        spec = expected_by_id[conversation_id]
        for field in (
            "batch_id",
            "task_family",
            "difficulty",
            "source_language",
            "target_language",
        ):
            if conversation.get(field) != spec[field]:
                raise ValueError(f"{job_id}/{conversation_id}: changed {field}")
        messages = conversation.get("messages")
        if not isinstance(messages, list) or len(messages) != spec["message_count"]:
            raise ValueError(f"{job_id}/{conversation_id}: wrong message count")
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if not isinstance(message, dict) or set(message) != MESSAGE_FIELDS:
                raise ValueError(
                    f"{job_id}/{conversation_id}: message {index} has wrong fields"
                )
            if message.get("role") != expected_role:
                raise ValueError(
                    f"{job_id}/{conversation_id}: message {index} must be "
                    f"{expected_role}"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"{job_id}/{conversation_id}: message {index} is empty"
                )
        actual_by_id[conversation_id] = conversation
    if set(actual_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(actual_by_id))
        raise ValueError(f"{job_id}: missing conversation IDs: {missing}")
    return [actual_by_id[spec["conversation_id"]] for spec in expected]


def salvage_complete_conversations(
    text: str,
    expected: list[dict[str, Any]],
    job_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    expected_by_id = {row["conversation_id"]: row for row in expected}
    salvaged: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for value in complete_array_prefix(text):
        if not isinstance(value, dict):
            errors.append(f"{job_id}: complete prefix item is not an object")
            continue
        conversation_id = value.get("conversation_id")
        if conversation_id not in expected_by_id:
            errors.append(
                f"{job_id}: complete prefix has unexpected conversation ID "
                f"{conversation_id!r}"
            )
            continue
        if conversation_id in seen:
            errors.append(
                f"{job_id}: duplicate complete-prefix conversation {conversation_id}"
            )
            continue
        try:
            normalized = validate_conversations(
                [value],
                [expected_by_id[conversation_id]],
                f"{job_id}/{conversation_id}",
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        seen.add(conversation_id)
        salvaged.extend(normalized)
    return salvaged, errors


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite different immutable file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def collect_results(
    jobs: dict[str, dict[str, Any]],
    result_rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    conversations_by_job: dict[str, list[dict[str, Any]]] = {}
    partial_conversations_by_job: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    raw_dir = output_dir / "raw-provider-results"

    for row in result_rows:
        job_id = row.get("key")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("provider result is missing key")
        if job_id not in jobs:
            raise ValueError(f"provider returned unknown job: {job_id}")
        if job_id in seen:
            raise ValueError(f"provider returned duplicate job: {job_id}")
        seen.add(job_id)
        serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        raw = (serialized + "\n").encode()
        write_immutable(raw_dir / f"{job_id}.jsonl", raw)
        try:
            text = response_text(row, job_id)
            parsed = json.loads(text)
            conversations_by_job[job_id] = validate_conversations(
                parsed,
                jobs[job_id]["expected_conversations"],
                job_id,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            salvaged: list[dict[str, Any]] = []
            salvage_errors: list[str] = []
            if "text" in locals():
                salvaged, salvage_errors = salvage_complete_conversations(
                    text,
                    jobs[job_id]["expected_conversations"],
                    job_id,
                )
            if salvaged:
                partial_conversations_by_job[job_id] = salvaged
            errors.append(
                {
                    "job_id": job_id,
                    "error": str(exc),
                    "salvaged_conversation_ids": [
                        row["conversation_id"] for row in salvaged
                    ],
                    "salvage_errors": salvage_errors,
                }
            )
        finally:
            text = ""

    missing = sorted(set(jobs) - seen)
    conversations = [
        conversation
        for job_id in jobs
        for conversation in (
            conversations_by_job.get(job_id)
            or partial_conversations_by_job.get(job_id, [])
        )
    ]
    report = {
        "expected_job_count": len(jobs),
        "received_job_count": len(seen),
        "valid_job_count": len(conversations_by_job),
        "partial_job_count": len(partial_conversations_by_job),
        "salvaged_conversation_count": sum(
            len(rows) for rows in partial_conversations_by_job.values()
        ),
        "conversation_count": len(conversations),
        "missing_job_ids": missing,
        "errors": errors,
        "complete": (
            not missing and not errors and len(conversations_by_job) == len(jobs)
        ),
    }
    return conversations, report


def main() -> int:
    args = parse_args()
    jobs = load_jobs(args.jobs)
    result_rows = read_jsonl(args.results)
    conversations, report = collect_results(jobs, result_rows, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["results_sha256"] = hashlib.sha256(args.results.read_bytes()).hexdigest()
    report_path = args.output_dir / "collection-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["complete"] or args.allow_partial:
        content = (
            json.dumps(conversations, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        write_immutable(args.output_dir / "candidate-conversations.raw.json", content)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

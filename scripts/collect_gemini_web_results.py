#!/usr/bin/env python3
"""Normalize only queue metadata and validate immutable Gemini web drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.collect_gemini_batch_results import (
        CONVERSATION_FIELDS,
        MESSAGE_FIELDS,
        read_jsonl,
        validate_conversations,
        write_immutable,
    )
except ModuleNotFoundError:
    from collect_gemini_batch_results import (
        CONVERSATION_FIELDS,
        MESSAGE_FIELDS,
        read_jsonl,
        validate_conversations,
        write_immutable,
    )

QUEUE_ONLY_FIELDS = {"message_count", "scenario_seed"}


def attempt_is_eligible(attempt: Path) -> bool:
    provenance_path = Path(f"{attempt}.provenance.json")
    if not provenance_path.exists():
        return True
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    return provenance.get("counts_toward_pro_token_target") is not False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def normalize_candidate(
    value: Any,
    expected: list[dict[str, Any]],
    job_id: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError(f"{job_id}: candidate output must be a JSON array")
    expected_by_id = {row["conversation_id"]: row for row in expected}
    normalized = []
    dropped: set[str] = set()
    role_corrections = []
    for conversation in value:
        if not isinstance(conversation, dict):
            raise ValueError(f"{job_id}: conversation must be an object")
        conversation_id = conversation.get("conversation_id")
        spec = expected_by_id.get(conversation_id)
        if spec is None:
            raise ValueError(f"{job_id}: unexpected conversation ID")
        extra = set(conversation) - CONVERSATION_FIELDS
        if not extra <= QUEUE_ONLY_FIELDS:
            raise ValueError(f"{job_id}: unsupported extra fields: {sorted(extra)}")
        for field in extra:
            if conversation[field] != spec[field]:
                raise ValueError(f"{job_id}/{conversation_id}: changed {field}")
        dropped.update(extra)
        messages = conversation.get("messages")
        if not isinstance(messages, list) or len(messages) != spec["message_count"]:
            raise ValueError(f"{job_id}/{conversation_id}: wrong message count")
        normalized_messages = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != MESSAGE_FIELDS:
                raise ValueError(
                    f"{job_id}/{conversation_id}: message {index} has wrong fields"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"{job_id}/{conversation_id}: message {index} is empty"
                )
            role = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != role:
                role_corrections.append(
                    {
                        "conversation_id": conversation_id,
                        "message_index": index,
                        "from": message.get("role"),
                        "to": role,
                    }
                )
            normalized_messages.append({"role": role, "content": content})
        row = {
            field: conversation[field]
            for field in conversation
            if field in CONVERSATION_FIELDS and field != "messages"
        }
        row["messages"] = normalized_messages
        normalized.append(row)
    validated = validate_conversations(normalized, expected, job_id)
    return validated, sorted(dropped), role_corrections


def salvage_attempt(
    value: Any,
    expected: list[dict[str, Any]],
    job_id: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, str]],
]:
    if not isinstance(value, list):
        return {}, {}, [{"error": f"{job_id}: candidate output must be a JSON array"}]
    expected_by_id = {row["conversation_id"]: row for row in expected}
    observed_counts: dict[str, int] = {}
    for conversation in value:
        if isinstance(conversation, dict):
            conversation_id = conversation.get("conversation_id")
            if isinstance(conversation_id, str):
                observed_counts[conversation_id] = (
                    observed_counts.get(conversation_id, 0) + 1
                )

    rows: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for conversation in value:
        if not isinstance(conversation, dict):
            errors.append({"error": f"{job_id}: conversation must be an object"})
            continue
        conversation_id = conversation.get("conversation_id")
        if conversation_id not in expected_by_id:
            errors.append(
                {"error": f"{job_id}: unexpected conversation ID {conversation_id!r}"}
            )
            continue
        if observed_counts.get(conversation_id) != 1:
            errors.append(
                {"error": f"{job_id}: duplicate conversation ID {conversation_id}"}
            )
            continue
        try:
            normalized, dropped, role_corrections = normalize_candidate(
                [conversation],
                [expected_by_id[conversation_id]],
                f"{job_id}/{conversation_id}",
            )
        except ValueError as exc:
            errors.append({"error": str(exc)})
            continue
        rows[conversation_id] = normalized[0]
        metadata[conversation_id] = {
            "dropped_queue_only_fields": dropped,
            "message_role_corrections": role_corrections,
        }
    return rows, metadata, errors


def collect_web_results(
    jobs: list[dict[str, Any]],
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    completed = []
    missing = []
    invalid = []
    conversation_count = 0
    assistant_response_count = 0
    for job in jobs:
        job_id = job["web_job_id"]
        attempts = sorted(raw_dir.glob(f"{job_id}.attempt-*.raw.txt"))
        legacy_raw = raw_dir / f"{job_id}.raw.json"
        if legacy_raw.exists():
            attempts.append(legacy_raw)
        selected = None
        errors = []
        salvaged_rows: dict[str, dict[str, Any]] = {}
        salvage_sources: dict[str, dict[str, Any]] = {}
        for attempt in attempts:
            if not attempt_is_eligible(attempt):
                errors.append(
                    {
                        "attempt": attempt.name,
                        "error": "excluded by teacher provenance",
                    }
                )
                continue
            try:
                value = json.loads(attempt.read_text(encoding="utf-8"))
                rows, dropped, role_corrections = normalize_candidate(
                    value,
                    job["expected_conversations"],
                    job_id,
                )
                selected = (attempt, rows, dropped, role_corrections)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append({"attempt": attempt.name, "error": str(exc)})
                if isinstance(exc, json.JSONDecodeError):
                    continue
            partial_rows, partial_metadata, partial_errors = salvage_attempt(
                value,
                job["expected_conversations"],
                job_id,
            )
            for conversation_id, row in partial_rows.items():
                if conversation_id in salvaged_rows:
                    continue
                salvaged_rows[conversation_id] = row
                salvage_sources[conversation_id] = {
                    "attempt": attempt,
                    **partial_metadata[conversation_id],
                }
            errors.extend(
                {"attempt": attempt.name, **error} for error in partial_errors
            )
        expected_ids = [
            row["conversation_id"] for row in job["expected_conversations"]
        ]
        salvaged_complete = all(
            conversation_id in salvaged_rows for conversation_id in expected_ids
        )
        if selected is None:
            if salvaged_complete:
                rows = [
                    salvaged_rows[conversation_id]
                    for conversation_id in expected_ids
                ]
                selected = (None, rows, [], [])
            elif attempts:
                invalid.append(
                    {
                        "web_job_id": job_id,
                        "attempts": errors,
                        "valid_conversation_ids": sorted(salvaged_rows),
                        "missing_conversation_ids": sorted(
                            set(expected_ids) - set(salvaged_rows)
                        ),
                    }
                )
            else:
                missing.append(job_id)
            if selected is None:
                continue
        attempt, rows, dropped, role_corrections = selected
        candidate_content = (
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        candidate_path = output_dir / f"{job_id}.validated.json"
        write_immutable(candidate_path, candidate_content)
        if attempt is not None:
            manifest = {
                "web_job_id": job_id,
                "source_attempt": attempt.name,
                "source_sha256": hashlib.sha256(attempt.read_bytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate_content).hexdigest(),
                "dropped_queue_only_fields": dropped,
                "message_content_changed": False,
            }
            if role_corrections:
                manifest["message_role_corrections"] = role_corrections
        else:
            source_attempts: dict[str, dict[str, Any]] = {}
            all_dropped: set[str] = set()
            all_role_corrections = []
            for conversation_id in expected_ids:
                source = salvage_sources[conversation_id]
                source_attempt = source["attempt"]
                entry = source_attempts.setdefault(
                    source_attempt.name,
                    {
                        "source_attempt": source_attempt.name,
                        "source_sha256": hashlib.sha256(
                            source_attempt.read_bytes()
                        ).hexdigest(),
                        "conversation_ids": [],
                    },
                )
                entry["conversation_ids"].append(conversation_id)
                all_dropped.update(source["dropped_queue_only_fields"])
                all_role_corrections.extend(source["message_role_corrections"])
            manifest = {
                "web_job_id": job_id,
                "source_attempts": list(source_attempts.values()),
                "candidate_sha256": hashlib.sha256(candidate_content).hexdigest(),
                "dropped_queue_only_fields": sorted(all_dropped),
                "message_content_changed": False,
                "assembled_from_valid_conversations": True,
            }
            if all_role_corrections:
                manifest["message_role_corrections"] = all_role_corrections
        write_immutable(
            output_dir / f"{job_id}.normalization.json",
            (json.dumps(manifest, indent=2) + "\n").encode(),
        )
        completed.append(job_id)
        conversation_count += len(rows)
        assistant_response_count += sum(
            message["role"] == "assistant"
            for row in rows
            for message in row["messages"]
        )
    return {
        "completed_web_jobs": len(completed),
        "conversation_count": conversation_count,
        "assistant_response_count": assistant_response_count,
        "completed_job_ids": completed,
        "missing_job_ids": missing,
        "invalid_jobs": invalid,
    }


def main() -> int:
    args = parse_args()
    jobs = read_jsonl(args.jobs)
    report = collect_web_results(jobs, args.raw_dir, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report)
    summary = {
        "completed_web_jobs": report["completed_web_jobs"],
        "conversation_count": report["conversation_count"],
        "assistant_response_count": report["assistant_response_count"],
        "missing_web_jobs": len(report["missing_job_ids"]),
        "invalid_web_jobs": len(report["invalid_jobs"]),
        "first_missing_job": next(iter(report["missing_job_ids"]), None),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

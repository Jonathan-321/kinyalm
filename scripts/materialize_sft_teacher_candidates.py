#!/usr/bin/env python3
"""Materialize schema-valid teacher responses as unreviewed SFT candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def normalize_messages(
    messages: Any, expected_turns: int, job_id: str
) -> tuple[list[dict[str, str]], list[str], list[str] | None, int]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{job_id}: messages must be a non-empty list")
    if len(messages) % 2:
        raise ValueError(
            f"{job_id}: conversation has an incomplete user/assistant pair"
        )

    normalized: list[dict[str, str]] = []
    original_roles: list[str] = []
    flags: list[str] = []
    actual_turns = len(messages) // 2
    if actual_turns != expected_turns:
        flags.append(
            f"turn-count-deviation:expected-{expected_turns}-actual-{actual_turns}"
        )
    expected_role = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{job_id}: message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        original_roles.append(str(role))
        if role != expected_role:
            flags.append("role-sequence-normalized")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{job_id}: message {index} has empty content")
        normalized.append({"role": expected_role, "content": content.strip()})
        expected_role = "assistant" if expected_role == "user" else "user"
    unique_flags = sorted(set(flags))
    return (
        normalized,
        unique_flags,
        original_roles if "role-sequence-normalized" in unique_flags else None,
        actual_turns,
    )


def materialize(
    jobs: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    response_by_id = {str(row.get("job_id", "")): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("responses contain duplicate or empty job ids")
    if len(jobs) != len(responses):
        raise ValueError(
            f"job/response count mismatch: {len(jobs)} != {len(responses)}"
        )

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content_keys: set[str] = set()
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        response = response_by_id.get(job_id)
        if response is None:
            raise ValueError(f"missing response for {job_id}")
        if response.get("error"):
            raise ValueError(f"{job_id}: response contains an error")
        try:
            generated = json.loads(str(response["output_text"]))
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"{job_id}: invalid output_text JSON") from error
        if not isinstance(generated, dict):
            raise ValueError(f"{job_id}: output_text must contain an object")

        turn_count = int(job["turn_count"])
        (
            messages,
            candidate_flags,
            raw_message_roles,
            actual_turn_count,
        ) = normalize_messages(generated.get("messages"), turn_count, job_id)
        self_check = generated.get("self_check")
        required_checks = {
            "natural_kinyarwanda",
            "factually_grounded",
            "turns_consistent",
            "safe_and_low_stakes",
        }
        if not isinstance(self_check, dict) or not required_checks.issubset(self_check):
            raise ValueError(f"{job_id}: self_check is incomplete")
        candidate_flags.extend(
            f"teacher-self-check-failed:{name}"
            for name in sorted(required_checks)
            if self_check[name] is not True
        )

        row_id = f"sft-{job_id}"
        content_key = str(job["content_key"])
        if row_id in seen_ids or content_key in seen_content_keys:
            raise ValueError(f"{job_id}: duplicate row id or content key")
        seen_ids.add(row_id)
        seen_content_keys.add(content_key)

        candidates.append(
            {
                "id": row_id,
                "content_key": content_key,
                "difficulty": job["difficulty"],
                "evaluation_categories": job["evaluation_categories"],
                "generation_profile": job["profile_id"],
                "language_mix": job["language_mix"],
                "lesson_focus": generated["lesson_focus"],
                "messages": messages,
                "prompt_version": job["prompt_version"],
                "review_status": "candidate-unreviewed",
                "reviewer_notes": (
                    "Model-generated candidate retained after strict schema and "
                    "job-lineage validation; native-speaker review remains pending."
                ),
                "skills": generated["skills"],
                "source": "synthetic-distillation",
                "source_assertion": generated["source_assertion"],
                "source_group_id": job["source_group_id"],
                "source_record_id": job_id,
                "source_status": "model-generated",
                "split": "candidate",
                "task_family": job["task_family"],
                "task_type": job["task_type"],
                "teacher_model": response["model"],
                "teacher_provider": response["provider"],
                "teacher_requested_model": response["requested_model"],
                "teacher_response_id": response["response_id"],
                "teacher_self_check": self_check,
                "requested_turn_count": turn_count,
                "turn_count": actual_turn_count,
                **(
                    {
                        "candidate_flags": sorted(set(candidate_flags)),
                        "raw_message_roles": raw_message_roles,
                    }
                    if candidate_flags
                    else {}
                ),
            }
        )
    return candidates


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(rows: list[dict[str, Any]], data_path: Path) -> dict[str, Any]:
    assistant_responses = sum(
        1
        for row in rows
        for message in row["messages"]
        if message["role"] == "assistant"
    )
    flag_counts = Counter(
        flag for row in rows for flag in row.get("candidate_flags", [])
    )
    return {
        "schema_version": 1,
        "dataset_id": "kinyalm-sft-10k-v4-candidates",
        "created_at": datetime.now(UTC).isoformat(),
        "review_status": "candidate-unreviewed",
        "conversation_count": len(rows),
        "assistant_response_count": assistant_responses,
        "multi_turn_conversation_count": sum(row["turn_count"] > 1 for row in rows),
        "flagged_conversation_count": sum("candidate_flags" in row for row in rows),
        "candidate_flag_counts": dict(sorted(flag_counts.items())),
        "task_family_counts": dict(
            sorted(Counter(row["task_family"] for row in rows).items())
        ),
        "task_type_counts": dict(
            sorted(Counter(row["task_type"] for row in rows).items())
        ),
        "data_file": data_path.name,
        "data_file_sha256": file_sha256(data_path),
        "training_note": (
            "Rows passed strict response-schema, turn-count, uniqueness, and "
            "job-lineage validation. They have not been approved by native speakers."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    try:
        rows = materialize(load_jsonl(args.jobs), load_jsonl(args.responses))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"candidate materialization failed: {error}") from error

    write_jsonl(args.out, rows)
    manifest = build_manifest(rows, args.out)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"conversations: {manifest['conversation_count']}")
    print(f"assistant responses: {manifest['assistant_response_count']}")
    print(f"multi-turn conversations: {manifest['multi_turn_conversation_count']}")
    print(f"data: {args.out}")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

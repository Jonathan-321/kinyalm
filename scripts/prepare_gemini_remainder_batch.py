#!/usr/bin/env python3
"""Create an immutable Gemini API batch containing only unresolved web jobs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-jobs", type=Path, required=True)
    parser.add_argument("--api-input", type=Path, required=True)
    parser.add_argument("--web-jobs", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--web-job-limit", type=int, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def job_number(job_id: str) -> int:
    match = re.search(r"-(\d+)$", job_id)
    if not match:
        raise ValueError(f"job ID has no numeric suffix: {job_id!r}")
    return int(match.group(1))


def index_unique(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"row is missing {field}")
        if value in indexed:
            raise ValueError(f"duplicate {field}: {value}")
        indexed[value] = row
    return indexed


def accepted_conversation_ids(validated_dir: Path) -> set[str]:
    accepted: set[str] = set()
    for path in sorted(validated_dir.glob("*.validated.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array")
        for row in rows:
            conversation_id = row.get("conversation_id")
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError(f"{path}: row is missing conversation_id")
            if conversation_id in accepted:
                raise ValueError(f"duplicate accepted conversation: {conversation_id}")
            accepted.add(conversation_id)
    return accepted


def unresolved_web_job_ids(collection_report: dict[str, Any]) -> list[str]:
    values = set(collection_report.get("missing_job_ids", []))
    for row in collection_report.get("invalid_jobs", []):
        job_id = row.get("web_job_id")
        if isinstance(job_id, str) and job_id:
            values.add(job_id)
    if not values:
        raise ValueError("collection report contains no unresolved web jobs")
    return sorted(values, key=job_number)


def select_remainder_batch(
    *,
    api_jobs: list[dict[str, Any]],
    api_input: list[dict[str, Any]],
    web_jobs: list[dict[str, Any]],
    collection_report: dict[str, Any],
    accepted_ids: set[str],
    web_job_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if web_job_limit <= 0:
        raise ValueError("web_job_limit must be positive")
    api_jobs_by_id = index_unique(api_jobs, "job_id")
    api_input_by_id = index_unique(api_input, "key")
    web_jobs_by_id = index_unique(web_jobs, "web_job_id")

    unresolved_ids = unresolved_web_job_ids(collection_report)
    if web_job_limit > len(unresolved_ids):
        raise ValueError("web_job_limit exceeds unresolved job count")
    selected_web_ids = unresolved_ids[:web_job_limit]

    source_job_ids: list[str] = []
    for web_job_id in selected_web_ids:
        web_job = web_jobs_by_id.get(web_job_id)
        if web_job is None:
            raise ValueError(f"unknown web job: {web_job_id}")
        values = web_job.get("source_job_ids")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{web_job_id}: source_job_ids must be non-empty")
        source_job_ids.extend(values)
    if len(source_job_ids) != len(set(source_job_ids)):
        raise ValueError("selected web jobs contain duplicate source jobs")

    selected_jobs: list[dict[str, Any]] = []
    selected_input: list[dict[str, Any]] = []
    selected_conversation_ids: set[str] = set()
    assistant_responses = 0
    for source_job_id in source_job_ids:
        if source_job_id not in api_jobs_by_id or source_job_id not in api_input_by_id:
            raise ValueError(f"missing API source job: {source_job_id}")
        job = api_jobs_by_id[source_job_id]
        expected = job.get("expected_conversations")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{source_job_id}: missing expected conversations")
        for conversation in expected:
            conversation_id = conversation.get("conversation_id")
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError(f"{source_job_id}: missing conversation_id")
            if conversation_id in selected_conversation_ids:
                raise ValueError(f"duplicate selected conversation: {conversation_id}")
            selected_conversation_ids.add(conversation_id)
            assistant_responses += int(conversation["message_count"]) // 2
        selected_jobs.append(job)
        selected_input.append(api_input_by_id[source_job_id])

    overlap = sorted(selected_conversation_ids & accepted_ids)
    if overlap:
        raise ValueError(
            "selected API batch overlaps accepted web conversations: "
            + ", ".join(overlap[:5])
        )
    manifest = {
        "selected_web_job_count": len(selected_web_ids),
        "selected_source_job_count": len(source_job_ids),
        "conversation_count": len(selected_conversation_ids),
        "assistant_response_count": assistant_responses,
        "accepted_conversation_overlap_count": 0,
        "selected_web_job_ids": selected_web_ids,
        "selected_source_job_ids": source_job_ids,
    }
    return selected_jobs, selected_input, manifest


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite different file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()


def main() -> int:
    args = parse_args()
    collection_report = json.loads(args.collection_report.read_text(encoding="utf-8"))
    selected_jobs, selected_input, manifest = select_remainder_batch(
        api_jobs=read_jsonl(args.api_jobs),
        api_input=read_jsonl(args.api_input),
        web_jobs=read_jsonl(args.web_jobs),
        collection_report=collection_report,
        accepted_ids=accepted_conversation_ids(args.validated_dir),
        web_job_limit=args.web_job_limit,
    )
    write_immutable(args.output_dir / "teacher-jobs.jsonl", jsonl_bytes(selected_jobs))
    write_immutable(
        args.output_dir / "gemini-batch-input.jsonl",
        jsonl_bytes(selected_input),
    )
    write_immutable(
        args.output_dir / "generation-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

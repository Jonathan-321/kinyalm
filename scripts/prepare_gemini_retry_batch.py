#!/usr/bin/env python3
"""Build an immutable low-reasoning retry queue for missing conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jobs", type=Path, required=True)
    parser.add_argument("--usable-conversations", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
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


def response_schema(conversation_count: int) -> dict[str, Any]:
    message = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": ["user", "assistant"]},
            "content": {"type": "string"},
        },
        "required": ["role", "content"],
    }
    conversation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "conversation_id": {"type": "string"},
            "batch_id": {"type": "string"},
            "task_family": {"type": "string"},
            "difficulty": {"type": "string"},
            "source_language": {"type": "string"},
            "target_language": {"type": "string"},
            "messages": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": message,
            },
        },
        "required": [
            "conversation_id",
            "batch_id",
            "task_family",
            "difficulty",
            "source_language",
            "target_language",
            "messages",
        ],
    }
    return {
        "type": "array",
        "minItems": conversation_count,
        "maxItems": conversation_count,
        "items": conversation,
    }


def load_usable_ids(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array")
    usable: set[str] = set()
    for row in rows:
        conversation_id = row.get("conversation_id") if isinstance(row, dict) else None
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"{path}: conversation is missing conversation_id")
        if conversation_id in usable:
            raise ValueError(f"{path}: duplicate conversation_id={conversation_id}")
        usable.add(conversation_id)
    return usable


def build_retry_batch(
    *,
    source_jobs: list[dict[str, Any]],
    usable_ids: set[str],
    prompt_template: str,
    generation_config: dict[str, Any],
    conversations_per_request: int = 1,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if conversations_per_request <= 0:
        raise ValueError("conversations_per_request must be positive")

    specs: list[tuple[str, dict[str, Any]]] = []
    source_ids: set[str] = set()
    all_conversation_ids: set[str] = set()
    for source_job in source_jobs:
        source_job_id = source_job.get("job_id")
        expected = source_job.get("expected_conversations")
        if not isinstance(source_job_id, str) or not source_job_id:
            raise ValueError("source job is missing job_id")
        if source_job_id in source_ids:
            raise ValueError(f"duplicate source job: {source_job_id}")
        source_ids.add(source_job_id)
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"{source_job_id}: expected_conversations must be non-empty"
            )
        for spec in expected:
            conversation_id = (
                spec.get("conversation_id") if isinstance(spec, dict) else None
            )
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError(f"{source_job_id}: missing conversation_id")
            if conversation_id in all_conversation_ids:
                raise ValueError(f"duplicate source conversation: {conversation_id}")
            all_conversation_ids.add(conversation_id)
            specs.append((source_job_id, spec))

    unknown_usable = usable_ids - all_conversation_ids
    if unknown_usable:
        raise ValueError(
            "usable set contains conversations outside the source queue: "
            + ", ".join(sorted(unknown_usable)[:5])
        )

    missing = [
        (job_id, spec)
        for job_id, spec in specs
        if spec["conversation_id"] not in usable_ids
    ]
    selected = missing[:limit] if limit is not None else missing
    teacher_jobs: list[dict[str, Any]] = []
    api_rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), conversations_per_request):
        chunk = selected[start : start + conversations_per_request]
        chunk_specs = [spec for _, spec in chunk]
        first_id = chunk_specs[0]["conversation_id"]
        retry_job_id = first_id.replace(
            "KINYA-SCALE-", "KINYA-GEMINI-RETRY-"
        )
        prompt = prompt_template.replace(
            "{{BATCH_SPECIFICATION_JSON}}",
            json.dumps(chunk_specs, ensure_ascii=False, indent=2),
        )
        teacher_jobs.append(
            {
                "job_id": retry_job_id,
                "source_job_ids": list(dict.fromkeys(row[0] for row in chunk)),
                "expected_conversations": chunk_specs,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
        request_config = dict(generation_config)
        request_config["responseJsonSchema"] = response_schema(len(chunk_specs))
        api_rows.append(
            {
                "key": retry_job_id,
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ],
                    "generationConfig": request_config,
                },
            }
        )

    manifest = {
        "source_conversation_count": len(specs),
        "already_usable_conversation_count": len(usable_ids),
        "missing_conversation_count": len(missing),
        "selected_retry_count": len(selected),
        "request_count": len(teacher_jobs),
        "remaining_after_selected_count": len(missing) - len(selected),
        "conversations_per_request": conversations_per_request,
        "generation_config": generation_config,
        "selected_conversation_ids": [
            spec["conversation_id"] for _, spec in selected
        ],
    }
    return teacher_jobs, api_rows, manifest


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
    config = json.loads(args.config.read_text(encoding="utf-8"))
    conversations_per_request = config.get("conversations_per_request")
    if not isinstance(conversations_per_request, int):
        raise ValueError("retry config is missing conversations_per_request")
    teacher_jobs, api_rows, manifest = build_retry_batch(
        source_jobs=read_jsonl(args.source_jobs),
        usable_ids=load_usable_ids(args.usable_conversations),
        prompt_template=args.prompt_template.read_text(encoding="utf-8"),
        generation_config=config["generation_config"],
        conversations_per_request=conversations_per_request,
        limit=args.limit,
    )
    manifest["model"] = config["model"]
    write_immutable(args.output_dir / "teacher-jobs.jsonl", jsonl_bytes(teacher_jobs))
    write_immutable(args.output_dir / "gemini-batch-input.jsonl", jsonl_bytes(api_rows))
    write_immutable(
        args.output_dir / "generation-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

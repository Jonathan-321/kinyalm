#!/usr/bin/env python3
"""Measure qualifying Gemini web drafts with the pinned Gemma tokenizer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-dir", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_conversations(validated_dir: Path) -> tuple[list[dict[str, Any]], int]:
    conversations: list[dict[str, Any]] = []
    files = sorted(validated_dir.glob("*.validated.json"))
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON array")
        conversations.extend(value)
    return conversations, len(files)


def measure_progress(
    conversations: list[dict[str, Any]],
    *,
    token_counter: Callable[[str], int],
    completed_web_jobs: int,
    existing_assistant_tokens: int,
    target_assistant_tokens: int,
) -> dict[str, Any]:
    conversation_ids: set[str] = set()
    response_tokens: list[int] = []
    multi_turn_conversations = 0
    for row in conversations:
        conversation_id = row.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation is missing conversation_id")
        if conversation_id in conversation_ids:
            raise ValueError(f"duplicate conversation_id: {conversation_id}")
        conversation_ids.add(conversation_id)
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{conversation_id}: messages must be a list")
        assistant_count = 0
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{conversation_id}: empty assistant response")
            response_tokens.append(token_counter(content))
            assistant_count += 1
        if assistant_count > 1:
            multi_turn_conversations += 1

    new_tokens = sum(response_tokens)
    combined_tokens = existing_assistant_tokens + new_tokens
    remaining_tokens = max(0, target_assistant_tokens - combined_tokens)
    return {
        "completed_web_jobs": completed_web_jobs,
        "conversation_count": len(conversations),
        "multi_turn_conversation_count": multi_turn_conversations,
        "assistant_response_count": len(response_tokens),
        "new_assistant_tokens": new_tokens,
        "existing_assistant_tokens": existing_assistant_tokens,
        "combined_assistant_tokens": combined_tokens,
        "target_assistant_tokens": target_assistant_tokens,
        "remaining_assistant_tokens": remaining_tokens,
        "progress_percent": round(100 * combined_tokens / target_assistant_tokens, 2),
        "assistant_tokens_min": min(response_tokens, default=0),
        "assistant_tokens_max": max(response_tokens, default=0),
        "assistant_tokens_mean": round(
            new_tokens / len(response_tokens), 2
        ) if response_tokens else 0,
    }


def main() -> int:
    args = parse_args()
    curriculum = json.loads(args.curriculum.read_text(encoding="utf-8"))
    token_budget = curriculum["token_budget"]
    conversations, completed_web_jobs = load_conversations(args.validated_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = measure_progress(
        conversations,
        token_counter=lambda text: len(
            tokenizer.encode(text, add_special_tokens=False)
        ),
        completed_web_jobs=completed_web_jobs,
        existing_assistant_tokens=token_budget["existing_assistant_tokens"],
        target_assistant_tokens=token_budget["target_total_assistant_tokens_min"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

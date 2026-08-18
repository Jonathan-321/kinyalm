#!/usr/bin/env python3
"""Score a blind review CSV with Gemini without modifying the source sheet."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCORE_FIELDS = (
    "prompt_validity",
    "kinyarwanda_correctness_1_5",
    "beginner_clarity_1_5",
    "grammar_explanation_1_5",
    "cultural_register_1_5",
    "helpfulness_1_5",
    "uncertainty_behavior_1_5",
    "hallucination_flag",
    "repetition_flag",
    "pass_fail",
    "failure_tags",
    "rewrite_priority",
    "corrected_response",
    "reviewer",
    "reviewer_notes",
)

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prompt_validity": {"type": "string", "enum": ["valid", "invalid"]},
        "kinyarwanda_correctness_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "beginner_clarity_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "grammar_explanation_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "cultural_register_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "helpfulness_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "uncertainty_behavior_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "hallucination_flag": {"type": "string", "enum": ["yes", "no"]},
        "repetition_flag": {"type": "string", "enum": ["yes", "no"]},
        "pass_fail": {"type": "string", "enum": ["pass", "fail"]},
        "failure_tags": {"type": "string"},
        "rewrite_priority": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
        "corrected_response": {"type": "string"},
        "reviewer_notes": {"type": "string"},
    },
    "required": [field for field in SCORE_FIELDS if field != "reviewer"],
}

BATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "blind_id": {"type": "string"},
                    **RESPONSE_SCHEMA["properties"],
                },
                "required": ["blind_id", *RESPONSE_SCHEMA["required"]],
            },
        }
    },
    "required": ["judgments"],
}

SYSTEM_INSTRUCTION = """You are a strict bilingual Kinyarwanda-English
evaluation judge. Evaluate only the supplied learner prompt and assistant
response. You do not know which model produced it. Judge natural Kinyarwanda,
factual and grammatical correctness, instruction following, register,
usefulness, uncertainty, hallucination, and repetition.

Use pass only when the response is materially correct, natural, relevant, and
follows the learner's request. Minor stylistic imperfections may pass. Fail
major translation or grammar errors, invented language rules, wrong register
that changes appropriateness, failure to answer, material hallucination, or
damaging repetition.

Score dimensions from 1 (unacceptable) to 5 (excellent). If a dimension is not
central to the prompt, score how responsibly the response handles it rather
than automatically lowering the score. Use corrected_response only for failed
rows; otherwise return an empty string. Keep reviewer_notes concise and
identify concrete evidence. Never infer or mention model identity."""


def resolve_api_key(keychain_service: str | None) -> str | None:
    environment_key = os.environ.get("GEMINI_API_KEY")
    if environment_key:
        return environment_key
    if not keychain_service:
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                keychain_service,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def reviewer_name(model: str) -> str:
    names = {
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro preliminary judge",
        "gemini-3-flash-preview": "Gemini 3 Flash preliminary judge",
        "gemini-2.5-pro": "Gemini 2.5 Pro preliminary judge",
    }
    return names.get(model, f"{model} preliminary judge")


class RequestRateLimiter:
    """Space direct API request starts across all worker threads."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if delay:
            time.sleep(delay)


def build_prompt(row: dict[str, str]) -> str:
    return (
        f"Task category: {row['category']}\n"
        f"Review focus: {row['review_focus']}\n\n"
        f"Learner prompt:\n{row['prompt']}\n\n"
        f"Assistant response:\n{row['response']}"
    )


def build_batch_prompt(rows: list[dict[str, str]]) -> str:
    tasks = [
        {
            "blind_id": row["blind_id"],
            "category": row["category"],
            "review_focus": row["review_focus"],
            "learner_prompt": row["prompt"],
            "assistant_response": row["response"],
        }
        for row in rows
    ]
    return (
        "Judge every item independently. Return exactly one judgment for each "
        "blind_id and do not compare items with one another.\n\n"
        + json.dumps(tasks, ensure_ascii=False)
    )


def validate_score(
    value: Any,
    *,
    reviewer: str = "Gemini 3.1 Pro preliminary judge",
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    missing = sorted(set(RESPONSE_SCHEMA["required"]).difference(value))
    if missing:
        raise ValueError("judge response missing fields: " + ", ".join(missing))
    result = {field: str(value[field]) for field in SCORE_FIELDS if field != "reviewer"}
    for field in SCORE_FIELDS[1:7]:
        rating = int(result[field])
        if not 1 <= rating <= 5:
            raise ValueError(f"{field} must be between 1 and 5")
    if result["prompt_validity"] not in {"valid", "invalid"}:
        raise ValueError("prompt_validity must be valid or invalid")
    if result["pass_fail"] not in {"pass", "fail"}:
        raise ValueError("pass_fail must be pass or fail")
    for field in ("hallucination_flag", "repetition_flag"):
        if result[field] not in {"yes", "no"}:
            raise ValueError(f"{field} must be yes or no")
    result["reviewer"] = reviewer
    return result


def load_raw_results(path: Path) -> dict[str, dict[str, str]]:
    results = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            results[str(record["blind_id"])] = validate_score(
                record["score"],
                reviewer=str(
                    record["score"].get("reviewer", reviewer_name(record["model"]))
                ),
            )
    return results


def append_raw(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def score_batch(
    rows: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    rate_limiter: RequestRateLimiter,
    attempts: int = 5,
) -> dict[str, dict[str, str]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    schema = RESPONSE_SCHEMA if len(rows) == 1 else BATCH_RESPONSE_SCHEMA
    config = types.GenerateContentConfig(
        systemInstruction=SYSTEM_INSTRUCTION,
        temperature=0,
        maxOutputTokens=max(2048, 2048 * len(rows)),
        responseMimeType="application/json",
        responseJsonSchema=schema,
        thinkingConfig=types.ThinkingConfig(
            thinkingLevel=types.ThinkingLevel.MEDIUM
        ),
    )
    for attempt in range(attempts):
        rate_limiter.wait()
        try:
            contents = (
                build_prompt(rows[0])
                if len(rows) == 1
                else build_batch_prompt(rows)
            )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            value = json.loads(response.text)
            if len(rows) == 1:
                return {
                    rows[0]["blind_id"]: validate_score(
                        value,
                        reviewer=reviewer_name(model),
                    )
                }
            judgments = value.get("judgments") if isinstance(value, dict) else None
            if not isinstance(judgments, list):
                raise ValueError("batch judge response must contain a judgments array")
            expected_ids = {row["blind_id"] for row in rows}
            result = {}
            for judgment in judgments:
                blind_id = str(judgment.get("blind_id", ""))
                if blind_id not in expected_ids:
                    raise ValueError(
                        f"unexpected blind_id in judge response: {blind_id}"
                    )
                if blind_id in result:
                    raise ValueError(
                        f"duplicate blind_id in judge response: {blind_id}"
                    )
                result[blind_id] = validate_score(
                    judgment,
                    reviewer=reviewer_name(model),
                )
            if set(result) != expected_ids:
                missing = sorted(expected_ids.difference(result))
                raise ValueError(
                    "batch judge response missing blind IDs: " + ", ".join(missing)
                )
            return result
        except Exception as exc:
            if attempt + 1 == attempts:
                raise
            delay = 35 if "429" in str(exc) else min(30, 2 ** (attempt + 1))
            time.sleep(delay)
    raise RuntimeError("unreachable")


def write_scored_csv(
    source_rows: list[dict[str, str]],
    fieldnames: list[str],
    scores: dict[str, dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_row in source_rows:
            row = dict(source_row)
            row.update(scores.get(row["blind_id"], {}))
            writer.writerow(row)
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--requests-per-minute", type=int, default=22)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--acknowledge-paid-direct", action="store_true")
    parser.add_argument("--keychain-service", default="kinyalm-gemini-api-key")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.acknowledge_paid_direct:
        raise SystemExit("paid direct API acknowledgement is required")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.requests_per_minute < 1:
        raise SystemExit("--requests-per-minute must be positive")
    api_key = resolve_api_key(args.keychain_service)
    if not api_key:
        raise SystemExit("Gemini API key is unavailable")

    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        source_rows = list(reader)
    missing_fields = sorted(set(SCORE_FIELDS).difference(fieldnames))
    if missing_fields:
        raise SystemExit("review CSV is missing fields: " + ", ".join(missing_fields))

    raw_path = args.output_dir / "raw-judge-results.jsonl"
    scores = load_raw_results(raw_path)
    pending = [row for row in source_rows if row["blind_id"] not in scores]
    if args.limit is not None:
        pending = pending[: args.limit]
    batches = [
        pending[index : index + args.batch_size]
        for index in range(0, len(pending), args.batch_size)
    ]
    failures = 0
    rate_limiter = RequestRateLimiter(args.requests_per_minute)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                score_batch,
                batch,
                api_key=api_key,
                model=args.model,
                rate_limiter=rate_limiter,
            ): batch
            for batch in batches
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            batch = futures[future]
            try:
                batch_scores = future.result()
                for blind_id, score in batch_scores.items():
                    scores[blind_id] = score
                    append_raw(
                        raw_path,
                        {
                            "blind_id": blind_id,
                            "status": "ok",
                            "model": args.model,
                            "score": score,
                        },
                    )
            except Exception as exc:
                failures += len(batch)
                for row in batch:
                    append_raw(
                        raw_path,
                        {
                            "blind_id": row["blind_id"],
                            "status": "error",
                            "model": args.model,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
            print(
                f"scored={len(scores)}/{len(source_rows)} "
                f"batch={completed}/{len(batches)} failures={failures}",
                flush=True,
            )

    write_scored_csv(
        source_rows,
        fieldnames,
        scores,
        args.output_dir / "model-assisted-review.csv",
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build deterministic eight-conversation jobs for Gemini Pro web chat."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.prepare_gemini_batch_jobs import (
        build_jobs,
        write_immutable,
        write_jsonl,
    )
except ModuleNotFoundError:
    from prepare_gemini_batch_jobs import build_jobs, write_immutable, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/gemini_longform_batch_v1.json"),
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("prompts/gemini_longform_web_batch_v1.txt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-jobs-per-web-job", type=int, default=2)
    return parser.parse_args()


def build_web_jobs(
    config: dict,
    curriculum: dict,
    source_prompt: str,
    web_prompt: str,
    *,
    source_jobs_per_web_job: int,
) -> list[dict]:
    source_jobs = build_jobs(config, curriculum, source_prompt)
    if source_jobs_per_web_job < 1:
        raise ValueError("source_jobs_per_web_job must be positive")
    if len(source_jobs) % source_jobs_per_web_job:
        raise ValueError("source jobs must divide evenly into web jobs")

    jobs = []
    for start in range(0, len(source_jobs), source_jobs_per_web_job):
        number = start // source_jobs_per_web_job + 1
        group = source_jobs[start : start + source_jobs_per_web_job]
        expected = [
            conversation
            for source_job in group
            for conversation in source_job["expected_conversations"]
        ]
        prompt = web_prompt.replace(
            "{{BATCH_SPECIFICATION_JSON}}",
            json.dumps(expected, ensure_ascii=False, indent=2),
        )
        jobs.append(
            {
                "web_job_id": f"KINYA-GEMINI-WEB-JOB-{number:04d}",
                "source_job_ids": [job["job_id"] for job in group],
                "expected_conversations": expected,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    return jobs


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    curriculum = json.loads(Path(config["curriculum"]).read_text(encoding="utf-8"))
    source_prompt = Path(config["prompt_template"]).read_text(encoding="utf-8")
    web_prompt = args.prompt_template.read_text(encoding="utf-8")
    jobs = build_web_jobs(
        config,
        curriculum,
        source_prompt,
        web_prompt,
        source_jobs_per_web_job=args.source_jobs_per_web_job,
    )
    write_jsonl(args.output_dir / "web-jobs.jsonl", jobs)
    assistant_responses = sum(
        spec["message_count"] // 2
        for job in jobs
        for spec in job["expected_conversations"]
    )
    manifest = {
        "generation_id": "kinyalm-gemini-web-longform-1m-v1",
        "teacher": "Gemini Pro web chat subscription",
        "paid_api_used": False,
        "web_job_count": len(jobs),
        "conversation_count": sum(
            len(job["expected_conversations"]) for job in jobs
        ),
        "assistant_response_count": assistant_responses,
        "source_job_count": sum(len(job["source_job_ids"]) for job in jobs),
        "task_family_counts": dict(
            sorted(
                Counter(
                    spec["task_family"]
                    for job in jobs
                    for spec in job["expected_conversations"]
                ).items()
            )
        ),
        "immutable_raw_outputs": True,
        "teacher_self_repair_allowed": False,
    }
    write_immutable(
        args.output_dir / "generation-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

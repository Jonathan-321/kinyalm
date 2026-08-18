#!/usr/bin/env python3
"""Build deterministic Gemini Batch API jobs for the 1M-token curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

DOMAINS = (
    "school and study",
    "family routines",
    "workplace communication",
    "shopping at a market",
    "public transport",
    "agriculture",
    "technology",
    "community events",
    "sports and recreation",
    "food and cooking",
    "travel in Rwanda",
    "weather and daily plans",
    "literature and storytelling",
    "environmental care",
    "housing and neighbors",
    "arts and music",
    "customer service",
    "time and scheduling",
    "friendship",
    "formal correspondence",
)
CONTEXTS = (
    "a beginner asking for a concrete example",
    "an intermediate learner correcting a misunderstanding",
    "an advanced learner comparing two expressions",
    "a visitor preparing for an everyday interaction",
    "a student practicing for class",
    "a colleague choosing an appropriate register",
    "a learner following up on an earlier explanation",
    "a speaker resolving an ambiguous word",
    "a reader grounding answers in a short passage",
    "a bilingual speaker reducing unnecessary code-switching",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/gemini_longform_batch_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def expanded_counts(counts: dict[str, int]) -> list[str]:
    return [name for name, count in counts.items() for _ in range(count)]


def ratio_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {name: total * value for name, value in ratios.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        exact,
        key=lambda name: (exact[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def language_pair(family: str, language_mix: str) -> tuple[str, str]:
    if family == "en-to-rw":
        return "en", "rw"
    if family == "rw-to-en":
        return "rw", "en"
    if family == "code-switching":
        return "rw+en", "rw+en"
    if language_mix == "kinyarwanda":
        return "rw", "rw"
    if language_mix == "english":
        return "en", "rw+en"
    return "rw+en", "rw+en"


def response_schema(conversations_per_job: int) -> dict:
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
        "minItems": conversations_per_job,
        "maxItems": conversations_per_job,
        "items": conversation,
    }


def build_specs(curriculum: dict, *, seed: int, first_number: int) -> list[dict]:
    total = curriculum["token_budget"]["planned_new_conversations"]
    families = expanded_counts(curriculum["task_family_targets"])
    shapes = expanded_counts(curriculum["conversation_shape"])
    language_counts = ratio_counts(total, curriculum["language_mix_targets"])
    languages = expanded_counts(language_counts)
    difficulties = expanded_counts(
        ratio_counts(total, {"beginner": 0.3, "intermediate": 0.45, "advanced": 0.25})
    )
    distributions = (families, shapes, languages, difficulties)
    if not all(len(values) == total for values in distributions):
        raise ValueError("curriculum distributions do not match planned conversations")

    rng = random.Random(seed)
    for values in (families, shapes, languages, difficulties):
        rng.shuffle(values)

    specs: list[dict] = []
    for offset in range(total):
        family = families[offset]
        source, target = language_pair(family, languages[offset])
        message_count = int(shapes[offset].split("_")[0])
        specs.append(
            {
                "conversation_id": f"KINYA-SCALE-{first_number + offset:06d}",
                "task_family": family,
                "difficulty": difficulties[offset],
                "source_language": source,
                "target_language": target,
                "message_count": message_count,
                "scenario_seed": (
                    f"{DOMAINS[offset % len(DOMAINS)]}; "
                    f"{CONTEXTS[(offset // len(DOMAINS)) % len(CONTEXTS)]}"
                ),
            }
        )
    return specs


def build_jobs(config: dict, curriculum: dict, prompt_template: str) -> list[dict]:
    per_job = curriculum["batch_policy"]["conversations_per_generation_batch"]
    specs = build_specs(
        curriculum,
        seed=config["seed"],
        first_number=config["first_conversation_number"],
    )
    if len(specs) % per_job:
        raise ValueError("planned conversations must divide evenly across jobs")

    jobs: list[dict] = []
    for start in range(0, len(specs), per_job):
        job_number = start // per_job + 1
        job_id = f"KINYA-GEMINI-JOB-{job_number:04d}"
        batch_id = f"GEMINI-SCALE-BATCH-{job_number:04d}"
        batch_specs = []
        for spec in specs[start : start + per_job]:
            spec = dict(spec)
            spec["batch_id"] = batch_id
            batch_specs.append(spec)
        prompt = prompt_template.replace(
            "{{BATCH_SPECIFICATION_JSON}}",
            json.dumps(batch_specs, ensure_ascii=False, indent=2),
        )
        jobs.append(
            {
                "job_id": job_id,
                "expected_conversations": batch_specs,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    return jobs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode()
    write_immutable(path, content)


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite different queue file: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    curriculum_path = Path(config["curriculum"])
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    prompt_template = Path(config["prompt_template"]).read_text(encoding="utf-8")
    jobs = build_jobs(config, curriculum, prompt_template)
    per_job = curriculum["batch_policy"]["conversations_per_generation_batch"]
    schema = response_schema(per_job)

    api_rows = []
    for job in jobs:
        generation_config = dict(config["generation_config"])
        generation_config["responseJsonSchema"] = schema
        api_rows.append(
            {
                "key": job["job_id"],
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": job["prompt"]}],
                        }
                    ],
                    "generationConfig": generation_config,
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "teacher-jobs.jsonl", jobs)
    write_jsonl(args.output_dir / "gemini-batch-input.jsonl", api_rows)
    manifest = {
        "generation_id": config["generation_id"],
        "model": config["model"],
        "job_count": len(jobs),
        "conversation_count": sum(
            len(job["expected_conversations"]) for job in jobs
        ),
        "assistant_response_count": sum(
            spec["message_count"] // 2
            for job in jobs
            for spec in job["expected_conversations"]
        ),
        "task_family_counts": dict(
            sorted(
                Counter(
                    spec["task_family"]
                    for job in jobs
                    for spec in job["expected_conversations"]
                ).items()
            )
        ),
        "message_shape_counts": dict(
            sorted(
                Counter(
                    f"{spec['message_count']}_messages"
                    for job in jobs
                    for spec in job["expected_conversations"]
                ).items()
            )
        ),
        "immutable_raw_outputs": True,
        "teacher_self_repair_allowed": False,
        "submitted": False,
    }
    write_immutable(
        args.output_dir / "generation-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

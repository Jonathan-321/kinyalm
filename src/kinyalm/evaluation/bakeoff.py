"""Configuration and blind-review helpers for base-model bake-offs."""

from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinyalm.evaluation.task_bank import TutorTask

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REVIEW_COLUMNS = (
    "blind_id",
    "model_label",
    "task_id",
    "category",
    "prompt",
    "review_focus",
    "response",
    "kinyarwanda_correctness_1_5",
    "beginner_clarity_1_5",
    "grammar_explanation_1_5",
    "cultural_register_1_5",
    "helpfulness_1_5",
    "uncertainty_behavior_1_5",
    "hallucination_flag",
    "pass_fail",
    "reviewer",
    "reviewer_notes",
)


@dataclass(frozen=True)
class CandidateSpec:
    """One unchanged model candidate in a bake-off."""

    id: str
    model_id: str
    revision: str


@dataclass(frozen=True)
class BakeoffConfig:
    """Pinned settings shared by every candidate in one bake-off."""

    schema_version: int
    run_name: str
    task_bank: str
    task_split: str
    expected_task_count: int
    system_prompt: str
    seed: int
    max_new_tokens: int
    enable_thinking: bool
    candidates: tuple[CandidateSpec, ...]


def load_bakeoff_config(path: str | Path) -> BakeoffConfig:
    """Load and validate a pinned bake-off JSON configuration."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("bake-off config must be a JSON object")

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError("bake-off config candidates must be a list")
    candidates = tuple(
        _candidate(item, index) for index, item in enumerate(candidates_raw)
    )
    config = BakeoffConfig(
        schema_version=_integer(raw, "schema_version"),
        run_name=_string(raw, "run_name"),
        task_bank=_string(raw, "task_bank"),
        task_split=_string(raw, "task_split"),
        expected_task_count=_integer(raw, "expected_task_count"),
        system_prompt=_string(raw, "system_prompt"),
        seed=_integer(raw, "seed"),
        max_new_tokens=_integer(raw, "max_new_tokens"),
        enable_thinking=_boolean(raw, "enable_thinking"),
        candidates=candidates,
    )
    _validate_config(config)
    return config


def latest_results(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return the latest JSONL record for each task ID."""

    result_path = Path(path)
    if not result_path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        result_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{result_path}:{line_number} must be a JSON object")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{result_path}:{line_number} has no task_id")
        latest[task_id] = record
    return latest


def append_result(path: str | Path, record: dict[str, Any]) -> None:
    """Append one durable UTF-8 result row."""

    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def write_blind_review_pack(
    *,
    output_csv: str | Path,
    key_path: str | Path,
    tasks: Iterable[TutorTask],
    candidate_results: dict[str, dict[str, dict[str, Any]]],
    seed: int,
) -> tuple[int, dict[str, str]]:
    """Write a reviewer CSV without model IDs and a separate private key."""

    task_list = list(tasks)
    candidate_ids = sorted(candidate_results)
    if len(candidate_ids) < 2:
        raise ValueError("blind review requires results from at least two candidates")

    labels = [f"Model {chr(ord('A') + index)}" for index in range(len(candidate_ids))]
    rng = random.Random(seed)
    rng.shuffle(labels)
    label_by_candidate = dict(zip(candidate_ids, labels, strict=True))

    review_rows: list[dict[str, str]] = []
    blind_key_rows: list[dict[str, str]] = []
    blind_index = 1
    for task in task_list:
        task_candidates = candidate_ids.copy()
        rng.shuffle(task_candidates)
        for candidate_id in task_candidates:
            result = candidate_results[candidate_id].get(task.id)
            if result is None:
                raise ValueError(f"{candidate_id} is missing result for {task.id}")
            blind_id = f"B{blind_index:03d}"
            blind_index += 1
            response = _review_response(result)
            review_rows.append(
                {
                    "blind_id": blind_id,
                    "model_label": label_by_candidate[candidate_id],
                    "task_id": task.id,
                    "category": task.category,
                    "prompt": task.prompt,
                    "review_focus": task.review_focus,
                    "response": response,
                    **{column: "" for column in REVIEW_COLUMNS[7:]},
                }
            )
            blind_key_rows.append(
                {
                    "blind_id": blind_id,
                    "model_label": label_by_candidate[candidate_id],
                    "candidate_id": candidate_id,
                    "model_id": str(result.get("model_id", "")),
                    "model_revision": str(result.get("model_revision", "")),
                }
            )

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)

    private_key_path = Path(key_path)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_text(
        json.dumps(
            {
                "warning": "Do not share this key with reviewers before scoring.",
                "labels": label_by_candidate,
                "rows": blind_key_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(review_rows), label_by_candidate


def _candidate(raw: Any, index: int) -> CandidateSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"candidate {index} must be a JSON object")
    return CandidateSpec(
        id=_string(raw, "id", label=f"candidate {index}"),
        model_id=_string(raw, "model_id", label=f"candidate {index}"),
        revision=_string(raw, "revision", label=f"candidate {index}"),
    )


def _validate_config(config: BakeoffConfig) -> None:
    if config.schema_version != 1:
        raise ValueError("unsupported bake-off schema_version")
    if config.task_split != "benchmark-only":
        raise ValueError("bake-off task_split must be benchmark-only")
    if config.expected_task_count < 1:
        raise ValueError("expected_task_count must be positive")
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if len(config.candidates) < 2:
        raise ValueError("bake-off requires at least two candidates")

    ids = [candidate.id for candidate in config.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    for candidate in config.candidates:
        if not REVISION_PATTERN.fullmatch(candidate.revision):
            raise ValueError(f"{candidate.id}: revision must be a 40-character SHA")


def _review_response(result: dict[str, Any]) -> str:
    if result.get("status") == "ok":
        return str(result.get("response", ""))
    error = str(result.get("error", "unknown generation error"))
    return f"[GENERATION ERROR] {error}"


def _string(raw: dict[str, Any], key: str, *, label: str = "config") -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {key} must be a non-empty string")
    return value.strip()


def _integer(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config {key} must be an integer")
    return value


def _boolean(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"config {key} must be a boolean")
    return value

"""Build explicitly experimental SFT baselines from the HF data lake."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from kinyalm.data.sft import validate_sft_records
from kinyalm.data.splits import assign_grouped_splits, connected_record_groups

DEFAULT_REPO_ID = "kinyalm/kinyalm-data-lake"
DEFAULT_BATCH = "sft-distillation-production-1000-v3-final"
BASELINE_MODES = {"critic-accepted", "critic-accepted-and-repaired"}


def materialize_hf_baseline(
    *,
    repo_id: str,
    revision: str,
    output_dir: str | Path,
    mode: str,
    train_ratio: float,
    split_seed: str,
    batch: str = DEFAULT_BATCH,
) -> dict[str, Any]:
    """Download a pinned HF revision and write experimental SFT splits."""

    api = HfApi()
    resolved_revision = api.dataset_info(repo_id, revision=revision).sha
    filenames = _batch_filenames(batch)
    local_files = {
        name: Path(
            hf_hub_download(
                repo_id,
                filename,
                repo_type="dataset",
                revision=resolved_revision,
            )
        )
        for name, filename in filenames.items()
    }
    drafts = _load_jsonl(local_files["drafts"])
    critic_jobs = _load_jsonl(local_files["critic_jobs"])
    critic_responses = _load_jsonl(local_files["critic_responses"])
    source_summary = json.loads(
        local_files["source_summary"].read_text(encoding="utf-8")
    )

    records, build_report = build_baseline_records(
        drafts,
        critic_jobs,
        critic_responses,
        mode=mode,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.jsonl"
    validation_path = output_path / "validation.jsonl"
    manifest_path = output_path / "dataset-manifest.json"
    train_rows = [row for row in records if row["split"] == "experimental-train"]
    validation_rows = [
        row for row in records if row["split"] == "experimental-validation"
    ]
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_tier": "experimental-critic-filtered",
        "production_eligible": False,
        "human_reviewed": False,
        "intended_use": (
            "Baseline fine-tuning and infrastructure evaluation only; replace "
            "with fluent-human-approved rows for the curated model run."
        ),
        "source": {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "batch": batch,
            "source_policy_can_train": source_summary.get("can_train"),
            "source_policy_status": source_summary.get("status"),
            "files": {
                name: {
                    "path": filenames[name],
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in local_files.items()
            },
        },
        "build": {
            "mode": mode,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            **build_report,
        },
        "outputs": {
            "train": _output_metadata(train_path, train_rows),
            "validation": _output_metadata(validation_path, validation_rows),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_baseline_records(
    drafts: list[dict[str, Any]],
    critic_jobs: list[dict[str, Any]],
    critic_responses: list[dict[str, Any]],
    *,
    mode: str,
    train_ratio: float,
    split_seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join draft and critic records into a clearly labeled experiment tier."""

    if mode not in BASELINE_MODES:
        choices = ", ".join(sorted(BASELINE_MODES))
        raise ValueError(f"mode must be one of: {choices}")

    draft_by_id = _unique_by(drafts, "id", "draft")
    job_by_id = _unique_by(critic_jobs, "job_id", "critic job")
    response_by_job = _unique_by(critic_responses, "job_id", "critic response")
    if set(job_by_id) != set(response_by_job):
        missing = sorted(set(job_by_id).symmetric_difference(response_by_job))
        raise ValueError(f"critic job/response mismatch: {', '.join(missing[:5])}")

    assessments: dict[str, dict[str, Any]] = {}
    for job_id, job in job_by_id.items():
        source_id = job.get("source_record_id")
        if source_id not in draft_by_id:
            raise ValueError(f"critic job {job_id} references unknown row {source_id}")
        response = response_by_job[job_id]
        if response.get("error"):
            raise ValueError(f"critic response {job_id} has error: {response['error']}")
        try:
            assessment = json.loads(response.get("output_text", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"critic response {job_id} is not valid JSON") from exc
        _validate_assessment(job_id, assessment)
        assessments[source_id] = {
            **assessment,
            "critic_model": response.get("model"),
            "critic_response_id": response.get("response_id"),
        }

    recommendation_counts = Counter()
    selected: list[dict[str, Any]] = []
    dropped_exact_duplicates: list[dict[str, str]] = []
    seen_conversations: dict[str, str] = {}
    for draft in drafts:
        row_id = draft.get("id")
        assessment = assessments.get(row_id)
        if not assessment:
            raise ValueError(f"draft row {row_id} has no critic assessment")
        recommendation = assessment["recommendation"]
        recommendation_counts[recommendation] += 1
        if recommendation == "reject":
            continue
        if recommendation == "repair" and mode == "critic-accepted":
            continue

        record = json.loads(json.dumps(draft, ensure_ascii=False))
        if recommendation == "repair":
            original_messages = _canonical_messages(record["messages"])
            record["messages"] = assessment["revised_messages"]
            record["review_status"] = "critic-repaired"
            record["original_messages_sha256"] = sha256(
                original_messages.encode("utf-8")
            ).hexdigest()
        else:
            record["review_status"] = "critic-accepted"

        record["source_status"] = "model-generated"
        record["training_tier"] = "experimental-critic-filtered"
        record["critic_assessment"] = {
            "model": assessment.get("critic_model"),
            "response_id": assessment.get("critic_response_id"),
            "recommendation": recommendation,
            "scores": assessment["scores"],
            "failure_tags": assessment["failure_tags"],
            "brief_notes": assessment["brief_notes"],
        }
        record["reviewer_notes"] = (
            "Independent model critic only; fluent-human review is still "
            f"required. Critic recommendation={recommendation}."
        )

        conversation_hash = sha256(
            _canonical_messages(record["messages"]).encode("utf-8")
        ).hexdigest()
        duplicate_of = seen_conversations.get(conversation_hash)
        if duplicate_of:
            dropped_exact_duplicates.append(
                {"id": str(row_id), "duplicate_of": duplicate_of}
            )
            continue
        seen_conversations[conversation_hash] = str(row_id)
        selected.append(record)

    assign_grouped_splits(
        selected,
        train_ratio=train_ratio,
        seed=split_seed,
        train_split="experimental-train",
        validation_split="experimental-validation",
    )
    failures = [result for result in validate_sft_records(selected) if not result.ok]
    if failures:
        preview = "; ".join(
            f"row {result.line_number}: {', '.join(result.errors)}"
            for result in failures[:10]
        )
        raise ValueError(f"experimental SFT validation failed: {preview}")

    split_counts = Counter(record["split"] for record in selected)
    family_counts = Counter(record.get("task_family", "unknown") for record in selected)
    return selected, {
        "draft_rows": len(drafts),
        "critic_recommendation_counts": dict(sorted(recommendation_counts.items())),
        "selected_rows": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "task_family_counts": dict(sorted(family_counts.items())),
        "connected_groups": len(connected_record_groups(selected)),
        "dropped_exact_duplicate_rows": dropped_exact_duplicates,
    }


def _batch_filenames(batch: str) -> dict[str, str]:
    base = "sft-distillation-production-1000-v3-final"
    if batch != base:
        raise ValueError(f"unsupported batch: {batch}")
    return {
        "drafts": f"data/drafts/{base}/distillation-drafts.jsonl",
        "critic_jobs": f"generation/{base}/critic-jobs.jsonl",
        "critic_responses": f"generation/{base}/critic-responses.jsonl",
        "source_summary": f"quality/generation/{base}/production-review-summary.json",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number}: expected an object")
            rows.append(row)
    return rows


def _unique_by(
    rows: list[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} is missing {key}")
        if value in result:
            raise ValueError(f"duplicate {label} {key}: {value}")
        result[value] = row
    return result


def _validate_assessment(job_id: str, assessment: dict[str, Any]) -> None:
    recommendation = assessment.get("recommendation")
    if recommendation not in {"accept", "repair", "reject"}:
        raise ValueError(f"critic response {job_id} has invalid recommendation")
    scores = assessment.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"critic response {job_id} has no scores")
    failure_tags = assessment.get("failure_tags")
    if not isinstance(failure_tags, list):
        raise ValueError(f"critic response {job_id} has invalid failure_tags")
    if not isinstance(assessment.get("brief_notes"), str):
        raise ValueError(f"critic response {job_id} has invalid brief_notes")
    if recommendation == "accept":
        if failure_tags or any(
            not isinstance(value, int) or value < 4 for value in scores.values()
        ):
            raise ValueError(f"critic response {job_id} has an invalid accept gate")
    if recommendation == "repair":
        revised_messages = assessment.get("revised_messages")
        if not isinstance(revised_messages, list) or len(revised_messages) < 2:
            raise ValueError(f"critic response {job_id} has no usable repair")


def _canonical_messages(messages: Any) -> str:
    return json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_metadata(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }

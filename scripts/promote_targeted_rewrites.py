#!/usr/bin/env python3
"""Promote 500-1,000 approved native rewrites into leak-resistant SFT data."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kinyalm.data.sft import TASK_TYPES, validate_sft_records
from kinyalm.data.splits import assign_grouped_splits, normalize_text
from kinyalm.evaluation import load_task_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-csv", type=Path, required=True)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-approved", type=int, default=500)
    parser.add_argument("--maximum-approved", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", default="native-recovery-rewrites-v1")
    return parser.parse_args()


def promote_rewrites(
    review_rows: list[dict[str, str]],
    held_out_prompts: list[str],
    *,
    minimum_approved: int,
    maximum_approved: int,
    train_ratio: float,
    split_seed: str,
) -> list[dict]:
    if minimum_approved < 1 or maximum_approved < minimum_approved:
        raise ValueError("approved row bounds are invalid")
    approved = [
        row
        for row in review_rows
        if row.get("review_status", "").strip().casefold() == "approved"
    ]
    if not minimum_approved <= len(approved) <= maximum_approved:
        raise ValueError(
            f"approved row count must be {minimum_approved}-{maximum_approved}; "
            f"found {len(approved)}"
        )

    held_out_normalized = [normalize_text(prompt) for prompt in held_out_prompts]
    seen_ids = set()
    seen_prompts = set()
    records = []
    for row in approved:
        rewrite_id = row.get("rewrite_id", "").strip()
        prompt = row.get("new_user_prompt", "").strip()
        response = row.get("gold_assistant_response", "").strip()
        reviewer = row.get("reviewer", "").strip()
        task_type = row.get("task_type", "").strip()
        language_mix = row.get("language_mix", "").strip()
        source_task_id = row.get("source_task_id", "").strip()
        if not rewrite_id or rewrite_id in seen_ids:
            raise ValueError(f"missing or duplicate rewrite_id: {rewrite_id!r}")
        seen_ids.add(rewrite_id)
        if not prompt or not response or not reviewer or not source_task_id:
            raise ValueError(
                f"{rewrite_id}: prompt, response, reviewer, and source task "
                "are required"
            )
        if task_type not in TASK_TYPES:
            raise ValueError(f"{rewrite_id}: unsupported task_type {task_type!r}")
        if language_mix not in {"kinyarwanda", "english", "kinyarwanda+english"}:
            raise ValueError(f"{rewrite_id}: unsupported language_mix {language_mix!r}")
        normalized_prompt = normalize_text(prompt)
        if normalized_prompt in seen_prompts:
            raise ValueError(f"{rewrite_id}: duplicate rewritten prompt")
        seen_prompts.add(normalized_prompt)
        closest = max(
            (
                difflib.SequenceMatcher(None, normalized_prompt, held_out).ratio()
                for held_out in held_out_normalized
            ),
            default=0.0,
        )
        if closest >= 0.88:
            raise ValueError(
                f"{rewrite_id}: rewritten prompt is too similar to a held-out prompt "
                f"({closest:.3f})"
            )
        records.append(
            {
                "id": rewrite_id,
                "task_type": task_type,
                "split": "train",
                "source": "native-recovery-rewrite-v1",
                "source_status": "team-authored",
                "review_status": "approved",
                "language_mix": language_mix,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "reviewer_notes": (
                    f"Reviewer: {reviewer}. "
                    f"{row.get('reviewer_notes', '').strip()}"
                ).strip(),
                "source_group_id": source_task_id,
                "source_record_id": rewrite_id,
                "training_tier": "native-model-failure-rewrite",
                "failure_tags": row.get("failure_tags", "").strip(),
            }
        )

    assign_grouped_splits(
        records,
        train_ratio=train_ratio,
        seed=split_seed,
        train_split="train",
        validation_split="validation",
    )
    failures = [result for result in validate_sft_records(records) if not result.ok]
    if failures:
        details = "; ".join(
            f"row {result.line_number}: {', '.join(result.errors)}"
            for result in failures[:5]
        )
        raise ValueError(f"promoted SFT validation failed: {details}")
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    with args.rewrite_csv.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    held_out_prompts = [task.prompt for task in load_task_bank(args.task_bank)]
    try:
        records = promote_rewrites(
            review_rows,
            held_out_prompts,
            minimum_approved=args.minimum_approved,
            maximum_approved=args.maximum_approved,
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
        )
    except ValueError as exc:
        raise SystemExit(f"promotion failed: {exc}") from exc

    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": "native-recovery-rewrites-v1",
        "dataset_tier": "human-reviewed-recovery-sft",
        "human_reviewed": True,
        "training_eligible": True,
        "production_eligible": False,
        "source_review_csv_sha256": _sha256(args.rewrite_csv),
        "held_out_task_bank_sha256": _sha256(args.task_bank),
        "build": {
            "approved_rows": len(records),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_ratio": args.train_ratio,
            "split_seed": args.split_seed,
            "leakage_similarity_limit": 0.88,
        },
        "outputs": {
            "train": {
                "path": train_path.name,
                "rows": len(train),
                "sha256": _sha256(train_path),
            },
            "validation": {
                "path": validation_path.name,
                "rows": len(validation),
                "sha256": _sha256(validation_path),
            },
        },
    }
    manifest_path = args.output_dir / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Promoted {len(records)} native rewrites: "
        f"{len(train)} train, {len(validation)} validation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

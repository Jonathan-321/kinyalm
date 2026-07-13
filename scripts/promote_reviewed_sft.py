"""Promote reviewed SFT draft rows into train/validation JSONL files.

The script joins a draft JSONL file with a reviewer TSV. Only rows marked
review_status=approved in the TSV are exported. Everything else is excluded.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.sft import REVIEW_STATUSES, load_jsonl, validate_sft_records


SCORE_FIELDS = ("correctness_1_5", "naturalness_1_5", "helpfulness_1_5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-jsonl", required=True)
    parser.add_argument("--review-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")

    draft_records = load_jsonl(args.draft_jsonl)
    try:
        review_rows = load_review_rows(args.review_tsv)
        approved = promote_approved_rows(
            draft_records,
            review_rows,
            train_ratio=args.train_ratio,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not approved:
        raise SystemExit("no approved rows found in review TSV")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    validation_path = out_dir / "validation.jsonl"
    summary_path = out_dir / "promotion-summary.md"

    train_rows = [row for row in approved if row["split"] == "train"]
    validation_rows = [row for row in approved if row["split"] == "validation"]
    validate_or_exit(train_rows + validation_rows)
    write_jsonl(train_rows, train_path)
    write_jsonl(validation_rows, validation_path)
    write_summary(
        summary_path,
        draft_count=len(draft_records),
        review_count=len(review_rows),
        train_count=len(train_rows),
        validation_count=len(validation_rows),
    )
    print(f"approved rows: {len(approved)}")
    print(f"train: {train_path}")
    print(f"validation: {validation_path}")
    print(f"summary: {summary_path}")
    return 0


def load_review_rows(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).expanduser().open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = {}
        for row in reader:
            row_id = row.get("id", "").strip()
            if row_id:
                if row_id in rows:
                    raise ValueError(f"duplicate review id: {row_id}")
                status = row.get("review_status", "").strip()
                if status not in REVIEW_STATUSES:
                    allowed = ", ".join(sorted(REVIEW_STATUSES))
                    raise ValueError(
                        f"review row {row_id} has invalid review_status "
                        f"{status!r}; expected one of: {allowed}"
                    )
                rows[row_id] = row
        return rows


def promote_approved_rows(
    draft_records: list[dict],
    review_rows: dict[str, dict[str, str]],
    *,
    train_ratio: float,
) -> list[dict]:
    approved: list[dict] = []
    draft_ids = {record.get("id") for record in draft_records}
    unknown_review_ids = sorted(set(review_rows).difference(draft_ids))
    if unknown_review_ids:
        preview = ", ".join(unknown_review_ids[:5])
        raise ValueError(f"review TSV has ids not found in draft JSONL: {preview}")

    for record in draft_records:
        if record.get("split") != "draft":
            raise ValueError(
                f"draft input contains non-draft row {record.get('id')} "
                f"with split={record.get('split')}"
            )
        row = review_rows.get(record["id"])
        if not row:
            continue
        if row.get("review_status", "").strip() != "approved":
            continue
        validate_approved_review_row(row)
        promoted = json.loads(json.dumps(record, ensure_ascii=False))
        promoted["review_status"] = "approved"
        promoted["reviewer_notes"] = build_reviewer_notes(row)
        approved.append(promoted)

    train_cutoff = int(len(approved) * train_ratio)
    for index, record in enumerate(approved):
        record["split"] = "train" if index < train_cutoff else "validation"
    return approved


def validate_approved_review_row(row: dict[str, str]) -> None:
    row_id = row.get("id", "").strip()
    errors: list[str] = []

    if not row.get("reviewer", "").strip():
        errors.append("reviewer is required")
    if row.get("failure_tags", "").strip():
        errors.append("approved rows must not have failure_tags")

    for field in SCORE_FIELDS:
        value = row.get(field, "").strip()
        try:
            score = int(value)
        except ValueError:
            errors.append(f"{field} must be an integer from 4 to 5")
            continue
        if score < 4 or score > 5:
            errors.append(f"{field} must be 4 or 5 for approved rows")

    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"approved review row {row_id} is invalid: {joined}")


def build_reviewer_notes(row: dict[str, str]) -> str:
    fields = [
        ("correctness", row.get("correctness_1_5", "")),
        ("naturalness", row.get("naturalness_1_5", "")),
        ("helpfulness", row.get("helpfulness_1_5", "")),
        ("failure_tags", row.get("failure_tags", "")),
        ("reviewer", row.get("reviewer", "")),
        ("notes", row.get("reviewer_notes", "")),
    ]
    notes = "; ".join(f"{key}={value}" for key, value in fields if value)
    return notes or "approved by reviewer"


def validate_or_exit(records: list[dict]) -> None:
    failures = [result for result in validate_sft_records(records) if not result.ok]
    if failures:
        for result in failures[:20]:
            for error in result.errors:
                print(f"record {result.line_number}: {error}", file=sys.stderr)
        raise SystemExit(f"promotion produced invalid rows: {len(failures)} failures")


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    *,
    draft_count: int,
    review_count: int,
    train_count: int,
    validation_count: int,
) -> None:
    text = f"""# SFT Promotion Summary

Draft rows: {draft_count}
Review rows: {review_count}
Promoted train rows: {train_count}
Promoted validation rows: {validation_count}

Only rows marked `approved` in the review TSV were exported.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

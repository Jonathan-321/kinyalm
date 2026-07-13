import csv
import json
import subprocess
import sys


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def draft_row(row_id):
    return {
        "id": row_id,
        "task_type": "greeting",
        "split": "draft",
        "source": "synthetic-draft",
        "source_status": "team-authored",
        "review_status": "needs-review",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": "Teach me a greeting."},
            {"role": "assistant", "content": "Muraho means hello."},
        ],
        "reviewer_notes": "draft",
    }


def write_review_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "id",
                "review_status",
                "correctness_1_5",
                "naturalness_1_5",
                "helpfulness_1_5",
                "failure_tags",
                "reviewer",
                "reviewer_notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def approved_review(row_id, **overrides):
    row = {
        "id": row_id,
        "review_status": "approved",
        "correctness_1_5": "5",
        "naturalness_1_5": "5",
        "helpfulness_1_5": "5",
        "failure_tags": "",
        "reviewer": "Tessy",
        "reviewer_notes": "looks good",
    }
    row.update(overrides)
    return row


def run_promotion(draft_path, review_path, out_dir):
    return subprocess.run(
        [
            sys.executable,
            "scripts/promote_reviewed_sft.py",
            "--draft-jsonl",
            str(draft_path),
            "--review-tsv",
            str(review_path),
            "--out-dir",
            str(out_dir),
            "--train-ratio",
            "0.5",
        ],
        capture_output=True,
        text=True,
    )


def test_promote_reviewed_sft_exports_only_approved_rows(tmp_path):
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.tsv"
    out_dir = tmp_path / "approved"
    write_jsonl(draft_path, [draft_row("row-001"), draft_row("row-002")])
    write_review_tsv(
        review_path,
        [
            approved_review("row-001"),
            {
                "id": "row-002",
                "review_status": "needs-fix",
                "correctness_1_5": "2",
                "naturalness_1_5": "2",
                "helpfulness_1_5": "3",
                "failure_tags": "grammar",
                "reviewer": "Tessy",
                "reviewer_notes": "fix grammar",
            },
        ],
    )

    result = run_promotion(draft_path, review_path, out_dir)
    assert result.returncode == 0, result.stderr

    validation_text = (out_dir / "validation.jsonl").read_text(encoding="utf-8")
    promoted = [json.loads(line) for line in validation_text.splitlines() if line]
    train_text = (out_dir / "train.jsonl").read_text(encoding="utf-8")

    assert train_text == ""
    assert len(promoted) == 1
    assert promoted[0]["id"] == "row-001"
    assert promoted[0]["review_status"] == "approved"


def test_promote_reviewed_sft_rejects_non_draft_input(tmp_path):
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.tsv"
    out_dir = tmp_path / "approved"
    row = draft_row("row-001")
    row["split"] = "benchmark-only"
    write_jsonl(draft_path, [row])
    write_review_tsv(review_path, [approved_review("row-001")])

    result = run_promotion(draft_path, review_path, out_dir)

    assert result.returncode != 0
    assert "non-draft row row-001" in result.stderr


def test_promote_reviewed_sft_rejects_low_approved_scores(tmp_path):
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.tsv"
    out_dir = tmp_path / "approved"
    write_jsonl(draft_path, [draft_row("row-001")])
    write_review_tsv(
        review_path,
        [approved_review("row-001", correctness_1_5="3")],
    )

    result = run_promotion(draft_path, review_path, out_dir)

    assert result.returncode != 0
    assert "correctness_1_5 must be 4 or 5" in result.stderr


def test_promote_reviewed_sft_rejects_duplicate_review_ids(tmp_path):
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.tsv"
    out_dir = tmp_path / "approved"
    write_jsonl(draft_path, [draft_row("row-001")])
    write_review_tsv(
        review_path,
        [
            approved_review("row-001"),
            approved_review("row-001", reviewer_notes="duplicate"),
        ],
    )

    result = run_promotion(draft_path, review_path, out_dir)

    assert result.returncode != 0
    assert "duplicate review id: row-001" in result.stderr

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


def test_promote_reviewed_sft_exports_only_approved_rows(tmp_path):
    draft_path = tmp_path / "draft.jsonl"
    review_path = tmp_path / "review.tsv"
    out_dir = tmp_path / "approved"
    write_jsonl(draft_path, [draft_row("row-001"), draft_row("row-002")])
    with review_path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerow(
            {
                "id": "row-001",
                "review_status": "approved",
                "correctness_1_5": "5",
                "naturalness_1_5": "5",
                "helpfulness_1_5": "5",
                "failure_tags": "",
                "reviewer": "Tessy",
                "reviewer_notes": "looks good",
            }
        )
        writer.writerow(
            {
                "id": "row-002",
                "review_status": "needs-fix",
                "correctness_1_5": "2",
                "naturalness_1_5": "2",
                "helpfulness_1_5": "3",
                "failure_tags": "grammar",
                "reviewer": "Tessy",
                "reviewer_notes": "fix grammar",
            }
        )

    subprocess.run(
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
        check=True,
    )

    validation_text = (out_dir / "validation.jsonl").read_text(encoding="utf-8")
    promoted = [json.loads(line) for line in validation_text.splitlines() if line]
    train_text = (out_dir / "train.jsonl").read_text(encoding="utf-8")

    assert train_text == ""
    assert len(promoted) == 1
    assert promoted[0]["id"] == "row-001"
    assert promoted[0]["review_status"] == "approved"

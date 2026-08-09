import csv
import json
import subprocess
import sys

import pytest

from scripts.build_native_review_sft import (
    build_records,
    convert_review_row,
    prompt_overlap_report,
)

FIELDS = [
    "conversation_id",
    "batch_id",
    "task_family",
    "difficulty",
    "source_language",
    "target_language",
    "user_turns",
    "assistant_turns",
    "is_multi_turn",
    "messages_json",
    "teacher_model",
    "generation_status",
    "reviewer_name",
    "review_status",
    "corrected_messages_json",
    "reviewer_notes",
    "repetition_flag",
    "factual_error_flag",
    "naturalness_score_1_5",
    "correctness_score_1_5",
    "approved_for_training",
]


def review_row(index=1, **overrides):
    messages = [
        {"role": "user", "content": f"Ikibazo cya {index}?"},
        {"role": "assistant", "content": f"Igisubizo cya {index}."},
        {"role": "user", "content": f"Sobanura {index}."},
        {"role": "assistant", "content": f"Ibisobanuro bya {index}."},
    ]
    row = {
        "conversation_id": f"KINYA-SFT-{index:04d}",
        "batch_id": "BATCH-01",
        "task_family": "natural-conversation",
        "difficulty": "beginner",
        "source_language": "rw",
        "target_language": "rw",
        "user_turns": "2",
        "assistant_turns": "2",
        "is_multi_turn": "TRUE",
        "messages_json": json.dumps(messages),
        "teacher_model": "Gemini 3.1 Pro web",
        "generation_status": "candidate",
        "reviewer_name": "Jonathan Muhire",
        "review_status": "approved",
        "corrected_messages_json": "",
        "reviewer_notes": "Native review complete.",
        "repetition_flag": "FALSE",
        "factual_error_flag": "FALSE",
        "naturalness_score_1_5": "",
        "correctness_score_1_5": "",
        "approved_for_training": "TRUE",
    }
    row.update(overrides)
    return row


def test_approved_multiturn_row_maps_to_canonical_schema():
    record = convert_review_row(review_row(), dataset_id="native-v1")

    assert record["id"] == "kinya-sft-0001"
    assert record["task_type"] == "dialogue"
    assert record["review_status"] == "approved"
    assert record["language_mix"] == "kinyarwanda"
    assert record["assistant_turn_count"] == 2
    assert record["reviewers"] == ["Jonathan Muhire"]


def test_corrected_row_uses_corrected_messages():
    corrected = [
        {"role": "user", "content": "Kosora iyi nteruro."},
        {"role": "assistant", "content": "Dore interuro ikosoye."},
    ]
    row = review_row(
        review_status="corrected",
        corrected_messages_json=json.dumps(corrected),
        user_turns="1",
        assistant_turns="1",
        is_multi_turn="FALSE",
    )

    record = convert_review_row(row, dataset_id="native-v1")

    assert record["messages"] == corrected
    assert "Basis: corrected_messages_json" in record["reviewer_notes"]


def test_approved_row_with_failure_flag_is_rejected():
    with pytest.raises(ValueError, match="repetition_flag=true"):
        convert_review_row(review_row(repetition_flag="TRUE"), dataset_id="native-v1")


def test_build_records_is_exact_stratified_and_deterministic():
    rows = [
        review_row(
            index,
            task_family=("natural-conversation" if index <= 10 else "en-to-rw"),
            source_language=("rw" if index <= 10 else "en"),
            target_language="rw",
        )
        for index in range(1, 21)
    ]

    first, report = build_records(
        rows,
        dataset_id="native-v1",
        train_ratio=0.8,
        split_seed="fixed",
        expected_records=20,
    )
    second, _ = build_records(
        rows,
        dataset_id="native-v1",
        train_ratio=0.8,
        split_seed="fixed",
        expected_records=20,
    )

    assert report["split_counts"] == {"train": 16, "validation": 4}
    assert {row["task_type"] for row in first if row["split"] == "validation"} == {
        "dialogue",
        "translation-en-rw",
    }
    assert [(row["id"], row["split"]) for row in first] == [
        (row["id"], row["split"]) for row in second
    ]


def test_prompt_overlap_report_detects_evaluation_copy():
    record = convert_review_row(review_row(), dataset_id="native-v1")

    report = prompt_overlap_report([record], [record])

    assert report["exact_pairs"] == 1
    assert report["near_pairs"] == 1


def test_cli_writes_hash_pinned_training_package(tmp_path):
    csv_path = tmp_path / "review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(1, 21):
            writer.writerow(
                review_row(
                    index,
                    task_family=("natural-conversation" if index <= 10 else "en-to-rw"),
                    source_language=("rw" if index <= 10 else "en"),
                    target_language="rw",
                )
            )
    output_dir = tmp_path / "package"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_native_review_sft.py",
            "--review-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--dataset-id",
            "native-v1",
            "--train-ratio",
            "0.8",
            "--expected-records",
            "20",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (output_dir / "dataset-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["human_reviewed"] is True
    assert manifest["training_eligible"] is True
    assert manifest["production_eligible"] is False
    assert manifest["outputs"]["train"]["rows"] == 16
    assert manifest["outputs"]["validation"]["rows"] == 4

    verified_dir = tmp_path / "verified"
    verify_result = subprocess.run(
        [
            sys.executable,
            "scripts/download_reviewed_sft.py",
            "--revision",
            "local-test-snapshot",
            "--source-dir",
            str(output_dir),
            "--output-dir",
            str(verified_dir),
            "--minimum-rows",
            "20",
            "--maximum-rows",
            "20",
        ],
        capture_output=True,
        text=True,
    )
    assert verify_result.returncode == 0, verify_result.stderr
    assert (verified_dir / "dataset-manifest.json").is_file()

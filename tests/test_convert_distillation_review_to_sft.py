import json
import subprocess
import sys

import pytest

from scripts.convert_distillation_review_to_sft import (
    build_row,
    convert_review_rows,
)

CONVERSATION = """USER:
Muraho.

ASSISTANT:
Muraho neza.

USER:
Amakuru?

ASSISTANT:
Ni meza, urakoze."""


def review_row(**overrides):
    row = {
        "conversation_id": "sft-distill-production-dialogue-0001-abcd1234",
        "task_family": "conversation-practice",
        "priority": "Critic accepted",
        "my_flag": "Keep",
        "original_conversation": CONVERSATION,
        "suggested_revision": "",
        "critic_feedback": "The conversation is correct.",
    }
    row.update(overrides)
    return row


def test_build_row_preserves_the_full_multiturn_conversation():
    row = build_row(review_row(), "Tessy Mugisha", 0.9)

    assert row is not None
    assert row["id"] == "sft-distill-production-dialogue-0001-abcd1234"
    assert [message["role"] for message in row["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert row["messages"][2]["content"] == "Amakuru?"


def test_disputed_keep_is_withheld_without_explicit_adjudication():
    approved, withheld = convert_review_rows(
        [review_row(priority="Repair first")],
        "Tessy Mugisha",
        0.9,
    )

    assert approved == []
    assert withheld == ["sft-distill-production-dialogue-0001-abcd1234"]


def test_disputed_keep_can_be_explicitly_promoted():
    approved, withheld = convert_review_rows(
        [review_row(priority="Repair first")],
        "Tessy Mugisha",
        0.9,
        accept_disputed_keeps=True,
    )

    assert len(approved) == 1
    assert withheld == []


def test_unknown_task_family_fails_instead_of_silently_becoming_dialogue():
    with pytest.raises(ValueError, match="unknown task family"):
        build_row(
            review_row(task_family="unknown-family"),
            "Tessy Mugisha",
            0.9,
        )


def test_cli_writes_canonical_and_mlx_compatible_splits(tmp_path):
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        json.dumps(review_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out_prefix = tmp_path / "reviewed"
    mlx_dir = tmp_path / "mlx-data"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_distillation_review_to_sft.py",
            "--review-jsonl",
            str(review_path),
            "--out-prefix",
            str(out_prefix),
            "--mlx-data-dir",
            str(mlx_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "reviewed.train.jsonl").exists()
    assert (tmp_path / "reviewed.validation.jsonl").exists()
    assert (mlx_dir / "train.jsonl").exists()
    assert (mlx_dir / "valid.jsonl").exists()

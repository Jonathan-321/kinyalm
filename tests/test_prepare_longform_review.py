import json

import pytest

from scripts.prepare_longform_review import validate_and_prepare


def conversation(row_id="KINYA-LONG-0001", assistant_text=None):
    answer = assistant_text or " ".join(["igisubizo"] * 120)
    return {
        "conversation_id": row_id,
        "batch_id": "BATCH-LONG-01",
        "task_family": "morphology",
        "difficulty": "intermediate",
        "source_language": "rw",
        "target_language": "rw",
        "messages": [
            {"role": "user", "content": f"Sobanura ikibazo {row_id}."},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Tanga urundi rugero."},
            {"role": "assistant", "content": answer + " urugero"},
        ],
    }


def test_prepares_candidate_review_rows_with_no_implicit_approval():
    prepared, report = validate_and_prepare(
        [conversation()],
        teacher_model="Gemini 3.1 Pro web",
        token_counter=lambda text: len(text.split()),
    )

    assert prepared[0]["review_status"] == "needs-review"
    assert prepared[0]["approved_for_training"] == "FALSE"
    assert prepared[0]["assistant_turns"] == "2"
    assert report["assistant_response_count"] == 2
    assert report["training_eligible_rows"] == 0


def test_flags_short_and_generic_praise_without_dropping_candidate():
    row = conversation(assistant_text="Excellent job. Iki ni igisubizo kigufi.")
    prepared, report = validate_and_prepare(
        [row],
        teacher_model="Gemini 3.1 Pro web",
        token_counter=lambda text: len(text.split()),
    )

    assert prepared[0]["generation_status"] == "candidate-needs-fix"
    assert "assistant-under-token-minimum" in prepared[0]["reviewer_notes"]
    assert "generic-praise" in prepared[0]["reviewer_notes"]
    assert report["rows_needing_fix"] == 1


def test_rejects_duplicate_ids_and_broken_role_order():
    with pytest.raises(ValueError, match="duplicate conversation_id"):
        validate_and_prepare(
            [conversation(), conversation()],
            teacher_model="Gemini",
            token_counter=lambda text: len(text.split()),
        )

    broken = json.loads(json.dumps(conversation()))
    broken["conversation_id"] = "KINYA-LONG-0002"
    broken["messages"][1]["role"] = "user"
    with pytest.raises(ValueError, match="role=assistant"):
        validate_and_prepare(
            [broken],
            teacher_model="Gemini",
            token_counter=lambda text: len(text.split()),
        )

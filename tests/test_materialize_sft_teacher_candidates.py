import json

import pytest

from scripts.materialize_sft_teacher_candidates import materialize


def make_job(turn_count: int = 1) -> dict:
    return {
        "content_key": "profile:family:one",
        "difficulty": "beginner",
        "evaluation_categories": ["translation"],
        "job_id": "job-one",
        "language_mix": "kinyarwanda+english",
        "profile_id": "profile",
        "prompt_version": "prompt-v1",
        "source_group_id": "profile:family:one",
        "task_family": "translation",
        "task_type": "translation-en-rw",
        "turn_count": turn_count,
    }


def make_response(messages: list[dict] | None = None) -> dict:
    generated = {
        "lesson_focus": "Greeting",
        "messages": messages
        or [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Muraho."},
        ],
        "self_check": {
            "factually_grounded": True,
            "natural_kinyarwanda": True,
            "safe_and_low_stakes": True,
            "turns_consistent": True,
        },
        "skills": ["greeting"],
        "source_assertion": "original-model-authored-no-copied-source-text",
    }
    return {
        "error": None,
        "job_id": "job-one",
        "model": "teacher",
        "output_text": json.dumps(generated),
        "provider": "provider",
        "requested_model": "provider/teacher",
        "response_id": "response-one",
    }


def test_materialize_preserves_schema_valid_candidate() -> None:
    rows = materialize([make_job()], [make_response()])

    assert len(rows) == 1
    assert rows[0]["review_status"] == "candidate-unreviewed"
    assert rows[0]["messages"][1]["content"] == "Muraho."


def test_materialize_flags_turn_count_mismatch() -> None:
    row = materialize([make_job(turn_count=2)], [make_response()])[0]

    assert row["requested_turn_count"] == 2
    assert row["turn_count"] == 1
    assert row["candidate_flags"] == [
        "turn-count-deviation:expected-2-actual-1"
    ]


def test_materialize_rejects_incomplete_turn_pair() -> None:
    response = make_response([{"role": "user", "content": "Say hello."}])

    with pytest.raises(ValueError, match="incomplete user/assistant pair"):
        materialize([make_job()], [response])


def test_materialize_normalizes_role_labels_and_records_flag() -> None:
    response = make_response(
        [
            {"role": "assistant", "content": "Say hello."},
            {"role": "assistant", "content": "Muraho."},
        ]
    )

    row = materialize([make_job()], [response])[0]

    assert [message["role"] for message in row["messages"]] == [
        "user",
        "assistant",
    ]
    assert row["candidate_flags"] == ["role-sequence-normalized"]
    assert row["raw_message_roles"] == ["assistant", "assistant"]

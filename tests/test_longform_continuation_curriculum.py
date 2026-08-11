from copy import deepcopy

from scripts.build_longform_continuation_curriculum import build_curriculum


def _record(row_id, split, family, assistant_turns=1):
    messages = []
    for turn in range(assistant_turns):
        messages.extend(
            [
                {"role": "user", "content": f"Prompt {row_id} {turn}"},
                {"role": "assistant", "content": f"Answer {row_id} {turn}"},
            ]
        )
    task_type = {
        "formal-informal": "culture-register",
        "sentence-correction": "sentence-correction",
        "uncertainty-clarification": "uncertainty",
        "natural-conversation": "dialogue",
    }[family]
    return {
        "id": row_id,
        "task_type": task_type,
        "task_family": family,
        "split": split,
        "source": "team-reviewed",
        "source_status": "team-authored",
        "review_status": "approved",
        "language_mix": "kinyarwanda+english",
        "messages": messages,
        "reviewer_notes": "Reviewed by the KinyaLM team.",
    }


def test_curriculum_repeats_only_target_families_without_editing_messages():
    train = [
        _record("formal-1", "train", "formal-informal", 2),
        _record("correction-1", "train", "sentence-correction", 1),
        _record("uncertainty-1", "train", "uncertainty-clarification", 2),
        _record("conversation-1", "train", "natural-conversation", 2),
    ]
    validation = [
        _record("validation-1", "validation", "natural-conversation", 1)
    ]
    original_train = deepcopy(train)
    original_validation = deepcopy(validation)

    weighted_train, weighted_validation, report = build_curriculum(
        train,
        validation,
    )

    assert len(weighted_train) == 7
    assert len(weighted_validation) == 1
    assert report["curriculum"]["duplicate_conversations"] == 3
    assert report["curriculum"]["duplicate_assistant_turns"] == 5
    assert report["curriculum"]["message_content_changed"] is False
    assert report["curriculum"]["validation_changed"] is False
    duplicates = [
        row for row in weighted_train if "curriculum_source_record_id" in row
    ]
    originals = {row["id"]: row for row in original_train}
    assert {
        row["curriculum_reason"] for row in duplicates
    } == {
        "formal-informal",
        "sentence-correction",
        "uncertainty-clarification",
    }
    for row in duplicates:
        assert row["messages"] == originals[row["curriculum_source_record_id"]][
            "messages"
        ]
    assert weighted_validation == original_validation

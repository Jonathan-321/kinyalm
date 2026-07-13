from kinyalm.data.sft import validate_sft_records


def valid_record(**overrides):
    record = {
        "id": "sft-greeting-001",
        "task_type": "greeting",
        "split": "benchmark-only",
        "source": "team-authored",
        "source_status": "team-authored",
        "review_status": "needs-review",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": "Teach me a greeting."},
            {"role": "assistant", "content": "Muraho means hello."},
        ],
        "reviewer_notes": "",
    }
    record.update(overrides)
    return record


def test_valid_benchmark_record_passes():
    results = validate_sft_records([valid_record()])

    assert results[0].ok


def test_draft_record_can_need_review_without_passing_training_gate():
    results = validate_sft_records(
        [valid_record(split="draft", review_status="needs-review")]
    )

    assert results[0].ok


def test_train_rows_require_approved_review_and_trainable_source():
    results = validate_sft_records(
        [
            valid_record(
                split="train",
                source_status="reference-only",
                review_status="needs-review",
            )
        ]
    )

    assert not results[0].ok
    assert "train/validation rows must have review_status=approved" in results[0].errors
    assert any("source_status approved" in error for error in results[0].errors)


def test_duplicate_ids_are_rejected():
    results = validate_sft_records([valid_record(), valid_record()])

    assert results[0].ok
    assert not results[1].ok
    assert "duplicate id: sft-greeting-001" in results[1].errors


def test_message_roles_are_strict_for_first_run():
    results = validate_sft_records(
        [
            valid_record(
                messages=[
                    {"role": "assistant", "content": "Muraho means hello."},
                    {"role": "user", "content": "Teach me a greeting."},
                ]
            )
        ]
    )

    assert not results[0].ok
    assert "messages[0].role must be user" in results[0].errors
    assert "messages[1].role must be assistant" in results[0].errors

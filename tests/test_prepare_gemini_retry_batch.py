import pytest

from scripts.prepare_gemini_retry_batch import build_retry_batch


def source_job(number: int, conversation_numbers: list[int]) -> dict:
    return {
        "job_id": f"KINYA-GEMINI-JOB-{number:04d}",
        "expected_conversations": [
            {
                "conversation_id": f"KINYA-SCALE-{conversation_number:06d}",
                "batch_id": f"GEMINI-SCALE-BATCH-{number:04d}",
                "task_family": "natural-conversation",
                "difficulty": "beginner",
                "source_language": "rw",
                "target_language": "rw",
                "message_count": 4,
                "scenario_seed": "school and study",
            }
            for conversation_number in conversation_numbers
        ],
    }


def test_builds_one_low_reasoning_request_for_each_missing_conversation():
    jobs, api_rows, manifest = build_retry_batch(
        source_jobs=[source_job(1, [1, 2]), source_job(2, [3])],
        usable_ids={"KINYA-SCALE-000001"},
        prompt_template="Generate {{BATCH_SPECIFICATION_JSON}}",
        generation_config={
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    )

    assert [row["job_id"] for row in jobs] == [
        "KINYA-GEMINI-RETRY-000002",
        "KINYA-GEMINI-RETRY-000003",
    ]
    assert all(len(row["expected_conversations"]) == 1 for row in jobs)
    assert all(
        row["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"]
        == "low"
        for row in api_rows
    )
    assert all(
        row["request"]["generationConfig"]["responseJsonSchema"]["minItems"] == 1
        for row in api_rows
    )
    assert manifest["already_usable_conversation_count"] == 1
    assert manifest["missing_conversation_count"] == 2
    assert manifest["remaining_after_selected_count"] == 0


def test_limit_creates_a_smoke_subset_without_losing_gap_accounting():
    jobs, _, manifest = build_retry_batch(
        source_jobs=[source_job(1, [1, 2, 3])],
        usable_ids=set(),
        prompt_template="Generate {{BATCH_SPECIFICATION_JSON}}",
        generation_config={"thinkingConfig": {"thinkingLevel": "low"}},
        limit=2,
    )

    assert len(jobs) == 2
    assert manifest["missing_conversation_count"] == 3
    assert manifest["selected_retry_count"] == 2
    assert manifest["remaining_after_selected_count"] == 1


def test_packs_missing_conversations_and_sizes_each_response_schema():
    jobs, api_rows, manifest = build_retry_batch(
        source_jobs=[source_job(1, [1, 2, 3]), source_job(2, [4, 5])],
        usable_ids=set(),
        prompt_template="Generate {{BATCH_SPECIFICATION_JSON}}",
        generation_config={"thinkingConfig": {"thinkingLevel": "low"}},
        conversations_per_request=4,
    )

    assert [len(row["expected_conversations"]) for row in jobs] == [4, 1]
    assert [
        row["request"]["generationConfig"]["responseJsonSchema"]["minItems"]
        for row in api_rows
    ] == [4, 1]
    assert manifest["request_count"] == 2
    assert manifest["conversations_per_request"] == 4


def test_rejects_usable_conversations_outside_source_queue():
    with pytest.raises(ValueError, match="outside the source queue"):
        build_retry_batch(
            source_jobs=[source_job(1, [1])],
            usable_ids={"KINYA-SCALE-999999"},
            prompt_template="Generate {{BATCH_SPECIFICATION_JSON}}",
            generation_config={"thinkingConfig": {"thinkingLevel": "low"}},
        )

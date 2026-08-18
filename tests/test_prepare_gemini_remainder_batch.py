import pytest

from scripts.prepare_gemini_remainder_batch import select_remainder_batch


def api_job(number: int, conversation_id: str) -> dict:
    return {
        "job_id": f"KINYA-GEMINI-JOB-{number:04d}",
        "expected_conversations": [
            {"conversation_id": conversation_id, "message_count": 4}
        ],
    }


def api_input(number: int) -> dict:
    return {"key": f"KINYA-GEMINI-JOB-{number:04d}", "request": {}}


def web_job(number: int, source_numbers: list[int]) -> dict:
    return {
        "web_job_id": f"KINYA-GEMINI-WEB-JOB-{number:04d}",
        "source_job_ids": [
            f"KINYA-GEMINI-JOB-{source:04d}" for source in source_numbers
        ],
    }


def test_selects_missing_and_invalid_jobs_without_accepted_overlap():
    jobs = [
        api_job(1, "accepted"),
        api_job(2, "missing-a"),
        api_job(3, "invalid-a"),
    ]
    inputs = [api_input(number) for number in (1, 2, 3)]
    web_jobs = [
        web_job(1, [1]),
        web_job(2, [2]),
        web_job(3, [3]),
    ]
    report = {
        "missing_job_ids": ["KINYA-GEMINI-WEB-JOB-0002"],
        "invalid_jobs": [{"web_job_id": "KINYA-GEMINI-WEB-JOB-0003"}],
    }

    selected_jobs, selected_input, manifest = select_remainder_batch(
        api_jobs=jobs,
        api_input=inputs,
        web_jobs=web_jobs,
        collection_report=report,
        accepted_ids={"accepted"},
        web_job_limit=2,
    )

    assert [row["job_id"] for row in selected_jobs] == [
        "KINYA-GEMINI-JOB-0002",
        "KINYA-GEMINI-JOB-0003",
    ]
    assert [row["key"] for row in selected_input] == [
        "KINYA-GEMINI-JOB-0002",
        "KINYA-GEMINI-JOB-0003",
    ]
    assert manifest["conversation_count"] == 2
    assert manifest["assistant_response_count"] == 4
    assert manifest["accepted_conversation_overlap_count"] == 0


def test_rejects_overlap_with_accepted_web_conversation():
    with pytest.raises(ValueError, match="overlaps accepted web conversations"):
        select_remainder_batch(
            api_jobs=[api_job(1, "accepted")],
            api_input=[api_input(1)],
            web_jobs=[web_job(1, [1])],
            collection_report={
                "missing_job_ids": ["KINYA-GEMINI-WEB-JOB-0001"],
                "invalid_jobs": [],
            },
            accepted_ids={"accepted"},
            web_job_limit=1,
        )

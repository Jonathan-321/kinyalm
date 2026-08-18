import copy
import json

import pytest

from scripts.collect_gemini_batch_results import collect_results


def example_job():
    spec = {
        "conversation_id": "KINYA-SCALE-000001",
        "batch_id": "GEMINI-SCALE-BATCH-0001",
        "task_family": "natural-conversation",
        "difficulty": "beginner",
        "source_language": "rw",
        "target_language": "rw",
        "message_count": 2,
        "scenario_seed": "school and study",
    }
    return {"job_id": "KINYA-GEMINI-JOB-0001", "expected_conversations": [spec]}


def provider_row(job):
    spec = job["expected_conversations"][0]
    conversation = {
        key: value
        for key, value in spec.items()
        if key not in {"message_count", "scenario_seed"}
    }
    conversation["messages"] = [
        {"role": "user", "content": "Amakuru?"},
        {"role": "assistant", "content": "Ni meza, urakoze."},
    ]
    return {
        "key": job["job_id"],
        "response": {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps([conversation])}]}}
            ]
        },
    }


def test_collects_valid_result_and_preserves_raw_provider_row(tmp_path):
    job = example_job()
    row = provider_row(job)
    conversations, report = collect_results(
        {job["job_id"]: job}, [row], tmp_path
    )

    assert report["complete"] is True
    assert report["conversation_count"] == 1
    assert conversations[0]["conversation_id"] == "KINYA-SCALE-000001"
    raw_path = tmp_path / "raw-provider-results/KINYA-GEMINI-JOB-0001.jsonl"
    assert json.loads(raw_path.read_text()) == row


def test_rejects_metadata_drift_without_repairing_it(tmp_path):
    job = example_job()
    row = provider_row(job)
    text_part = row["response"]["candidates"][0]["content"]["parts"][0]
    payload = json.loads(text_part["text"])
    payload[0]["difficulty"] = "advanced"
    text_part["text"] = json.dumps(payload)

    conversations, report = collect_results(
        {job["job_id"]: job}, [row], tmp_path
    )

    assert conversations == []
    assert report["complete"] is False
    assert "changed difficulty" in report["errors"][0]["error"]


def test_refuses_to_overwrite_a_changed_provider_result(tmp_path):
    job = example_job()
    original = provider_row(job)
    collect_results({job["job_id"]: job}, [original], tmp_path)
    changed = copy.deepcopy(original)
    changed["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"

    with pytest.raises(ValueError, match="refusing to overwrite"):
        collect_results({job["job_id"]: job}, [changed], tmp_path)


def test_salvages_only_complete_conversations_from_truncated_array(tmp_path):
    job = example_job()
    second = copy.deepcopy(job["expected_conversations"][0])
    second["conversation_id"] = "KINYA-SCALE-000002"
    job["expected_conversations"].append(second)
    row = provider_row(job)
    first = json.loads(
        row["response"]["candidates"][0]["content"]["parts"][0]["text"]
    )[0]
    row["response"]["candidates"][0]["content"]["parts"][0]["text"] = (
        "[" + json.dumps(first) + ', {"conversation_id":"KINYA-SCALE-000002"'
    )

    conversations, report = collect_results(
        {job["job_id"]: job}, [row], tmp_path
    )

    assert [row["conversation_id"] for row in conversations] == [
        "KINYA-SCALE-000001"
    ]
    assert report["valid_job_count"] == 0
    assert report["partial_job_count"] == 1
    assert report["salvaged_conversation_count"] == 1
    assert report["complete"] is False

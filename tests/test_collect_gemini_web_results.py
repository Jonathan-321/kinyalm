import json
from pathlib import Path

import pytest

from scripts.collect_gemini_web_results import collect_web_results, normalize_candidate


def expected_spec():
    return {
        "conversation_id": "KINYA-SCALE-000001",
        "batch_id": "GEMINI-SCALE-BATCH-0001",
        "task_family": "natural-conversation",
        "difficulty": "beginner",
        "source_language": "rw",
        "target_language": "rw",
        "message_count": 2,
        "scenario_seed": "school and study",
    }


def candidate(spec):
    return {
        **spec,
        "messages": [
            {"role": "user", "content": "Amakuru?"},
            {"role": "assistant", "content": "Ni meza."},
        ],
    }


def test_drops_only_matching_queue_fields_without_changing_messages():
    spec = expected_spec()
    original = candidate(spec)
    messages_before = json.dumps(original["messages"], ensure_ascii=False)

    rows, dropped, role_corrections = normalize_candidate(
        [original], [spec], "web-job"
    )

    assert dropped == ["message_count", "scenario_seed"]
    assert set(rows[0]) == {
        "conversation_id",
        "batch_id",
        "task_family",
        "difficulty",
        "source_language",
        "target_language",
        "messages",
    }
    assert json.dumps(rows[0]["messages"], ensure_ascii=False) == messages_before
    assert role_corrections == []


def test_rejects_changed_queue_metadata():
    spec = expected_spec()
    original = candidate(spec)
    original["message_count"] = 8

    with pytest.raises(ValueError, match="changed message_count"):
        normalize_candidate([original], [spec], "web-job")


def test_rejects_unknown_extra_fields():
    spec = expected_spec()
    original = candidate(spec)
    original["teacher_note"] = "rewrite this"

    with pytest.raises(ValueError, match="unsupported extra fields"):
        normalize_candidate([original], [spec], "web-job")


def test_normalizes_role_labels_by_verified_message_position():
    spec = expected_spec()
    original = candidate(spec)
    original["messages"][1]["role"] = "completion"

    rows, _, corrections = normalize_candidate([original], [spec], "web-job")

    assert rows[0]["messages"][1]["role"] == "assistant"
    assert rows[0]["messages"][1]["content"] == "Ni meza."
    assert corrections == [
        {
            "conversation_id": "KINYA-SCALE-000001",
            "message_index": 1,
            "from": "completion",
            "to": "assistant",
        }
    ]


def test_salvages_valid_conversations_across_immutable_attempts(tmp_path):
    first = expected_spec()
    second = {
        **expected_spec(),
        "conversation_id": "KINYA-SCALE-000002",
    }
    job = {
        "web_job_id": "KINYA-GEMINI-WEB-JOB-0001",
        "expected_conversations": [first, second],
    }
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "validated"
    raw_dir.mkdir()

    first_attempt = [candidate(first), candidate(second)]
    first_attempt[1]["messages"].pop()
    second_attempt = [candidate(first), candidate(second)]
    second_attempt[0]["messages"].pop()
    for attempt_number, value in enumerate((first_attempt, second_attempt), start=1):
        path = raw_dir / (
            "KINYA-GEMINI-WEB-JOB-0001."
            f"attempt-{attempt_number:02d}.raw.txt"
        )
        path.write_text(json.dumps(value), encoding="utf-8")

    report = collect_web_results([job], raw_dir, output_dir)

    assert report["completed_web_jobs"] == 1
    assert report["conversation_count"] == 2
    manifest = json.loads(
        (output_dir / "KINYA-GEMINI-WEB-JOB-0001.normalization.json").read_text()
    )
    assert manifest["assembled_from_valid_conversations"] is True
    assert len(manifest["source_attempts"]) == 2


def test_does_not_publish_an_incomplete_salvaged_job(tmp_path):
    first = expected_spec()
    second = {
        **expected_spec(),
        "conversation_id": "KINYA-SCALE-000002",
    }
    job = {
        "web_job_id": "KINYA-GEMINI-WEB-JOB-0001",
        "expected_conversations": [first, second],
    }
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "validated"
    raw_dir.mkdir()
    only_first = raw_dir / "KINYA-GEMINI-WEB-JOB-0001.attempt-01.raw.txt"
    only_first.write_text(json.dumps([candidate(first)]), encoding="utf-8")

    report = collect_web_results([job], raw_dir, output_dir)

    assert report["completed_web_jobs"] == 0
    assert report["invalid_jobs"][0]["valid_conversation_ids"] == [
        "KINYA-SCALE-000001"
    ]
    assert not (output_dir / "KINYA-GEMINI-WEB-JOB-0001.validated.json").exists()


def test_excludes_attempt_marked_nonpro_by_provenance(tmp_path):
    spec = expected_spec()
    job = {
        "web_job_id": "KINYA-GEMINI-WEB-JOB-0001",
        "expected_conversations": [spec],
    }
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "validated"
    raw_dir.mkdir()
    attempt = raw_dir / "KINYA-GEMINI-WEB-JOB-0001.attempt-01.raw.txt"
    attempt.write_text(json.dumps([candidate(spec)]), encoding="utf-8")
    Path(f"{attempt}.provenance.json").write_text(
        json.dumps({"counts_toward_pro_token_target": False}),
        encoding="utf-8",
    )

    report = collect_web_results([job], raw_dir, output_dir)

    assert report["completed_web_jobs"] == 0
    assert report["invalid_jobs"][0]["attempts"] == [
        {
            "attempt": attempt.name,
            "error": "excluded by teacher provenance",
        }
    ]

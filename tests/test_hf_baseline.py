import json

import pytest

from kinyalm.data.hf_baseline import build_baseline_records


def draft(row_id, *, response=None):
    return {
        "id": row_id,
        "task_type": "vocabulary",
        "task_family": "vocabulary-definition-usage",
        "split": "draft",
        "source": "synthetic-distillation",
        "source_status": "model-generated",
        "review_status": "needs-review",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": f"Question for {row_id}"},
            {
                "role": "assistant",
                "content": response or f"Answer for {row_id}",
            },
        ],
        "reviewer_notes": "draft",
        "content_key": f"content-{row_id}",
        "source_group_id": f"group-{row_id}",
    }


def critic_job(row_id):
    return {"job_id": f"critic-{row_id}", "source_record_id": row_id}


def critic_response(row_id, recommendation="accept", *, revised_messages=None):
    scores = {
        "correctness": 5 if recommendation == "accept" else 3,
        "naturalness": 5,
        "helpfulness": 5,
        "consistency": 5,
        "difficulty_fit": 5,
    }
    assessment = {
        "recommendation": recommendation,
        "scores": scores,
        "failure_tags": [] if recommendation == "accept" else ["wrong-answer"],
        "brief_notes": "reviewed",
        "revised_messages": revised_messages,
    }
    return {
        "job_id": f"critic-{row_id}",
        "error": None,
        "model": "critic-model",
        "response_id": f"response-{row_id}",
        "output_text": json.dumps(assessment),
    }


def build_inputs():
    drafts = [draft(f"row-{index:03d}") for index in range(1, 7)]
    jobs = [critic_job(row["id"]) for row in drafts]
    responses = [critic_response(row["id"]) for row in drafts]
    repaired = [
        {"role": "user", "content": "Repaired question"},
        {"role": "assistant", "content": "Repaired answer"},
    ]
    responses[-1] = critic_response(
        drafts[-1]["id"], "repair", revised_messages=repaired
    )
    return drafts, jobs, responses, repaired


def test_accepted_baseline_excludes_repair_recommendations():
    drafts, jobs, responses, _ = build_inputs()

    records, report = build_baseline_records(
        drafts,
        jobs,
        responses,
        mode="critic-accepted",
        train_ratio=0.6,
        split_seed="test",
    )

    assert len(records) == 5
    assert {row["review_status"] for row in records} == {"critic-accepted"}
    assert {row["split"] for row in records} == {
        "experimental-train",
        "experimental-validation",
    }
    assert report["critic_recommendation_counts"] == {"accept": 5, "repair": 1}


def test_repaired_mode_uses_revised_messages_without_claiming_human_review():
    drafts, jobs, responses, repaired = build_inputs()

    records, _ = build_baseline_records(
        drafts,
        jobs,
        responses,
        mode="critic-accepted-and-repaired",
        train_ratio=0.6,
        split_seed="test",
    )

    repaired_row = next(row for row in records if row["id"] == "row-006")
    assert repaired_row["messages"] == repaired
    assert repaired_row["review_status"] == "critic-repaired"
    assert repaired_row["training_tier"] == "experimental-critic-filtered"
    assert len(repaired_row["original_messages_sha256"]) == 64


def test_duplicate_responses_stay_in_one_split():
    drafts, jobs, responses, _ = build_inputs()
    drafts[0]["messages"][1]["content"] = "Muraho means hello."
    drafts[4]["messages"][1]["content"] = "Muraho means hello!"

    records, _ = build_baseline_records(
        drafts,
        jobs,
        responses,
        mode="critic-accepted",
        train_ratio=0.5,
        split_seed="test",
    )
    split_by_id = {row["id"]: row["split"] for row in records}

    assert split_by_id["row-001"] == split_by_id["row-005"]


def test_invalid_critic_accept_gate_is_rejected():
    drafts = [draft("row-001")]
    jobs = [critic_job("row-001")]
    response = critic_response("row-001")
    assessment = json.loads(response["output_text"])
    assessment["scores"]["correctness"] = 3
    response["output_text"] = json.dumps(assessment)

    with pytest.raises(ValueError, match="invalid accept gate"):
        build_baseline_records(
            drafts,
            jobs,
            [response],
            mode="critic-accepted",
            train_ratio=0.5,
            split_seed="test",
        )

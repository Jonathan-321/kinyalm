from collections import Counter

import pytest

from kinyalm.data.human_reviewed import (
    DATASET_ID,
    build_dataset,
    prepare_existing_rows,
    prepare_tessy_rows,
)


def conversation(row_id, task_type="dialogue", **overrides):
    row = {
        "id": row_id,
        "task_type": task_type,
        "split": "train",
        "source": "team-authored",
        "source_status": "team-authored",
        "review_status": "approved",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": f"Question {row_id}"},
            {"role": "assistant", "content": f"Answer {row_id}"},
        ],
        "reviewer_notes": "Reviewer: Test Reviewer.",
        "curation_tier": "human-and-critic-agreed",
    }
    row.update(overrides)
    return row


def test_prepare_tessy_rows_preserves_full_conversation_and_source_metadata():
    converted = [
        conversation(
            "row-001",
            messages=[
                {"role": "user", "content": "Muraho."},
                {"role": "assistant", "content": "Muraho neza."},
                {"role": "user", "content": "Amakuru?"},
                {"role": "assistant", "content": "Ni meza."},
            ],
        )
    ]
    reviews = [
        {
            "conversation_id": "row-001",
            "priority": "Critic accepted",
        }
    ]
    drafts = [
        {
            "id": "row-001",
            "language_mix": "kinyarwanda",
            "task_family": "conversation-practice",
            "difficulty": "beginner",
            "evaluation_categories": ["multi-turn-conversation"],
        }
    ]

    rows = prepare_tessy_rows(converted, reviews, drafts)

    assert len(rows[0]["messages"]) == 4
    assert rows[0]["language_mix"] == "kinyarwanda"
    assert rows[0]["curation_tier"] == "human-and-critic-agreed"


def test_quality_gate_requires_explicit_critic_disagreement_decision():
    agreed = conversation("agreed-001")
    disputed = conversation(
        "disputed-001",
        curation_tier="human-approved-critic-disputed",
    )

    with pytest.raises(ValueError, match="quality-gated pool has 1 records"):
        build_dataset([[agreed, disputed]], minimum_records=2)

    rows, report = build_dataset(
        [[agreed, disputed]],
        minimum_records=2,
        include_critic_disagreements=True,
        train_ratio=0.5,
    )

    assert len(rows) == 2
    assert report["curation_tier_counts"] == {
        "human-and-critic-agreed": 1,
        "human-approved-critic-disputed": 1,
    }


def test_split_is_exact_stratified_and_keeps_complete_conversations():
    rows = [
        conversation(f"dialogue-{index:03d}", "dialogue")
        for index in range(20)
    ] + [
        conversation(f"uncertainty-{index:03d}", "uncertainty")
        for index in range(10)
    ]

    built, report = build_dataset(
        [rows],
        minimum_records=30,
        train_ratio=0.8,
    )

    assert report["split_counts"] == {"train": 24, "validation": 6}
    validation_tasks = Counter(
        row["task_type"] for row in built if row["split"] == "validation"
    )
    assert validation_tasks == {"dialogue": 4, "uncertainty": 2}
    assert all(row["dataset_version"] == DATASET_ID for row in built)


def test_duplicate_conversations_are_rejected_even_with_different_ids():
    first = conversation("row-001")
    second = conversation(
        "row-002",
        messages=first["messages"],
    )

    with pytest.raises(ValueError, match="duplicate normalized conversations"):
        build_dataset([[first, second]], minimum_records=2)


def test_prepare_existing_rows_rejects_nonapproved_input():
    with pytest.raises(ValueError, match="not human-approved"):
        prepare_existing_rows(
            [conversation("row-001", review_status="needs-review")],
            tier="human-reviewed-foundation",
            reviewer="Reviewer",
        )

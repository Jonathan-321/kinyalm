from scripts.prepare_hf_candidate_sft import (
    build_candidate_splits,
    unsupported_script_characters,
)


def candidate(row_id: str, family: str, *, flagged: bool = False) -> dict:
    row = {
        "id": row_id,
        "task_type": "dialogue",
        "task_family": family,
        "split": "candidate",
        "source": "synthetic-distillation",
        "source_status": "model-generated",
        "review_status": "candidate-unreviewed",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": f"Ikibazo {row_id}"},
            {"role": "assistant", "content": f"Igisubizo {row_id}"},
        ],
        "reviewer_notes": "Native-speaker review pending.",
    }
    if flagged:
        row["candidate_flags"] = ["teacher-self-check-failed:natural_kinyarwanda"]
    return row


def test_candidate_split_is_exact_stratified_and_reproducible() -> None:
    rows = [
        candidate(f"row-{family}-{index:03d}", family, flagged=index == 0)
        for family in ("conversation", "translation")
        for index in range(20)
    ]

    first, report = build_candidate_splits(
        rows,
        split_seed="split-v1",
        include_flagged=True,
    )
    second, _ = build_candidate_splits(
        rows,
        split_seed="split-v1",
        include_flagged=True,
    )

    assert report["selected_rows"] == 40
    assert report["flagged_rows"] == 2
    assert report["split_counts"] == {
        "experimental-test": 2,
        "experimental-train": 36,
        "experimental-validation": 2,
    }
    assert report["task_family_split_counts"]["conversation"] == {
        "experimental-test": 1,
        "experimental-train": 18,
        "experimental-validation": 1,
    }
    assert [(row["id"], row["split"]) for row in first] == [
        (row["id"], row["split"]) for row in second
    ]


def test_candidate_split_can_exclude_flagged_rows() -> None:
    rows = [
        candidate(f"row-{index:03d}", "conversation", flagged=index == 0)
        for index in range(20)
    ]

    selected, report = build_candidate_splits(
        rows,
        split_seed="split-v1",
        include_flagged=False,
    )

    assert len(selected) == 19
    assert report["flagged_rows"] == 0
    assert report["rejection_counts"] == {"candidate-flagged": 1}


def test_strict_policy_rejects_mixed_script_generation() -> None:
    rows = [
        candidate(f"row-{index:03d}", "conversation-practice")
        for index in range(20)
    ]
    rows[0]["messages"][1]["content"] = "Iyo nteruro irak дұрыс."

    selected, report = build_candidate_splits(
        rows,
        split_seed="split-v1",
        include_flagged=False,
        quality_policy="strict-script-clean",
    )

    assert len(selected) == 19
    assert unsupported_script_characters(rows[0]) == set("дұрыс")
    assert report["rejection_counts"] == {"unsupported-script-character": 1}


def test_core_policy_keeps_only_direct_tutoring_families() -> None:
    rows = [
        candidate(f"core-{index:03d}", "conversation-practice")
        for index in range(20)
    ] + [
        candidate(f"risk-{index:03d}", "ambiguity-and-hallucination-resistance")
        for index in range(20)
    ]

    selected, report = build_candidate_splits(
        rows,
        split_seed="split-v1",
        include_flagged=False,
        quality_policy="core-direct",
    )

    assert len(selected) == 20
    assert {row["task_family"] for row in selected} == {"conversation-practice"}
    assert report["rejection_counts"] == {
        "excluded-task-family:ambiguity-and-hallucination-resistance": 20
    }

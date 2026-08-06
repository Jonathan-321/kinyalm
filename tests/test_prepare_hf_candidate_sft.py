from scripts.prepare_hf_candidate_sft import build_candidate_splits


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

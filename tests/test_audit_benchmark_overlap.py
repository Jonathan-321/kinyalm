import json

from scripts.audit_benchmark_overlap import audit_overlap, normalize_text, similarity


def write_task_bank(path):
    path.write_text(
        """# Test bank

| ID | Category | Split | Learner Prompt | Review Focus |
| --- | --- | --- | --- | --- |
| T1001 | Translation | benchmark-only | Translate: I am a student. | accuracy |
| T1002 | Conversation | benchmark-only | Greet a teacher politely. | register |
""",
        encoding="utf-8",
    )


def test_normalization_and_similarity_are_case_and_punctuation_insensitive():
    assert normalize_text("Muraho, NEZA!") == "muraho neza"
    assert similarity("Muraho, neza!", "muraho neza")["token_jaccard"] == 1.0


def test_overlap_audit_checks_later_user_turns(tmp_path):
    task_bank = tmp_path / "tasks.md"
    write_task_bank(task_bank)
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "row-1",
                "split": "train",
                "task_family": "translation",
                "messages": [
                    {"role": "user", "content": "Unrelated opening"},
                    {"role": "assistant", "content": "Answer"},
                    {"role": "user", "content": "Translate: I am a student."},
                    {"role": "assistant", "content": "Ndi umunyeshuri."},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_overlap(task_bank=task_bank, dataset_paths=[dataset])

    assert report["benchmark_task_count"] == 2
    assert report["training_user_turn_count"] == 2
    assert report["exact_match_count"] == 1
    match = next(
        row for row in report["closest_matches"] if row["task_id"] == "T1001"
    )
    assert match["message_index"] == 2

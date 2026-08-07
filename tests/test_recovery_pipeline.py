import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinyalm.evaluation import (
    compare_probe_repetition,
    load_bakeoff_config,
    load_task_bank,
    summarize_native_review,
    write_blind_review_pack,
)
from scripts.build_targeted_rewrite_queue import build_rewrite_rows
from scripts.cloud.submit_recovery_arm import load_arm
from scripts.download_reviewed_sft import verify_package
from scripts.promote_targeted_rewrites import promote_rewrites
from scripts.publish_training_run import build_run_metadata
from scripts.train_qlora import parse_checkpoint_steps, parse_target_modules

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/gemma4_recovery_bakeoff.json"


def test_recovery_bank_has_150_unique_permanently_held_out_prompts():
    config = load_bakeoff_config(CONFIG)
    tasks = load_task_bank(ROOT / config.task_bank)

    assert config.expected_task_count == 150
    assert len(tasks) == 150
    assert {task.split for task in tasks} == {"benchmark-only"}
    assert len({task.prompt.casefold() for task in tasks}) == 150
    assert sum(task.category == "Morphology and grammar" for task in tasks) == 30
    assert sum(task.category == "Sentence correction" for task in tasks) == 20


def test_blind_pack_contains_native_correction_fields(tmp_path):
    task = load_task_bank(ROOT / load_bakeoff_config(CONFIG).task_bank)[0]
    review_path = tmp_path / "review.csv"
    key_path = tmp_path / "key.json"

    write_blind_review_pack(
        output_csv=review_path,
        key_path=key_path,
        tasks=[task],
        candidate_results={
            "base": {
                task.id: {
                    "status": "ok",
                    "response": "Igisubizo.",
                    "model_id": "private/base",
                    "model_revision": "a" * 40,
                }
            }
        },
        seed=1,
    )

    row = next(csv.DictReader(review_path.open(encoding="utf-8")))
    assert row["prompt_validity"] == ""
    assert row["repetition_flag"] == ""
    assert row["failure_tags"] == ""
    assert row["rewrite_priority"] == ""
    assert row["corrected_response"] == ""
    assert "private/base" not in review_path.read_text(encoding="utf-8")


def _scored_row(blind_id, candidate, *, passed, morphology=True):
    del candidate
    score = "4" if passed else "2"
    return {
        "blind_id": blind_id,
        "model_label": "hidden",
        "task_id": blind_id,
        "category": "Morphology and grammar" if morphology else "Vocabulary and usage",
        "prompt": "Held-out prompt",
        "review_focus": "accuracy",
        "response": "Model response",
        "prompt_validity": "valid",
        "kinyarwanda_correctness_1_5": score,
        "beginner_clarity_1_5": score,
        "grammar_explanation_1_5": score,
        "cultural_register_1_5": score,
        "helpfulness_1_5": score,
        "uncertainty_behavior_1_5": score,
        "hallucination_flag": "no" if passed else "yes",
        "repetition_flag": "no",
        "pass_fail": "pass" if passed else "fail",
        "failure_tags": "wrong-morphology" if not passed else "",
        "rewrite_priority": "high" if not passed else "none",
        "corrected_response": "Native correction" if not passed else "",
        "reviewer": "Native Reviewer",
        "reviewer_notes": "checked",
    }


def _write_review_fixture(tmp_path, rows, candidate_by_id):
    review_path = tmp_path / "review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    key_path = tmp_path / "key.json"
    key_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "blind_id": blind_id,
                        "candidate_id": candidate,
                        "model_id": f"private/{candidate}",
                        "model_revision": "a" * 40,
                    }
                    for blind_id, candidate in candidate_by_id.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    return review_path, key_path


def test_native_review_rejects_adapter_below_base_and_recommends_cpt(tmp_path):
    base_row = _scored_row("B001", "base", passed=False)
    for column in (
        "kinyarwanda_correctness_1_5",
        "beginner_clarity_1_5",
        "grammar_explanation_1_5",
        "cultural_register_1_5",
        "helpfulness_1_5",
        "uncertainty_behavior_1_5",
    ):
        base_row[column] = "3"
    rows = [base_row, _scored_row("B002", "adapter", passed=False)]
    review_path, key_path = _write_review_fixture(
        tmp_path, rows, {"B001": "base", "B002": "adapter"}
    )

    summary = summarize_native_review(
        review_path,
        key_path,
        baseline_candidate_id="base",
    )

    assert summary["complete"] is True
    assert summary["candidate_decisions"]["adapter"]["decision"] == "reject"
    assert summary["continued_pretraining"]["decision"] == "consider-cpt"


def test_rewrite_queue_is_blank_and_does_not_copy_held_out_text():
    review = _scored_row("B001", "base", passed=False)
    review["prompt"] = "Secret held-out prompt"
    review["response"] = "Bad response"
    key = {"rows": [{"blind_id": "B001", "candidate_id": "base"}]}

    rows = build_rewrite_rows(
        [review],
        key,
        candidate_id="base",
        target_rows=500,
    )

    assert len(rows) == 500
    assert all(row["new_user_prompt"] == "" for row in rows)
    assert all(row["gold_assistant_response"] == "" for row in rows)
    serialized = json.dumps(rows)
    assert "Secret held-out prompt" not in serialized
    assert "Bad response" not in serialized
    assert "Native correction" not in serialized


def test_promoter_accepts_500_unique_native_rows_and_groups_splits():
    rows = []
    for index in range(500):
        rows.append(
            {
                "rewrite_id": f"native-row-{index:04d}",
                "source_task_id": f"source-{index // 5:03d}",
                "task_type": "grammar-explanation",
                "language_mix": "kinyarwanda+english",
                "new_user_prompt": (
                    f"Fresh learner scenario number {index} about agreement."
                ),
                "gold_assistant_response": f"Native approved answer number {index}.",
                "review_status": "approved",
                "reviewer": "Native Reviewer",
                "reviewer_notes": "rewritten",
                "failure_tags": "agreement",
            }
        )

    records = promote_rewrites(
        rows,
        ["Unrelated permanently held-out sentence."],
        minimum_approved=500,
        maximum_approved=1000,
        train_ratio=0.9,
        split_seed="test",
    )

    assert len(records) == 500
    assert {row["split"] for row in records} == {"train", "validation"}
    split_by_source = {}
    for row in records:
        split_by_source.setdefault(row["source_group_id"], row["split"])
        assert split_by_source[row["source_group_id"]] == row["split"]


def test_promoter_rejects_held_out_prompt_copy():
    row = {
        "rewrite_id": "native-row-0001",
        "source_task_id": "T1001",
        "task_type": "greeting",
        "language_mix": "kinyarwanda",
        "new_user_prompt": "Explain when to use Muraho.",
        "gold_assistant_response": "Native answer.",
        "review_status": "approved",
        "reviewer": "Native Reviewer",
        "reviewer_notes": "checked",
        "failure_tags": "register",
    }

    with pytest.raises(ValueError, match="too similar"):
        promote_rewrites(
            [row],
            ["Explain when to use Muraho."],
            minimum_approved=1,
            maximum_approved=1,
            train_ratio=0.9,
            split_seed="test",
        )


def test_recovery_lora_and_checkpoint_parsers_are_strict():
    assert parse_target_modules("q_proj,v_proj") == ("q_proj", "v_proj")
    assert parse_checkpoint_steps("25,50,100") == (25, 50, 100)
    with pytest.raises(Exception, match="duplicates"):
        parse_target_modules("q_proj,q_proj")
    with pytest.raises(Exception, match="increasing"):
        parse_checkpoint_steps("50,25")


def test_recovery_arm_config_matches_requested_matrix():
    config_path = ROOT / "configs/training/gemma4_recovery_arms.json"
    config, first = load_arm(config_path, "qv-r8-lr2e6")
    _, second = load_arm(config_path, "qv-r8-lr5e6")
    _, third = load_arm(config_path, "qvo-r8-lr2e6")

    assert config["shared"]["checkpoint_steps"] == [25, 50, 100]
    assert first["target_modules"] == ["q_proj", "v_proj"]
    assert first["lora_r"] == 8 and first["learning_rate"] == 2e-6
    assert second["learning_rate"] == 5e-6
    assert third["target_modules"] == ["q_proj", "v_proj", "o_proj"]


def _sft_row(row_id, split):
    return {
        "id": row_id,
        "task_type": "greeting",
        "split": split,
        "source": "native-recovery-rewrite-v1",
        "source_status": "team-authored",
        "review_status": "approved",
        "language_mix": "kinyarwanda",
        "messages": [
            {"role": "user", "content": "Muraho."},
            {"role": "assistant", "content": "Muraho neza."},
        ],
        "reviewer_notes": "Reviewer: Native Reviewer.",
    }


def test_downloaded_reviewed_package_requires_hash_matched_human_data(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(
        json.dumps(_sft_row("native-row-1", "train")) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(_sft_row("native-row-2", "validation")) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_tier": "human-reviewed-recovery-sft",
        "human_reviewed": True,
        "training_eligible": True,
        "outputs": {
            "train": {
                "rows": 1,
                "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            },
            "validation": {
                "rows": 1,
                "sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            },
        },
    }
    (tmp_path / "dataset-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    paths = verify_package(tmp_path, minimum_rows=2, maximum_rows=2)

    assert [path.name for path in paths] == [
        "dataset-manifest.json",
        "train.jsonl",
        "validation.jsonl",
    ]


def test_publication_metadata_preserves_selected_checkpoints(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    for name in (
        "adapter_config.json",
        "run-preflight.json",
        "adapter_model.safetensors",
    ):
        (adapter_dir / name).write_text("{}", encoding="utf-8")
    for step in (25, 100):
        checkpoint = adapter_dir / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter_model.safetensors").write_text(
            f"step {step}", encoding="utf-8"
        )
    gate_dir = adapter_dir / "quality-gate"
    gate_dir.mkdir()
    (gate_dir / "summary.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "dataset_tier": "human-reviewed-recovery-sft",
                "human_reviewed": True,
                "production_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    training_log = tmp_path / "train.log"
    system_info = tmp_path / "system-info.txt"
    training_log.write_text("trained", encoding="utf-8")
    system_info.write_text("system", encoding="utf-8")
    args = SimpleNamespace(
        adapter_dir=str(adapter_dir),
        dataset_manifest=str(dataset_manifest),
        training_log=str(training_log),
        system_info=str(system_info),
        run_id="recovery",
        base_model="google/gemma-4-12B-it",
        base_model_revision="a" * 40,
        dataset_repo="kinyalm/kinyalm-data-lake",
        dataset_revision="b" * 40,
        checkpoint_steps="25,50,100",
    )

    metadata = build_run_metadata(args)

    assert [row["step"] for row in metadata["preserved_checkpoints"]] == [25, 100]
    assert metadata["quality_gate"]["passed"] is True


def test_repetition_gate_rejects_new_loop(tmp_path):
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    base.write_text(
        json.dumps({"prompt": "P", "completion": "This answer is short."}) + "\n",
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "prompt": "P",
                "completion": "one two three four " * 6,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = compare_probe_repetition(base, candidate)

    assert report["passed"] is False
    assert report["new_severe_repetition_rows"] == 1

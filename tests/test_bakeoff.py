import csv
import json
from pathlib import Path

import pytest

from kinyalm.evaluation import (
    TutorTask,
    append_result,
    latest_results,
    load_bakeoff_config,
    write_blind_review_pack,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "gemma4_bakeoff.json"


def test_project_bakeoff_config_is_pinned_and_held_out():
    config = load_bakeoff_config(CONFIG)

    assert config.task_split == "benchmark-only"
    assert config.expected_task_count == 26
    assert config.enable_thinking is False
    assert [candidate.id for candidate in config.candidates] == [
        "gemma4-12b-it",
        "gemma4-31b-it",
    ]
    assert all(len(candidate.revision) == 40 for candidate in config.candidates)


def test_config_rejects_training_split(tmp_path: Path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["task_split"] = "train-template"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must be benchmark-only"):
        load_bakeoff_config(path)


def test_latest_results_uses_last_attempt(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    append_result(path, {"task_id": "T001", "status": "error"})
    append_result(path, {"task_id": "T001", "status": "ok", "response": "Muraho"})

    assert latest_results(path)["T001"]["status"] == "ok"


def test_blind_pack_excludes_model_identity(tmp_path: Path):
    tasks = [
        TutorTask(
            id="T001",
            category="Greeting",
            split="benchmark-only",
            prompt="Explain Muraho.",
            review_focus="accuracy",
        )
    ]
    candidate_results = {
        "small": {
            "T001": {
                "status": "ok",
                "response": "Small answer",
                "model_id": "organization/secret-small-model",
                "model_revision": "a" * 40,
            }
        },
        "large": {
            "T001": {
                "status": "ok",
                "response": "Large answer",
                "model_id": "organization/secret-large-model",
                "model_revision": "b" * 40,
            }
        },
    }
    csv_path = tmp_path / "review.csv"
    key_path = tmp_path / "key.json"

    count, labels = write_blind_review_pack(
        output_csv=csv_path,
        key_path=key_path,
        tasks=tasks,
        candidate_results=candidate_results,
        seed=7,
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    review_text = csv_path.read_text(encoding="utf-8")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert count == 2
    assert len(labels) == 2
    assert {row["model_label"] for row in rows} == {"Model A", "Model B"}
    assert "secret-small-model" not in review_text
    assert "secret-large-model" not in review_text
    assert key["labels"] == labels

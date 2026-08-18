import csv
import json
from pathlib import Path

import pytest

from scripts.build_combined_continuation_review import (
    RAW_SOURCES,
    build_combined_review,
)


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_bank = tmp_path / "task-bank.md"
    task_bank.write_text(
        "\n".join(
            [
                "| ID | Category | Split | Learner Prompt | Review Focus |",
                "|---|---|---|---|---|",
                "| T001 | Greeting | benchmark-only | Muraho? | accuracy |",
                "| T002 | Translation | benchmark-only | Translate hello. | meaning |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_name": "test",
                "task_bank": str(task_bank),
                "task_split": "benchmark-only",
                "expected_task_count": 2,
                "system_prompt": "Tutor prompt",
                "seed": 7,
                "max_new_tokens": 32,
                "enable_thinking": False,
                "candidates": [
                    {"id": "a", "model_id": "org/a", "revision": "a" * 40},
                    {"id": "b", "model_id": "org/b", "revision": "b" * 40},
                ],
            }
        ),
        encoding="utf-8",
    )

    run_root = tmp_path / "run"
    for index, (_, relative_path) in enumerate(RAW_SOURCES):
        path = run_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate_id = f"candidate-{index}"
        rows = []
        for task_id, prompt, category, focus in (
            ("T001", "Muraho?", "Greeting", "accuracy"),
            ("T002", "Translate hello.", "Translation", "meaning"),
        ):
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "status": "ok",
                    "response": f"answer-{index}-{task_id}",
                    "prompt": prompt,
                    "category": category,
                    "review_focus": focus,
                    "split": "benchmark-only",
                    "system_prompt": "Tutor prompt",
                    "seed": 7,
                    "max_new_tokens": 32,
                    "enable_thinking": False,
                    "model_id": "mlx/base",
                    "model_revision": "c" * 40,
                    "inference_backend": "mlx",
                    "inference_backend_version": "0.31.3",
                    "quantization": "4-bit",
                    "adapter_id": None if index == 0 else f"org/adapter-{index}",
                    "adapter_revision": None if index == 0 else str(index) * 40,
                    "adapter_sha256": None if index == 0 else str(index) * 64,
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    return run_root, config


def test_builds_one_four_way_blind_pack(tmp_path: Path, monkeypatch):
    run_root, config = write_fixture(tmp_path)
    output_dir = run_root / "combined"
    monkeypatch.setattr(
        "scripts.build_combined_continuation_review.ROOT",
        tmp_path,
    )

    manifest = build_combined_review(
        run_root=run_root,
        config_path=config,
        output_dir=output_dir,
    )

    rows = list(
        csv.DictReader(
            (output_dir / "review/blind-review.csv").open(encoding="utf-8")
        )
    )
    assert len(rows) == 8
    assert {row["model_label"] for row in rows} == {
        "Model A",
        "Model B",
        "Model C",
        "Model D",
    }
    assert manifest["candidate_count"] == 4
    assert manifest["promotion_eligible"] is False
    assert len(list((output_dir / "raw").glob("*.jsonl"))) == 4


def test_rejects_incomplete_candidate(tmp_path: Path, monkeypatch):
    run_root, config = write_fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.build_combined_continuation_review.ROOT",
        tmp_path,
    )
    _, relative_path = RAW_SOURCES[-1]
    path = run_root / relative_path
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n")

    with pytest.raises(ValueError, match="task IDs do not match"):
        build_combined_review(
            run_root=run_root,
            config_path=config,
            output_dir=run_root / "combined",
        )

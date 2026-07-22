from pathlib import Path

import pytest

from kinyalm.evaluation import benchmark_tasks, load_task_bank

ROOT = Path(__file__).resolve().parents[1]
TASK_BANK = ROOT / "docs" / "evaluation" / "learning-task-bank.md"


def test_project_task_bank_has_expected_held_out_split():
    tasks = load_task_bank(TASK_BANK)
    held_out = benchmark_tasks(tasks)

    assert len(tasks) == 50
    assert len(held_out) == 26
    assert {task.split for task in held_out} == {"benchmark-only"}
    assert held_out[0].id == "T001"
    assert held_out[-1].id == "T050"


def test_task_bank_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "tasks.md"
    path.write_text(
        """| ID | Category | Split | Learner Prompt | Review Focus |
| --- | --- | --- | --- | --- |
| T001 | Greeting | benchmark-only | Hello | accuracy |
| T001 | Greeting | train-template | Hi | accuracy |
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate task id"):
        load_task_bank(path)


def test_task_bank_rejects_unknown_split(tmp_path: Path):
    path = tmp_path / "tasks.md"
    path.write_text(
        """| ID | Category | Split | Learner Prompt | Review Focus |
| --- | --- | --- | --- | --- |
| T001 | Greeting | validation | Hello | accuracy |
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported split"):
        load_task_bank(path)

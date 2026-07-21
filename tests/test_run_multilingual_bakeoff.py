from pathlib import Path

import pytest

from kinyalm.evaluation import load_bakeoff_config
from scripts.run_multilingual_bakeoff import (
    load_held_out_tasks,
    select_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "gemma4_bakeoff.json"


def test_runner_loads_only_held_out_tasks():
    config = load_bakeoff_config(CONFIG)

    _, tasks = load_held_out_tasks(config)

    assert len(tasks) == 26
    assert {task.split for task in tasks} == {"benchmark-only"}


def test_runner_selects_candidates_in_config_order():
    config = load_bakeoff_config(CONFIG)

    selected = select_candidates(config, ["gemma4-31b-it", "gemma4-12b-it"])

    assert [candidate.id for candidate in selected] == [
        "gemma4-12b-it",
        "gemma4-31b-it",
    ]


def test_runner_rejects_unknown_candidate():
    config = load_bakeoff_config(CONFIG)

    with pytest.raises(ValueError, match="unknown candidate"):
        select_candidates(config, ["not-a-model"])

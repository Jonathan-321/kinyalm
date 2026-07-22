from pathlib import Path

import pytest

from kinyalm.evaluation import load_bakeoff_config
from scripts.run_multilingual_bakeoff import (
    filter_ignored_weights,
    load_held_out_tasks,
    parse_gemma4_response,
    resolve_runtime_candidates,
    select_candidates,
    select_tasks,
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


def test_runner_resolves_pinned_local_mlx_checkpoint():
    config = load_bakeoff_config(CONFIG)

    candidates = resolve_runtime_candidates(config, None, "mlx")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.id == "gemma4-12b-it-qat-4bit-mlx"
    assert candidate.source_model_id == "google/gemma-4-12B-it"
    assert candidate.backend_version == "0.31.3"
    assert candidate.model_type_override == "gemma4"
    assert candidate.ignored_weight_prefixes == ("vision_embedder.",)
    assert candidate.suppress_token_ids == (258882, 258883)


def test_runner_rejects_candidate_without_local_mlx_runtime():
    config = load_bakeoff_config(CONFIG)

    with pytest.raises(ValueError, match="no pinned local MLX runtime"):
        resolve_runtime_candidates(config, ["gemma4-31b-it"], "mlx")


def test_select_tasks_preserves_bank_order_and_validates_limit():
    config = load_bakeoff_config(CONFIG)
    _, tasks = load_held_out_tasks(config)

    selected = select_tasks(tasks, ["T050", "T001"], 1)

    assert [task.id for task in selected] == ["T001"]
    with pytest.raises(ValueError, match="limit must be positive"):
        select_tasks(tasks, None, 0)


def test_parse_gemma4_response_hides_thought_channel():
    response, thinking = parse_gemma4_response(
        "<|channel>thought\ninternal notes<channel|>Muraho neza.<turn|>"
    )

    assert response == "Muraho neza."
    assert thinking == "internal notes"


def test_filter_ignored_weights_removes_only_pinned_prefixes():
    weights = {
        "language_model.layer.weight": 1,
        "vision_embedder.patch.weight": 2,
    }

    filtered = filter_ignored_weights(weights, ("vision_embedder.",))

    assert filtered == {"language_model.layer.weight": 1}
    with pytest.raises(ValueError, match="were not present"):
        filter_ignored_weights(weights, ("not_in_checkpoint.",))

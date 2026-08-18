import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from kinyalm.evaluation import load_bakeoff_config
from scripts.run_multilingual_bakeoff import (
    add_mlx_adapter_candidate,
    apply_adapter_variant,
    attach_transformers_adapter,
    filter_ignored_weights,
    load_held_out_tasks,
    parse_gemma4_response,
    resolve_runtime_candidates,
    select_candidates,
    select_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "gemma4_bakeoff.json"
CHECKPOINT_CONFIG = (
    ROOT / "configs" / "evaluation" / "gemma4_longform_checkpoint_bakeoff.json"
)


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


def test_runner_resolves_pinned_transformers_adapter_checkpoints():
    config = load_bakeoff_config(CHECKPOINT_CONFIG)

    candidates = resolve_runtime_candidates(config, None, "transformers")

    assert len(candidates) == 6
    assert candidates[0].adapter_id is None
    assert candidates[1].adapter_subfolder == "checkpoints/checkpoint-400"
    assert candidates[-1].adapter_subfolder is None
    assert candidates[-1].adapter_revision == "44f584225fb3cc215a52be69fcb107da2e743643"


def test_transformers_adapter_loader_pins_revision_and_subfolder(monkeypatch):
    config = load_bakeoff_config(CHECKPOINT_CONFIG)
    candidate = resolve_runtime_candidates(
        config, ["kinyalm-longform-step-400"], "transformers"
    )[0]
    calls = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, repo_id, **options):
            calls.append((model, repo_id, options))
            return "adapted-model"

    monkeypatch.setitem(
        sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel)
    )

    result = attach_transformers_adapter("base-model", candidate)

    assert result == "adapted-model"
    assert calls == [
        (
            "base-model",
            "kinyalm/kinyalm-gemma-4-12b-longform-fullepoch-qv-r8-lr2e5",
            {
                "revision": "44f584225fb3cc215a52be69fcb107da2e743643",
                "is_trainable": False,
                "subfolder": "checkpoints/checkpoint-400",
            },
        )
    ]


def test_runner_rejects_candidate_without_local_mlx_runtime():
    config = load_bakeoff_config(CONFIG)

    with pytest.raises(ValueError, match="no pinned local MLX runtime"):
        resolve_runtime_candidates(config, ["gemma4-31b-it"], "mlx")


def test_runner_adds_hash_verified_mlx_adapter_candidate(tmp_path: Path):
    config = load_bakeoff_config(CONFIG)
    candidates = resolve_runtime_candidates(config, ["gemma4-12b-it"], "mlx")
    adapter_dir = tmp_path / "adapter-mlx"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter_dir / "adapters.safetensors"
    weights.write_bytes(b"pinned adapter")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "base": {
            "repo_id": candidates[0].model_id,
            "revision": candidates[0].revision,
        },
        "adapter": {
            "repo_id": "kinyalm/kinyalm-gemma-4-12b-sft10k-v1",
            "revision": "7" * 40,
            "path": str(adapter_dir),
            "conversion": {"converted_sha256": digest},
        },
    }
    manifest_path = tmp_path / "runtime.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = add_mlx_adapter_candidate(candidates, manifest_path)

    assert [candidate.id for candidate in resolved] == [
        "gemma4-12b-it-qat-4bit-mlx",
        "kinyalm-gemma-4-12b-sft10k-v1-mlx",
    ]
    assert resolved[0].adapter_path is None
    assert resolved[1].adapter_id == manifest["adapter"]["repo_id"]
    assert resolved[1].adapter_revision == "7" * 40
    assert resolved[1].adapter_sha256 == digest


def test_runner_rejects_changed_mlx_adapter_weights(tmp_path: Path):
    config = load_bakeoff_config(CONFIG)
    candidates = resolve_runtime_candidates(config, ["gemma4-12b-it"], "mlx")
    adapter_dir = tmp_path / "adapter-mlx"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(b"changed")
    manifest_path = tmp_path / "runtime.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base": {
                    "repo_id": candidates[0].model_id,
                    "revision": candidates[0].revision,
                },
                "adapter": {
                    "repo_id": "kinyalm/adapter",
                    "revision": "7" * 40,
                    "path": str(adapter_dir),
                    "conversion": {"converted_sha256": "0" * 64},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash does not match"):
        add_mlx_adapter_candidate(candidates, manifest_path)


def test_runner_attaches_adapter_without_losing_base_identity():
    config = load_bakeoff_config(CONFIG)
    base = resolve_runtime_candidates(config, ["gemma4-12b-it"], "transformers")

    adapted = apply_adapter_variant(
        base,
        adapter_id="kinyalm/adapter",
        adapter_revision="d" * 40,
        run_as="original-peft",
    )

    assert adapted[0].id == "original-peft"
    assert adapted[0].model_id == "google/gemma-4-12B-it"
    assert adapted[0].revision == config.candidates[0].revision
    assert adapted[0].adapter_id == "kinyalm/adapter"
    assert adapted[0].adapter_revision == "d" * 40
    assert base[0].adapter_id is None


def test_runner_requires_unique_adapter_variant_id():
    config = load_bakeoff_config(CONFIG)
    base = resolve_runtime_candidates(config, ["gemma4-12b-it"], "transformers")

    with pytest.raises(ValueError, match="unique --run-as"):
        apply_adapter_variant(
            base,
            adapter_id="kinyalm/adapter",
            adapter_revision="d" * 40,
            run_as=None,
        )


def test_runner_requires_pinned_adapter_revision():
    config = load_bakeoff_config(CONFIG)
    base = resolve_runtime_candidates(config, ["gemma4-12b-it"], "transformers")

    with pytest.raises(ValueError, match="40-character"):
        apply_adapter_variant(
            base,
            adapter_id="kinyalm/adapter",
            adapter_revision=None,
            run_as="original-peft",
        )


def test_runner_rejects_unsafe_adapter_variant_id():
    config = load_bakeoff_config(CONFIG)
    base = resolve_runtime_candidates(config, ["gemma4-12b-it"], "transformers")

    with pytest.raises(ValueError, match="may contain only"):
        apply_adapter_variant(
            base,
            adapter_id="kinyalm/adapter",
            adapter_revision="d" * 40,
            run_as="../outside",
        )


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

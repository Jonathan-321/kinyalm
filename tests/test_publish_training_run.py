import argparse
import json

import pytest

from scripts.publish_training_run import (
    build_run_metadata,
    render_model_card,
    validate_publication_inputs,
)


def publication_args(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter_dir / "run-preflight.json").write_text("{}\n", encoding="utf-8")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    training_log = tmp_path / "train.log"
    training_log.write_text("loss=1.0\n", encoding="utf-8")
    system_info = tmp_path / "system-info.txt"
    system_info.write_text("gpu=A100\n", encoding="utf-8")
    return argparse.Namespace(
        adapter_dir=str(adapter_dir),
        dataset_manifest=str(dataset_manifest),
        training_log=str(training_log),
        system_info=str(system_info),
        repo_id="kinyalm/test-run",
        run_id="test-run-001",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        base_model_revision="a" * 40,
        dataset_repo="kinyalm/kinyalm-data-lake",
        dataset_revision="b" * 40,
    )


def test_publication_metadata_marks_run_experimental(tmp_path):
    args = publication_args(tmp_path)
    validate_publication_inputs(
        tmp_path / "adapter",
        tmp_path / "dataset-manifest.json",
        tmp_path / "train.log",
        tmp_path / "system-info.txt",
    )

    metadata = build_run_metadata(args)
    card = render_model_card(args, metadata)

    assert metadata["status"] == "experimental-critic-filtered"
    assert metadata["human_reviewed"] is False
    assert metadata["production_eligible"] is False
    assert len(metadata["artifacts"]) == 6
    assert "not production-eligible" in card
    assert args.base_model_revision in card


def test_publication_rejects_missing_adapter_weights(tmp_path):
    publication_args(tmp_path)
    (tmp_path / "adapter" / "adapter_model.safetensors").unlink()

    with pytest.raises(SystemExit, match=r"adapter_model\.\*"):
        validate_publication_inputs(
            tmp_path / "adapter",
            tmp_path / "dataset-manifest.json",
            tmp_path / "train.log",
            tmp_path / "system-info.txt",
        )

import hashlib
import json
import subprocess
import sys
from types import ModuleType

from scripts.train_qlora import (
    resolve_attention_implementation,
    verify_model_metadata,
)


def experimental_record(row_id, split):
    return {
        "id": row_id,
        "task_type": "dialogue",
        "split": split,
        "source": "synthetic-distillation",
        "source_status": "model-generated",
        "review_status": "critic-accepted",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": "Muraho."},
            {"role": "assistant", "content": "Muraho neza."},
        ],
        "reviewer_notes": "Model critic only.",
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_train_qlora_experimental_dry_run_writes_preflight(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    output_dir = tmp_path / "run"
    write_jsonl(
        train_path,
        [experimental_record("row-001", "experimental-train")],
    )
    write_jsonl(
        validation_path,
        [experimental_record("row-002", "experimental-validation")],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_qlora.py",
            "--train-file",
            str(train_path),
            "--eval-file",
            str(validation_path),
            "--output-dir",
            str(output_dir),
            "--experimental",
            "--attn-implementation",
            "sdpa",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "dry run complete" in result.stdout
    manifest = json.loads(
        (output_dir / "run-preflight.json").read_text(encoding="utf-8")
    )
    assert manifest["experimental"] is True
    assert manifest["data"]["train"]["rows"] == 1
    assert manifest["data"]["validation"]["rows"] == 1
    assert manifest["training"]["attention_implementation"] == "sdpa"


def test_train_qlora_requires_explicit_experimental_flag(tmp_path):
    train_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "run"
    write_jsonl(
        train_path,
        [experimental_record("row-001", "experimental-train")],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_qlora.py",
            "--train-file",
            str(train_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "no usable rows" in result.stderr


def test_train_qlora_rejects_data_that_does_not_match_manifest(tmp_path):
    train_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "dataset-manifest.json"
    write_jsonl(
        train_path,
        [experimental_record("row-001", "experimental-train")],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_tier": "experimental-critic-filtered",
                "human_reviewed": False,
                "production_eligible": False,
                "outputs": {
                    "train": {"rows": 1, "sha256": "0" * 64},
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_qlora.py",
            "--train-file",
            str(train_path),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--experimental",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert file_sha256(train_path) != "0" * 64
    assert result.returncode != 0
    assert "does not match dataset manifest sha256" in result.stderr


def test_attention_backend_only_special_cases_gemma_2():
    assert resolve_attention_implementation("google/gemma-2-9b-it", "auto") == "eager"
    assert resolve_attention_implementation("Qwen/Qwen2.5-7B-Instruct", "auto") is None
    assert resolve_attention_implementation("any/model", "sdpa") == "sdpa"


def test_verify_model_metadata_selects_multimodal_loader(monkeypatch):
    calls = []

    class FakeTokenizer:
        vocab_size = 262_144

    class FakeConfig:
        model_type = "gemma4_unified"
        _commit_hash = "abc123"

    class FakeModel:
        pass

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(("tokenizer", model, kwargs))
            return FakeTokenizer()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append(("config", model, kwargs))
            return FakeConfig()

    class FakeCausalAutoModel:
        _model_mapping = {}

    class FakeMultimodalAutoModel:
        _model_mapping = {FakeConfig: FakeModel}

    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "5.test"
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoModelForCausalLM = FakeCausalAutoModel
    fake_transformers.AutoModelForMultimodalLM = FakeMultimodalAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    metadata = verify_model_metadata("google/gemma-4-12B-it", "abc123")

    assert metadata == {
        "transformers_version": "5.test",
        "resolved_revision": "abc123",
        "model_type": "gemma4_unified",
        "model_class": "FakeModel",
        "tokenizer_class": "FakeTokenizer",
        "vocab_size": 262_144,
    }
    assert calls == [
        ("tokenizer", "google/gemma-4-12B-it", {"revision": "abc123"}),
        ("config", "google/gemma-4-12B-it", {"revision": "abc123"}),
    ]

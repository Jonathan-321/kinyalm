import hashlib
import json
import subprocess
import sys
from types import ModuleType

from scripts.train_qlora import (
    resolve_attention_implementation,
    to_prompt_completion_rows,
    tokenize_assistant_completion_rows,
    verify_model_metadata,
    write_generation_samples,
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
    assert manifest["data"]["train"]["supervised_assistant_turns"] == 1
    assert manifest["data"]["validation"]["supervised_assistant_turns"] == 1
    assert manifest["training"]["loss_scope"] == "assistant-completions-only"
    assert manifest["training"]["attention_implementation"] == "sdpa"


def test_prompt_completion_rows_preserve_history_and_mask_user_turns():
    record = experimental_record("row-001", "experimental-train")
    record["messages"].extend(
        [
            {"role": "user", "content": "Witwa nde?"},
            {"role": "assistant", "content": "Nitwa KinyaLM."},
        ]
    )

    examples = to_prompt_completion_rows([record])

    assert examples == [
        {
            "prompt": [{"role": "user", "content": "Muraho."}],
            "completion": [
                {"role": "assistant", "content": "Muraho neza."}
            ],
        },
        {
            "prompt": [
                {"role": "user", "content": "Muraho."},
                {"role": "assistant", "content": "Muraho neza."},
                {"role": "user", "content": "Witwa nde?"},
            ],
            "completion": [
                {"role": "assistant", "content": "Nitwa KinyaLM."}
            ],
        },
    ]


def test_tokenized_rows_preserve_generation_prefix_and_label_full_answer():
    calls = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages, kwargs))
            if kwargs["tokenize"]:
                return {"input_ids": [2, 10, 11, 12]}
            return (
                "<bos><user>Muraho.<assistant>"
                "KINYALM_ASSISTANT_COMPLETION_BOUNDARY_4C9E7A<turn-end>"
            )

        def __call__(self, text, **kwargs):
            calls.append(("tokenize", text, kwargs))
            assert text == "Muraho neza.<turn-end>"
            return {"input_ids": [21, 22, 1]}

    rows = tokenize_assistant_completion_rows(
        [experimental_record("row-001", "experimental-train")],
        FakeTokenizer(),
    )

    assert rows == [
        {
            "input_ids": [2, 10, 11, 12, 21, 22, 1],
            "labels": [-100, -100, -100, -100, 21, 22, 1],
        }
    ]
    template_calls = [call for call in calls if call[0] == "template"]
    assert all(call[2]["enable_thinking"] is False for call in template_calls)
    assert template_calls[0][2]["add_generation_prompt"] is True
    assert template_calls[1][2]["add_generation_prompt"] is False


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


def test_train_qlora_accepts_explicit_candidate_manifest(tmp_path):
    train_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "dataset-manifest.json"
    row = experimental_record("row-001", "experimental-train")
    row["review_status"] = "candidate-unreviewed"
    write_jsonl(train_path, [row])
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_tier": "experimental-candidate-unreviewed",
                "human_reviewed": False,
                "production_eligible": False,
                "outputs": {
                    "train": {
                        "rows": 1,
                        "sha256": file_sha256(train_path),
                    },
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

    assert result.returncode == 0, result.stderr
    assert "dry run complete" in result.stdout


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


def test_generation_samples_enable_and_restore_cache(tmp_path):
    calls = []

    class FakeConfig:
        use_cache = False

    class FakeInputIds:
        shape = (1, 2)

    class FakeBatch(dict):
        def to(self, device):
            calls.append(("to", device))
            return self

    class FakeModel:
        config = FakeConfig()
        device = "cpu"
        training = True

        def eval(self):
            calls.append(("eval",))

        def train(self):
            calls.append(("train",))

        def generate(self, **kwargs):
            calls.append(("generate", self.config.use_cache, kwargs["use_cache"]))
            return [[101, 102, 201, 202]]

    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages, kwargs))
            return FakeBatch(input_ids=FakeInputIds())

        def decode(self, tokens, **kwargs):
            calls.append(("decode", tokens, kwargs))
            return "Igisubizo."

    model = FakeModel()
    output_path = tmp_path / "samples.jsonl"

    write_generation_samples(
        model,
        FakeTokenizer(),
        ["Ikibazo."],
        output_path,
    )

    assert model.config.use_cache is False
    assert ("generate", True, True) in calls
    assert ("train",) in calls
    template_call = next(call for call in calls if call[0] == "template")
    assert template_call[2]["enable_thinking"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "prompt": "Ikibazo.",
        "completion": "Igisubizo.",
    }

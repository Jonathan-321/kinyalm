import hashlib
import json
import subprocess
import sys


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

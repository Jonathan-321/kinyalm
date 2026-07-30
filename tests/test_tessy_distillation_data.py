import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/sft/tessy-distill-review.train.jsonl"
VALIDATION_PATH = ROOT / "data/sft/tessy-distill-review.validation.jsonl"
EVAL_PROMPTS_PATH = ROOT / "data/sft/tessy-eval-prompts.txt"
UPLOAD_SCRIPT = ROOT / "scripts/push_tessy_contribution.sh"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_committed_splits_have_no_conversation_overlap():
    train_ids = {row["id"] for row in load_jsonl(TRAIN_PATH)}
    validation_ids = {row["id"] for row in load_jsonl(VALIDATION_PATH)}

    assert len(train_ids) == 258
    assert len(validation_ids) == 30
    assert train_ids.isdisjoint(validation_ids)


def test_committed_rows_keep_complete_conversations():
    rows = load_jsonl(TRAIN_PATH) + load_jsonl(VALIDATION_PATH)

    assert any(len(row["messages"]) > 2 for row in rows)
    assert all(len(row["messages"]) % 2 == 0 for row in rows)


def test_probe_prompts_do_not_copy_training_prompts():
    train_user_messages = {
        message["content"]
        for row in load_jsonl(TRAIN_PATH)
        for message in row["messages"]
        if message["role"] == "user"
    }
    eval_prompts = {
        line.strip()
        for line in EVAL_PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    assert train_user_messages.isdisjoint(eval_prompts)


def test_hf_upload_dry_run_stages_only_critic_agreed_rows(tmp_path):
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "HOME": str(tmp_path)})
    result = subprocess.run(
        ["bash", str(UPLOAD_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stage = (
        tmp_path
        / "KinyaLMData"
        / "hf_contributions"
        / "incoming"
        / "tessymugisha"
    )
    train = load_jsonl(stage / "sft-ready-critic-agreed-v1" / "train.jsonl")
    validation = load_jsonl(
        stage / "sft-ready-critic-agreed-v1" / "validation.jsonl"
    )
    readme = (stage / "README.md").read_text(encoding="utf-8")

    assert len(train) == 258
    assert len(validation) == 30
    assert "38 critic-disputed conversations" in readme
    assert "do not use it for the final training run" in readme

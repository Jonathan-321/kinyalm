import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/sft/tessy-distill-review.train.jsonl"
VALIDATION_PATH = ROOT / "data/sft/tessy-distill-review.validation.jsonl"
EVAL_PROMPTS_PATH = ROOT / "data/sft/tessy-eval-prompts.txt"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_committed_splits_have_no_conversation_overlap():
    train_ids = {row["id"] for row in load_jsonl(TRAIN_PATH)}
    validation_ids = {row["id"] for row in load_jsonl(VALIDATION_PATH)}

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

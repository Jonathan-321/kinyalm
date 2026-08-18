import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_longform_curriculum_counts_and_token_projection_are_consistent():
    config = json.loads(
        (ROOT / "configs/data/kinyalm_longform_curriculum_v1.json").read_text()
    )
    budget = config["token_budget"]
    shapes = config["conversation_shape"]

    conversations = sum(shapes.values())
    assistant_responses = sum(
        (int(message_count.split("_")[0]) // 2) * count
        for message_count, count in shapes.items()
    )
    assert conversations == budget["planned_new_conversations"]
    assert assistant_responses == budget["planned_new_assistant_responses"]
    assert sum(config["task_family_targets"].values()) == conversations
    assert budget["projected_new_assistant_tokens"] == (
        assistant_responses * budget["planning_mean_tokens_per_new_response"]
    )
    assert budget["projected_total_assistant_tokens"] == (
        budget["existing_assistant_tokens"]
        + budget["projected_new_assistant_tokens"]
    )
    assert budget["target_total_assistant_tokens_min"] <= (
        budget["projected_total_assistant_tokens"]
    )

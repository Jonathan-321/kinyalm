import json
from pathlib import Path

from scripts.prepare_gemini_batch_jobs import build_jobs, build_specs, ratio_counts

ROOT = Path(__file__).resolve().parents[1]


def load_inputs():
    config = json.loads(
        (ROOT / "configs/data/gemini_longform_batch_v1.json").read_text()
    )
    curriculum = json.loads((ROOT / config["curriculum"]).read_text())
    prompt = (ROOT / config["prompt_template"]).read_text()
    return config, curriculum, prompt


def test_ratio_counts_are_exact_and_deterministic():
    ratios = {"a": 0.65, "b": 0.3, "c": 0.05}
    assert sum(ratio_counts(2800, ratios).values()) == 2800
    assert ratio_counts(2800, ratios) == ratio_counts(2800, ratios)


def test_curriculum_builds_exact_unique_specs_and_jobs():
    config, curriculum, prompt = load_inputs()
    specs = build_specs(
        curriculum,
        seed=config["seed"],
        first_number=config["first_conversation_number"],
    )
    jobs = build_jobs(config, curriculum, prompt)

    assert len(specs) == 2800
    assert len({row["conversation_id"] for row in specs}) == 2800
    assert len(jobs) == 700
    assert all(len(job["expected_conversations"]) == 4 for job in jobs)
    assert sum(
        row["message_count"] // 2
        for job in jobs
        for row in job["expected_conversations"]
    ) == 6450
    assert "{{BATCH_SPECIFICATION_JSON}}" not in jobs[0]["prompt"]

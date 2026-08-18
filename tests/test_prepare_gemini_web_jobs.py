import json
from pathlib import Path

from scripts.prepare_gemini_web_jobs import build_web_jobs

ROOT = Path(__file__).resolve().parents[1]


def test_builds_exact_web_chat_workload():
    config = json.loads(
        (ROOT / "configs/data/gemini_longform_batch_v1.json").read_text()
    )
    curriculum = json.loads((ROOT / config["curriculum"]).read_text())
    source_prompt = (ROOT / config["prompt_template"]).read_text()
    web_prompt = (ROOT / "prompts/gemini_longform_web_batch_v1.txt").read_text()

    jobs = build_web_jobs(
        config,
        curriculum,
        source_prompt,
        web_prompt,
        source_jobs_per_web_job=2,
    )

    assert len(jobs) == 350
    assert sum(len(job["expected_conversations"]) for job in jobs) == 2800
    assert len({job["web_job_id"] for job in jobs}) == 350
    assert len({
        row["conversation_id"]
        for job in jobs
        for row in job["expected_conversations"]
    }) == 2800
    assert sum(
        row["message_count"] // 2
        for job in jobs
        for row in job["expected_conversations"]
    ) == 6450
    assert "{{BATCH_SPECIFICATION_JSON}}" not in jobs[0]["prompt"]

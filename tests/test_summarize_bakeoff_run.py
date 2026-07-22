import json
from pathlib import Path

import pytest

from scripts.summarize_bakeoff_run import summarize_run


def test_summarize_run_uses_latest_rows_and_flags_mechanical_failures(
    tmp_path: Path,
):
    (tmp_path / "raw").mkdir()
    (tmp_path / "run-manifest.json").write_text("{}", encoding="utf-8")
    rows = [
        {"task_id": "T001", "status": "error", "error": "first attempt"},
        {
            "candidate_id": "local",
            "task_id": "T001",
            "status": "ok",
            "response": "Mur<audio|>ho",
            "output_tokens": 600,
            "latency_seconds": 60,
            "tokens_per_second": 10,
            "peak_unified_memory_gb": 11.25,
            "finish_reason": "length",
        },
    ]
    (tmp_path / "raw" / "local.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarize_run(tmp_path)

    assert summary["latest_result_count"] == 1
    assert summary["status_counts"] == {"ok": 1}
    assert summary["peak_memory_gb"] == 11.25
    assert summary["flags"]["length_truncations"][0]["task_id"] == "T001"
    assert summary["flags"]["control_token_leaks"][0]["tokens"] == ["<audio|>"]
    assert summary["flags"]["long_responses"][0]["output_tokens"] == 600


def test_summarize_run_rejects_invalid_threshold(tmp_path: Path):
    with pytest.raises(ValueError, match="must be positive"):
        summarize_run(tmp_path, long_response_tokens=0)

import json
from pathlib import Path

import pytest

from scripts.publish_bakeoff_run import validate_run_dir


def test_validate_run_dir_accepts_partial_results(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "run-manifest.json").write_text(
        json.dumps({"run_name": "test"}), encoding="utf-8"
    )
    (tmp_path / "raw" / "candidate.jsonl").write_text(
        json.dumps({"task_id": "T001", "status": "ok"}) + "\n",
        encoding="utf-8",
    )

    assert validate_run_dir(tmp_path)["run_name"] == "test"


def test_validate_run_dir_rejects_missing_results(tmp_path: Path):
    (tmp_path / "run-manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no raw result files"):
        validate_run_dir(tmp_path)

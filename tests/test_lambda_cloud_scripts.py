import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts/cloud/run_lambda_baseline.sh"
SUBMIT_SCRIPT = ROOT / "scripts/cloud/submit_lambda_job.sh"


def run_script(script, *args, env=None):
    command_env = os.environ.copy()
    command_env.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=ROOT,
        env=command_env,
        capture_output=True,
        text=True,
    )


def test_gemma4_profile_is_pinned_without_starting_a_run():
    result = run_script(
        RUN_SCRIPT,
        env={"MODEL_PROFILE": "gemma4", "PROFILE_ONLY": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert "model_id=google/gemma-4-12B-it" in result.stdout
    assert (
        "model_revision=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
        in result.stdout
    )
    assert "output_repo=kinyalm/kinyalm-gemma-4-12b-experimental" in result.stdout


def test_gemma4_is_the_default_profile():
    result = run_script(RUN_SCRIPT, env={"PROFILE_ONLY": "1"})

    assert result.returncode == 0, result.stderr
    assert "model_profile=gemma4" in result.stdout
    assert "model_id=google/gemma-4-12B-it" in result.stdout


def test_submit_blocks_full_gemma4_run_before_reading_credentials(tmp_path):
    result = run_script(
        SUBMIT_SCRIPT,
        "127.0.0.1",
        env={
            "MODEL_PROFILE": "gemma4",
            "MAX_STEPS": "-1",
            "LAMBDA_SSH_KEY": str(tmp_path / "missing-key"),
        },
    )

    assert result.returncode == 2
    assert "limited to MAX_STEPS=1" in result.stderr
    assert "private key not found" not in result.stderr


def test_submit_allows_one_step_to_reach_credential_check(tmp_path):
    missing_key = tmp_path / "missing-key"
    result = run_script(
        SUBMIT_SCRIPT,
        "127.0.0.1",
        env={
            "MODEL_PROFILE": "gemma4",
            "MAX_STEPS": "1",
            "LAMBDA_SSH_KEY": str(missing_key),
        },
    )

    assert result.returncode == 1
    assert f"Lambda SSH private key not found: {missing_key}" in result.stderr

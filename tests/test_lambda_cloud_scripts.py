import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts/cloud/run_lambda_baseline.sh"
SUBMIT_SCRIPT = ROOT / "scripts/cloud/submit_lambda_job.sh"
RECOVERY_SUBMIT = ROOT / "scripts/cloud/submit_recovery_arm.py"
CONTINUATION_SUBMIT = ROOT / "scripts/cloud/submit_continuation_arm.py"


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
    assert "warmup_ratio=0.03" in result.stdout
    assert "learning_rate=5e-5" in result.stdout
    assert "epochs=1" in result.stdout
    assert "save_steps=25" in result.stdout
    assert "eval_steps=25" in result.stdout
    assert "samples_enabled=1" in result.stdout


def test_gemma4_sft10k_profile_is_pinned():
    result = run_script(
        RUN_SCRIPT,
        env={
            "MODEL_PROFILE": "gemma4",
            "DATA_PROFILE": "sft10k-v4",
            "PROFILE_ONLY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "data_profile=sft10k-v4" in result.stdout
    assert (
        "data_revision=2092af49ec3478dbdebee9e65a936ec0dc1b3e7c"
        in result.stdout
    )
    assert "output_repo=kinyalm/kinyalm-gemma-4-12b-sft10k-v1" in result.stdout
    assert "save_steps=200" in result.stdout
    assert "eval_steps=200" in result.stdout


def test_gemma4_human_reviewed_profile_is_pinned():
    result = run_script(
        RUN_SCRIPT,
        env={
            "MODEL_PROFILE": "gemma4",
            "DATA_PROFILE": "human-reviewed-432",
            "PROFILE_ONLY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "data_profile=human-reviewed-432" in result.stdout
    assert (
        "data_revision=9e599494681e30beac36e5d5b95ffc193d3bb99c"
        in result.stdout
    )
    assert "output_repo=kinyalm/kinyalm-gemma-4-12b-human432" in result.stdout
    assert "save_steps=25" in result.stdout
    assert "eval_steps=25" in result.stdout


def test_gemma4_team_reviewed_longform_profile_uses_exact_package_gate():
    result = run_script(
        RUN_SCRIPT,
        env={
            "MODEL_PROFILE": "gemma4",
            "DATA_PROFILE": "team-reviewed-longform-v1",
            "DATA_REVISION": "a" * 40,
            "MAX_SEQ_LEN": "1536",
            "PROFILE_ONLY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "data_profile=team-reviewed-longform-v1" in result.stdout
    assert "data_minimum_rows=3144" in result.stdout
    assert "data_maximum_rows=3144" in result.stdout
    assert (
        "data_path_in_repo=data/reviewed/"
        "kinyalm-team-reviewed-longform-sft3144-v1"
    ) in result.stdout
    assert "max_sequence_length=1536" in result.stdout


def test_team_reviewed_longform_profile_uses_approved_split_names():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert '&& "$DATA_PROFILE" != "team-reviewed-longform-v1"' in script


def test_one_step_smoke_disables_warmup_and_samples():
    result = run_script(
        RUN_SCRIPT,
        env={
            "MODEL_PROFILE": "gemma4",
            "MAX_STEPS": "1",
            "PROFILE_ONLY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "max_steps=1" in result.stdout
    assert "warmup_ratio=0" in result.stdout
    assert "samples_enabled=0" in result.stdout


def test_explicit_empty_sample_file_disables_final_sample_generation():
    result = run_script(
        RUN_SCRIPT,
        env={
            "MODEL_PROFILE": "gemma4",
            "SAMPLE_PROMPTS_FILE": "",
            "PROFILE_ONLY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "samples_enabled=0" in result.stdout
    assert "sample_prompts_file=\n" in result.stdout


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


def test_submit_dry_run_preserves_experiment_overrides():
    result = run_script(
        SUBMIT_SCRIPT,
        "203.0.113.10",
        "codex/experiment-matrix",
        env={
            "MODEL_PROFILE": "gemma4",
            "DATA_PROFILE": "sft10k-v4",
            "MAX_STEPS": "100",
            "ALLOW_EXPERIMENTAL_FULL_RUN": "1",
            "CANDIDATE_QUALITY_POLICY": "core-direct",
            "LEARNING_RATE": "1e-5",
            "WARMUP_RATIO": "0.03",
            "EPOCHS": "1",
            "MAX_SEQ_LEN": "1536",
            "SAVE_STEPS": "50",
            "EVAL_STEPS": "50",
            "QUALITY_GATE_POLICY": "record",
            "SAMPLE_PROMPTS_FILE": "",
            "OUTPUT_REPO": "kinyalm/core-smoke",
            "RUN_ID": "core-smoke-v1",
            "SUBMIT_DRY_RUN": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "git_ref=codex/experiment-matrix" in result.stdout
    assert "candidate_quality_policy=core-direct" in result.stdout
    assert "learning_rate=1e-5" in result.stdout
    assert "save_steps=50" in result.stdout
    assert "eval_steps=50" in result.stdout
    assert "quality_gate_policy=record" in result.stdout
    assert "sample_prompts_file=\n" in result.stdout
    assert "max_sequence_length=1536" in result.stdout
    assert "output_repo=kinyalm/core-smoke" in result.stdout
    assert "run_id=core-smoke-v1" in result.stdout


def test_continuation_arm_uses_fresh_optimizer_from_pinned_adapter():
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(CONTINUATION_SUBMIT),
            "203.0.113.10",
            "control",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "max_steps=780" in result.stdout
    assert "learning_rate=2e-06" in result.stdout
    assert "quality_gate_steps=130,260,390,520,650,780" in result.stdout
    assert "quality_gate_policy=record" in result.stdout
    assert "resume_from_checkpoint=\n" in result.stdout
    assert (
        "init_adapter=kinyalm/"
        "kinyalm-gemma-4-12b-longform-fullepoch-qv-r8-lr2e5"
    ) in result.stdout
    assert (
        "init_adapter_revision="
        "44f584225fb3cc215a52be69fcb107da2e743643"
    ) in result.stdout
    assert "data_minimum_rows=3144" in result.stdout
    assert "data_maximum_rows=3144" in result.stdout
    assert (
        "data_revision=1e44922ebea410ffac2423d348be285c288be1db"
        in result.stdout
    )


def test_targeted_continuation_uses_curriculum_package_gate():
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(CONTINUATION_SUBMIT),
            "203.0.113.10",
            "targeted",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "data_minimum_rows=3858" in result.stdout
    assert "data_maximum_rows=3858" in result.stdout
    assert "kinyalm-team-reviewed-longform-curriculum-v1" in result.stdout
    assert "continuation-targeted-lr2e6" in result.stdout
    assert (
        "data_revision=55e8704731fdcbf589da91768bc8739c0ae741b5"
        in result.stdout
    )


def test_submit_rejects_init_adapter_with_optimizer_resume():
    result = run_script(
        SUBMIT_SCRIPT,
        "203.0.113.10",
        env={
            "MODEL_PROFILE": "gemma4",
            "MAX_STEPS": "100",
            "ALLOW_EXPERIMENTAL_FULL_RUN": "1",
            "INIT_ADAPTER": "kinyalm/adapter",
            "INIT_ADAPTER_REVISION": "a" * 40,
            "RESUME_FROM_CHECKPOINT": "/tmp/checkpoint-100",
            "SUBMIT_DRY_RUN": "1",
        },
    )

    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_extended_recovery_probe_is_record_only_and_preserves_five_checkpoints():
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(RECOVERY_SUBMIT),
            "203.0.113.10",
            "qv-r8-lr3e5",
            "a" * 40,
            "--extended-probe",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "max_steps=250" in result.stdout
    assert "learning_rate=3e-05" in result.stdout
    assert "quality_gate_steps=50,100,150,200,250" in result.stdout
    assert "quality_gate_policy=record" in result.stdout
    assert "preserve_checkpoint_steps=50,100,150,200,250" in result.stdout
    assert "sample_prompts_file=\n" in result.stdout


def test_full_epoch_resume_keeps_scheduler_state_and_records_later_gates():
    checkpoint = "/home/ubuntu/kinyalm-runs/full/adapter/checkpoint-100"
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(RECOVERY_SUBMIT),
            "203.0.113.10",
            "qv-r8-lr2e5",
            "a" * 40,
            "--full-epoch",
            "--resume-from-checkpoint",
            checkpoint,
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "max_steps=780" in result.stdout
    assert f"resume_from_checkpoint={checkpoint}" in result.stdout
    assert "quality_gate_policy=record" in result.stdout
    assert "sample_prompts_file=\n" in result.stdout


def test_training_publish_command_keeps_checkpoint_argument_attached():
    source = RUN_SCRIPT.read_text(encoding="utf-8")

    assert (
        '--dataset-revision "$DATA_REVISION" \\\n'
        '  --checkpoint-steps "$PRESERVE_CHECKPOINT_STEPS"'
    ) in source

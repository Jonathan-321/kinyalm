#!/usr/bin/env python3
"""Submit a pinned Gemma 4 long-form probe or full epoch to Lambda."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/training/gemma4_recovery_arms.json"
SUBMIT_SCRIPT = ROOT / "scripts/cloud/submit_lambda_job.sh"


def load_arm(path: Path, arm_id: str) -> tuple[dict, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    arms = {arm["id"]: arm for arm in config.get("arms", [])}
    if arm_id not in arms:
        raise ValueError(
            f"unknown arm {arm_id!r}; choose from {', '.join(sorted(arms))}"
        )
    return config, arms[arm_id]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("arm_id")
    parser.add_argument("data_revision")
    parser.add_argument("--git-ref", default="codex/longform-sft-probes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--full-epoch",
        action="store_true",
        help="Start a fresh full-epoch run using the selected probe arm.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.data_revision):
        raise SystemExit("data_revision must be a 40-character commit SHA")
    try:
        config, arm = load_arm(args.config, args.arm_id)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid recovery arm: {exc}") from exc

    shared = config["shared"]
    dataset_gate = config["dataset_gate"]
    if args.full_epoch:
        phase = config["full_epoch"]
        max_steps = phase["max_steps"]
        save_steps = phase["save_steps"]
        eval_steps = phase["eval_steps"]
        quality_gate_steps = phase["quality_gate_steps"]
        preserve_checkpoint_steps = phase["preserve_checkpoint_steps"]
        run_kind = "fullepoch"
    else:
        max_steps = shared["max_steps"]
        save_steps = 25
        eval_steps = 25
        quality_gate_steps = shared["checkpoint_steps"]
        preserve_checkpoint_steps = shared["checkpoint_steps"]
        run_kind = "probe"
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PROFILE": "gemma4",
            "DATA_PROFILE": dataset_gate["profile"],
            "DATA_REVISION": args.data_revision,
            "DATA_PATH_IN_REPO": dataset_gate["path_in_repo"],
            "DATA_MINIMUM_ROWS": str(dataset_gate["minimum_rows"]),
            "DATA_MAXIMUM_ROWS": str(dataset_gate["maximum_rows"]),
            "MAX_STEPS": str(max_steps),
            "MAX_SEQ_LEN": str(shared["max_sequence_length"]),
            "ALLOW_EXPERIMENTAL_FULL_RUN": "1",
            "LORA_R": str(arm["lora_r"]),
            "LORA_ALPHA": str(shared["lora_alpha"]),
            "LORA_DROPOUT": str(shared["lora_dropout"]),
            "LORA_TARGET_MODULES": ",".join(arm["target_modules"]),
            "LEARNING_RATE": str(arm["learning_rate"]),
            "SAVE_STEPS": str(save_steps),
            "EVAL_STEPS": str(eval_steps),
            "QUALITY_GATE_CONFIG": config["quality_gate"]["held_out_config"],
            "QUALITY_GATE_STEPS": ",".join(
                str(step) for step in quality_gate_steps
            ),
            "PRESERVE_CHECKPOINT_STEPS": ",".join(
                str(step) for step in preserve_checkpoint_steps
            ),
            "OUTPUT_REPO": (
                f"kinyalm/kinyalm-gemma-4-12b-longform-{run_kind}-{args.arm_id}"
            ),
            "RUN_ID": f"gemma4-longform-{run_kind}-{args.arm_id}",
        }
    )
    if args.dry_run:
        env["SUBMIT_DRY_RUN"] = "1"
        return subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), args.host, args.git_ref],
            cwd=ROOT,
            env=env,
            check=False,
        ).returncode

    with tempfile.TemporaryDirectory(prefix="kinyalm-reviewed-sft-") as tmp:
        verification = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/download_reviewed_sft.py"),
                "--revision",
                args.data_revision,
                "--path-in-repo",
                dataset_gate["path_in_repo"],
                "--output-dir",
                tmp,
                "--minimum-rows",
                str(dataset_gate["minimum_rows"]),
                "--maximum-rows",
                str(dataset_gate["maximum_rows"]),
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if verification.returncode:
            return verification.returncode
        return subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), args.host, args.git_ref],
            cwd=ROOT,
            env=env,
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Submit one pinned step-780 continuation arm to Lambda."""

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
DEFAULT_CONFIG = ROOT / "configs/training/gemma4_continuation_arms.json"
SUBMIT_SCRIPT = ROOT / "scripts/cloud/submit_lambda_job.sh"


def load_variant(path: Path, variant_id: str) -> tuple[dict, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    variants = config.get("variants", {})
    if variant_id not in variants:
        raise ValueError(
            f"unknown variant {variant_id!r}; choose from "
            f"{', '.join(sorted(variants))}"
        )
    return config, variants[variant_id]


def build_environment(
    *,
    config: dict,
    variant: dict,
) -> dict[str, str]:
    shared = config["shared"]
    initial_adapter = config["initial_adapter"]
    checkpoints = shared["checkpoint_steps"]
    return {
        "MODEL_PROFILE": "gemma4",
        "DATA_PROFILE": variant["data_profile"],
        "DATA_REVISION": variant["revision"],
        "DATA_PATH_IN_REPO": variant["path_in_repo"],
        "DATA_MINIMUM_ROWS": str(variant["minimum_rows"]),
        "DATA_MAXIMUM_ROWS": str(variant["maximum_rows"]),
        "MAX_STEPS": str(shared["max_steps"]),
        "MAX_SEQ_LEN": str(shared["max_sequence_length"]),
        "ALLOW_EXPERIMENTAL_FULL_RUN": "1",
        "LORA_R": str(shared["lora_r"]),
        "LORA_ALPHA": str(shared["lora_alpha"]),
        "LORA_DROPOUT": str(shared["lora_dropout"]),
        "LORA_TARGET_MODULES": ",".join(shared["target_modules"]),
        "LEARNING_RATE": str(shared["learning_rate"]),
        "WARMUP_RATIO": str(shared["warmup_ratio"]),
        "SAVE_STEPS": str(shared["save_steps"]),
        "EVAL_STEPS": str(shared["eval_steps"]),
        "QUALITY_GATE_CONFIG": config["quality_gate"]["held_out_config"],
        "QUALITY_GATE_STEPS": ",".join(str(step) for step in checkpoints),
        "QUALITY_GATE_POLICY": shared["quality_gate_policy"],
        "PRESERVE_CHECKPOINT_STEPS": ",".join(
            str(step) for step in checkpoints
        ),
        "SAMPLE_PROMPTS_FILE": "",
        "RESUME_FROM_CHECKPOINT": "",
        "INIT_ADAPTER": initial_adapter["repo_id"],
        "INIT_ADAPTER_REVISION": initial_adapter["revision"],
        "INIT_ADAPTER_SUBFOLDER": initial_adapter["subfolder"],
        "OUTPUT_REPO": variant["output_repo"],
        "RUN_ID": variant["run_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("variant", choices=("control", "targeted"))
    parser.add_argument("--git-ref", default="codex/longform-sft-probes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config, variant = load_variant(args.config, args.variant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid continuation arm: {exc}") from exc

    if not re.fullmatch(r"[0-9a-f]{40}", str(variant.get("revision", ""))):
        raise SystemExit("variant revision must be a 40-character commit SHA")
    env = os.environ.copy()
    env.update(
        build_environment(
            config=config,
            variant=variant,
        )
    )
    if args.dry_run:
        env["SUBMIT_DRY_RUN"] = "1"
        return subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), args.host, args.git_ref],
            cwd=ROOT,
            env=env,
            check=False,
        ).returncode

    with tempfile.TemporaryDirectory(prefix="kinyalm-continuation-sft-") as tmp:
        verification = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/download_reviewed_sft.py"),
                "--revision",
                variant["revision"],
                "--path-in-repo",
                variant["path_in_repo"],
                "--output-dir",
                tmp,
                "--minimum-rows",
                str(variant["minimum_rows"]),
                "--maximum-rows",
                str(variant["maximum_rows"]),
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

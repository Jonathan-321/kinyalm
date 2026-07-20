#!/usr/bin/env python3
"""Build an explicitly experimental SFT baseline from a pinned HF revision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.hf_baseline import (  # noqa: E402
    BASELINE_MODES,
    DEFAULT_REPO_ID,
    materialize_hf_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=sorted(BASELINE_MODES),
        default="critic-accepted",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", default="kinyalm-hf-baseline-v1")
    parser.add_argument(
        "--acknowledge-experimental",
        action="store_true",
        help="Confirm that model-critic review is not fluent-human approval.",
    )
    args = parser.parse_args()

    if not args.acknowledge_experimental:
        raise SystemExit(
            "Pass --acknowledge-experimental to build this non-production tier."
        )

    try:
        manifest = materialize_hf_baseline(
            repo_id=args.repo_id,
            revision=args.revision,
            output_dir=args.output_dir,
            mode=args.mode,
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    build = manifest["build"]
    print(f"HF revision: {manifest['source']['resolved_revision']}")
    print(f"Selected rows: {build['selected_rows']}")
    print(f"Splits: {build['split_counts']}")
    print(f"Output: {Path(args.output_dir).expanduser()}")
    print("Tier: experimental critic-filtered; not human-approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

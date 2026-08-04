#!/usr/bin/env python3
"""Combine base, direct-PEFT, and MLX probe outputs into one blind review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kinyalm.evaluation import (  # noqa: E402
    BakeoffConfig,
    TutorTask,
    latest_results,
    load_bakeoff_config,
    write_blind_review_pack,
)
from scripts.run_multilingual_bakeoff import (  # noqa: E402
    DEFAULT_CONFIG,
    load_held_out_tasks,
    select_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        required=True,
        help="Raw JSONL result path; repeat once per candidate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_parity_results(
    result_paths: list[Path],
    tasks: list[TutorTask],
    config: BakeoffConfig,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load complete candidate outputs and enforce identical evaluation inputs."""

    if len(result_paths) < 2:
        raise ValueError("parity review requires at least two result files")

    expected_task_ids = {task.id for task in tasks}
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for path in result_paths:
        records = latest_results(path)
        selected = {
            task_id: record
            for task_id, record in records.items()
            if task_id in expected_task_ids
        }
        missing = sorted(expected_task_ids.difference(selected))
        if missing:
            raise ValueError(f"{path} is missing tasks: {', '.join(missing)}")

        candidate_ids = {
            str(record.get("candidate_id", "")) for record in selected.values()
        }
        if len(candidate_ids) != 1 or "" in candidate_ids:
            raise ValueError(f"{path} must contain exactly one candidate ID")
        candidate_id = candidate_ids.pop()
        if candidate_id in candidates:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")

        task_by_id = {task.id: task for task in tasks}
        for task_id, record in selected.items():
            task = task_by_id[task_id]
            expected = {
                "prompt": task.prompt,
                "system_prompt": config.system_prompt,
                "seed": config.seed,
                "max_new_tokens": config.max_new_tokens,
                "enable_thinking": config.enable_thinking,
            }
            mismatched = [
                key for key, value in expected.items() if record.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    f"{path}:{task_id} differs in {', '.join(mismatched)}"
                )
        candidates[candidate_id] = selected
    return candidates


def write_manifest(
    output_dir: Path,
    result_paths: list[Path],
    candidates: dict[str, dict[str, dict[str, Any]]],
    tasks: list[TutorTask],
) -> None:
    manifest = {
        "schema_version": 1,
        "candidate_ids": sorted(candidates),
        "task_ids": [task.id for task in tasks],
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in result_paths
        ],
    }
    (output_dir / "parity-review-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config = load_bakeoff_config(args.config.resolve())
    _, all_tasks = load_held_out_tasks(config)
    tasks = select_tasks(all_tasks, args.task_id, args.limit)
    result_paths = [path.expanduser().resolve() for path in args.result]
    candidates = load_parity_results(result_paths, tasks, config)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count, _ = write_blind_review_pack(
        output_csv=output_dir / "blind-review.csv",
        key_path=output_dir / "blind-key.json",
        tasks=tasks,
        candidate_results=candidates,
        seed=config.seed,
    )
    write_manifest(output_dir, result_paths, candidates, tasks)
    print(
        f"Wrote {row_count} blind rows for {len(candidates)} candidates "
        f"across {len(tasks)} tasks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build one blinded four-way review pack for continuation screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.evaluation import (  # noqa: E402
    benchmark_tasks,
    latest_results,
    load_bakeoff_config,
    load_task_bank,
    write_blind_review_pack,
)

DEFAULT_CONFIG = ROOT / "configs/evaluation/gemma4_recovery_bakeoff.json"
DEFAULT_RUN_ROOT = (
    ROOT
    / "outputs/evaluation/gemma4-continuation-native-dev-screen-v1-20260811"
)

RAW_SOURCES = (
    (
        "base",
        "source780/raw/gemma4-12b-it-qat-4bit-mlx.jsonl",
    ),
    (
        "source_step780",
        "source780/raw/kinyalm-gemma-4-12b-longform-fullepoch-qv-r8-lr2e5-mlx.jsonl",
    ),
    (
        "continuation_control",
        "control/raw/kinyalm-gemma-4-12b-continuation-control-lr2e6-mlx.jsonl",
    ),
    (
        "continuation_targeted",
        "targeted/raw/kinyalm-gemma-4-12b-continuation-targeted-lr2e6-mlx.jsonl",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_results(
    *,
    path: Path,
    tasks: list[Any],
    config: Any,
) -> tuple[str, dict[str, dict[str, Any]]]:
    results = latest_results(path)
    expected_ids = {task.id for task in tasks}
    actual_ids = set(results)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"{path}: task IDs do not match; missing={missing[:5]} extra={extra[:5]}"
        )

    candidate_ids = {str(row.get("candidate_id", "")) for row in results.values()}
    if len(candidate_ids) != 1 or "" in candidate_ids:
        raise ValueError(f"{path}: expected exactly one non-empty candidate_id")
    candidate_id = candidate_ids.pop()

    task_by_id = {task.id: task for task in tasks}
    for task_id, row in results.items():
        task = task_by_id[task_id]
        if row.get("status") != "ok" or not str(row.get("response", "")).strip():
            raise ValueError(f"{path}: {task_id} has no successful response")
        expected = {
            "prompt": task.prompt,
            "category": task.category,
            "review_focus": task.review_focus,
            "split": task.split,
            "system_prompt": config.system_prompt,
            "seed": config.seed,
            "max_new_tokens": config.max_new_tokens,
            "enable_thinking": config.enable_thinking,
        }
        mismatched = [key for key, value in expected.items() if row.get(key) != value]
        if mismatched:
            raise ValueError(
                f"{path}: {task_id} changed evaluation fields: "
                + ", ".join(mismatched)
            )
    return candidate_id, results


def build_combined_review(
    *,
    run_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_bakeoff_config(config_path)
    task_bank_path = ROOT / config.task_bank
    tasks = benchmark_tasks(load_task_bank(task_bank_path))
    if len(tasks) != config.expected_task_count:
        raise ValueError(
            f"expected {config.expected_task_count} benchmark tasks, found {len(tasks)}"
        )

    candidate_results: dict[str, dict[str, dict[str, Any]]] = {}
    sources = []
    shared_runtime: dict[str, Any] | None = None
    for role, relative_path in RAW_SOURCES:
        path = run_root / relative_path
        if not path.is_file():
            raise ValueError(f"missing raw evaluation output: {path}")
        candidate_id, results = validate_candidate_results(
            path=path,
            tasks=tasks,
            config=config,
        )
        if candidate_id in candidate_results:
            raise ValueError(f"duplicate candidate_id across raw files: {candidate_id}")

        first = results[tasks[0].id]
        runtime = {
            "model_id": first.get("model_id"),
            "model_revision": first.get("model_revision"),
            "inference_backend": first.get("inference_backend"),
            "inference_backend_version": first.get("inference_backend_version"),
            "quantization": first.get("quantization"),
        }
        if shared_runtime is None:
            shared_runtime = runtime
        elif runtime != shared_runtime:
            raise ValueError(f"{path}: runtime differs from the other candidates")

        candidate_results[candidate_id] = results
        sources.append(
            {
                "role": role,
                "candidate_id": candidate_id,
                "path": str(path.relative_to(run_root)),
                "sha256": sha256_file(path),
                "adapter_id": first.get("adapter_id"),
                "adapter_revision": first.get("adapter_revision"),
                "adapter_sha256": first.get("adapter_sha256"),
            }
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    for source in sources:
        source_path = run_root / str(source["path"])
        shutil.copy2(source_path, raw_dir / source_path.name)

    row_count, labels = write_blind_review_pack(
        output_csv=output_dir / "review/blind-review.csv",
        key_path=output_dir / "private/blind-key.json",
        tasks=tasks,
        candidate_results=candidate_results,
        seed=config.seed,
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_kind": "quantized-local-development-screen",
        "promotion_eligible": False,
        "promotion_blocker": (
            "These 150 prompts are development screening and may overlap the "
            "training curriculum. Promotion requires a new hidden native-reviewed set."
        ),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "task_bank_path": str(task_bank_path.relative_to(ROOT)),
        "task_bank_sha256": sha256_file(task_bank_path),
        "task_count": len(tasks),
        "candidate_count": len(candidate_results),
        "review_row_count": row_count,
        "labels_are_private": True,
        "shared_runtime": shared_runtime,
        "sources": sources,
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "combined"
    )
    manifest = build_combined_review(
        run_root=run_root,
        config_path=config_path,
        output_dir=output_dir,
    )
    print(
        f"Wrote {manifest['review_row_count']} blinded rows for "
        f"{manifest['candidate_count']} candidates to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

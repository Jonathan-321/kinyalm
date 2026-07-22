#!/usr/bin/env python3
"""Create objective metrics and mechanical flags for a bake-off run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTROL_TOKEN = re.compile(r"<\|?(?:audio|image|video)\|?>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--long-response-tokens", type=int, default=600)
    return parser.parse_args()


def summarize_run(run_dir: Path, long_response_tokens: int = 600) -> dict[str, Any]:
    """Summarize latest result rows without assigning language-quality scores."""

    if long_response_tokens < 1:
        raise ValueError("long_response_tokens must be positive")
    manifest_path = run_dir / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing run manifest: {manifest_path}")

    raw_files = sorted((run_dir / "raw").glob("*.jsonl"))
    if not raw_files:
        raise ValueError(f"no raw result files found under {run_dir / 'raw'}")

    rows: list[dict[str, Any]] = []
    raw_metadata = []
    for raw_path in raw_files:
        latest: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(
            raw_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("task_id"):
                raise ValueError(f"invalid result at {raw_path}:{line_number}")
            latest[str(row["task_id"])] = row
        rows.extend(latest.values())
        raw_metadata.append(
            {
                "path": str(raw_path.relative_to(run_dir)),
                "bytes": raw_path.stat().st_size,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    status_counts = Counter(str(row.get("status", "missing")) for row in rows)
    finish_counts = Counter(str(row.get("finish_reason", "missing")) for row in ok_rows)
    token_counts = [int(row["output_tokens"]) for row in ok_rows]
    latencies = [float(row["latency_seconds"]) for row in ok_rows]
    throughputs = [float(row["tokens_per_second"]) for row in ok_rows]
    memory_values = [
        float(value)
        for row in ok_rows
        for value in (
            row.get("peak_unified_memory_gb"),
            row.get("peak_gpu_memory_gb"),
        )
        if value is not None
    ]

    control_leaks = []
    for row in ok_rows:
        tokens = CONTROL_TOKEN.findall(str(row.get("response", "")))
        if tokens:
            control_leaks.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "task_id": row.get("task_id"),
                    "count": len(tokens),
                    "tokens": sorted(set(tokens)),
                }
            )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Objective runtime metrics and mechanical output flags only; "
            "not a Kinyarwanda quality score."
        ),
        "candidate_count": len({row.get("candidate_id") for row in rows}),
        "latest_result_count": len(rows),
        "unique_task_count": len({row.get("task_id") for row in rows}),
        "status_counts": dict(sorted(status_counts.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "output_tokens_total": sum(token_counts),
        "output_tokens_median": _median(token_counts),
        "output_tokens_min": min(token_counts, default=None),
        "output_tokens_max": max(token_counts, default=None),
        "latency_total_minutes": _round(sum(latencies) / 60),
        "latency_median_seconds": _median(latencies),
        "throughput_median_tokens_per_second": _median(throughputs),
        "peak_memory_gb": _round(max(memory_values)) if memory_values else None,
        "flags": {
            "error_rows": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "task_id": row.get("task_id"),
                    "error": row.get("error"),
                }
                for row in rows
                if row.get("status") != "ok"
            ],
            "length_truncations": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "task_id": row.get("task_id"),
                    "output_tokens": row.get("output_tokens"),
                }
                for row in ok_rows
                if row.get("finish_reason") == "length"
            ],
            "control_token_leaks": control_leaks,
            "long_responses": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "task_id": row.get("task_id"),
                    "output_tokens": row.get("output_tokens"),
                }
                for row in ok_rows
                if int(row.get("output_tokens", 0)) >= long_response_tokens
            ],
            "long_response_threshold_tokens": long_response_tokens,
        },
        "raw_files": raw_metadata,
    }


def _median(values: list[int] | list[float]) -> float | None:
    return _round(statistics.median(values)) if values else None


def _round(value: float) -> float:
    return round(value, 4)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = summarize_run(run_dir, args.long_response_tokens)
    output_path = run_dir / "summary" / "automatic-metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

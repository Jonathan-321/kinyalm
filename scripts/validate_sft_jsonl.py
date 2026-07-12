#!/usr/bin/env python3
"""Validate a project SFT JSONL file."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.sft import load_jsonl, validate_sft_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to an SFT JSONL file")
    args = parser.parse_args()

    path = Path(args.path)
    records = load_jsonl(path)
    results = validate_sft_records(records)
    failures = [result for result in results if not result.ok]

    print(f"Loaded {len(records)} SFT records from {path}.")
    if not failures:
        print("SFT JSONL validation passed.")
        return 0

    for result in failures:
        for error in result.errors:
            print(f"line {result.line_number}: {error}", file=sys.stderr)
    print(f"SFT JSONL validation failed: {len(failures)} bad rows.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

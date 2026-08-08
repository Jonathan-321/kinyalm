#!/usr/bin/env python3
"""Fail a checkpoint gate when it adds severe repetition versus the base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinyalm.evaluation.repetition import compare_probe_repetition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--minimum-occurrences", type=int, default=5)
    parser.add_argument("--maximum-new-rows", type=int, default=0)
    args = parser.parse_args()
    report = compare_probe_repetition(
        args.base,
        args.candidate,
        ngram_size=args.ngram_size,
        minimum_occurrences=args.minimum_occurrences,
        maximum_new_rows=args.maximum_new_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Repetition gate passed={report['passed']} "
        f"new_rows={report['new_severe_repetition_rows']}"
    )
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

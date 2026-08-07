"""Mechanical repetition checks for matched model probe outputs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_probe(path: str | Path) -> dict[str, str]:
    """Load one completion per unique prompt from a probe JSONL file."""

    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", row.get("response", ""))).strip()
        if not prompt or not completion:
            raise ValueError(f"{path}:{line_number} requires prompt and completion")
        if prompt in rows:
            raise ValueError(f"{path}:{line_number} duplicates prompt {prompt!r}")
        rows[prompt] = completion
    if not rows:
        raise ValueError(f"probe file is empty: {path}")
    return rows


def repetition_summary(
    completions: dict[str, str],
    *,
    ngram_size: int = 4,
    minimum_occurrences: int = 5,
) -> dict[str, Any]:
    """Count rows containing a repeatedly emitted word n-gram."""

    if ngram_size < 1 or minimum_occurrences < 2:
        raise ValueError("invalid repetition thresholds")
    flagged = []
    unique_ratios = []
    for prompt, completion in completions.items():
        words = re.findall(r"\w+", completion.casefold(), flags=re.UNICODE)
        ngrams = [
            tuple(words[index : index + ngram_size])
            for index in range(max(0, len(words) - ngram_size + 1))
        ]
        counts = Counter(ngrams)
        unique_ratios.append(len(counts) / len(ngrams) if ngrams else 1.0)
        if counts:
            repeated, occurrences = counts.most_common(1)[0]
            if occurrences >= minimum_occurrences:
                flagged.append(
                    {
                        "prompt": prompt,
                        "ngram": " ".join(repeated),
                        "occurrences": occurrences,
                    }
                )
    return {
        "row_count": len(completions),
        "severe_repetition_rows": len(flagged),
        "mean_unique_ngram_ratio": round(
            sum(unique_ratios) / len(unique_ratios), 4
        ),
        "flags": flagged,
        "ngram_size": ngram_size,
        "minimum_occurrences": minimum_occurrences,
    }


def compare_probe_repetition(
    base_path: str | Path,
    candidate_path: str | Path,
    *,
    ngram_size: int = 4,
    minimum_occurrences: int = 5,
    maximum_new_rows: int = 0,
) -> dict[str, Any]:
    """Require matched prompts and reject new severe repetition rows."""

    if maximum_new_rows < 0:
        raise ValueError("maximum_new_rows cannot be negative")
    base = load_probe(base_path)
    candidate = load_probe(candidate_path)
    if set(base) != set(candidate):
        missing = sorted(set(base).difference(candidate))
        extra = sorted(set(candidate).difference(base))
        raise ValueError(
            f"probe prompts differ; missing={missing[:3]}, extra={extra[:3]}"
        )
    base_summary = repetition_summary(
        base,
        ngram_size=ngram_size,
        minimum_occurrences=minimum_occurrences,
    )
    candidate_summary = repetition_summary(
        candidate,
        ngram_size=ngram_size,
        minimum_occurrences=minimum_occurrences,
    )
    new_rows = max(
        0,
        candidate_summary["severe_repetition_rows"]
        - base_summary["severe_repetition_rows"],
    )
    return {
        "passed": new_rows <= maximum_new_rows,
        "base": base_summary,
        "candidate": candidate_summary,
        "new_severe_repetition_rows": new_rows,
        "maximum_new_severe_repetition_rows": maximum_new_rows,
    }

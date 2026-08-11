#!/usr/bin/env python3
"""Audit benchmark prompts against every user turn in SFT JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from heapq import nlargest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kinyalm.evaluation import benchmark_tasks, load_task_bank  # noqa: E402

TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
DEFAULT_TASK_BANK = ROOT / "docs/evaluation/recovery-task-bank.md"
DEFAULT_DATASETS = (
    ROOT
    / "outputs/datasets/kinyalm-team-reviewed-longform-sft3144-v1/train.jsonl",
    ROOT
    / Path(
        "outputs/datasets/kinyalm-team-reviewed-longform-sft3144-v1/validation.jsonl"
    ),
)


@dataclass(frozen=True)
class TrainingPrompt:
    source_file: str
    record_id: str
    split: str
    task_family: str
    message_index: int
    text: str
    normalized: str
    tokens: frozenset[str]


def normalize_text(text: str) -> str:
    """Normalize text for reproducible lexical comparison."""

    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return " ".join(TOKEN.findall(normalized))


def similarity(left: str, right: str) -> dict[str, float]:
    """Return complementary lexical similarity scores in the zero-to-one range."""

    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    union = left_tokens | right_tokens
    token_jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
    sequence_ratio = SequenceMatcher(
        None, left_normalized, right_normalized, autojunk=False
    ).ratio()
    return {
        "token_jaccard": round(token_jaccard, 6),
        "sequence_ratio": round(sequence_ratio, 6),
        "combined": round(max(token_jaccard, sequence_ratio), 6),
    }


def load_training_prompts(paths: list[Path]) -> list[TrainingPrompt]:
    """Load every user message, not only the first turn, from SFT records."""

    prompts: list[TrainingPrompt] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list):
                    raise ValueError(f"{path}:{line_number}: messages must be a list")
                for message_index, message in enumerate(messages):
                    if not isinstance(message, dict) or message.get("role") != "user":
                        continue
                    text = str(message.get("content", "")).strip()
                    if not text:
                        raise ValueError(
                            f"{path}:{line_number}: empty user message at "
                            f"index {message_index}"
                        )
                    prompts.append(
                        TrainingPrompt(
                            source_file=str(path),
                            record_id=str(row.get("id", f"line-{line_number}")),
                            split=str(row.get("split", "unknown")),
                            task_family=str(
                                row.get("task_family", row.get("task_type", "unknown"))
                            ),
                            message_index=message_index,
                            text=text,
                            normalized=normalize_text(text),
                            tokens=frozenset(normalize_text(text).split()),
                        )
                    )
    if not prompts:
        raise ValueError("no user prompts were found in the supplied datasets")
    return prompts


def audit_overlap(
    *,
    task_bank: Path,
    dataset_paths: list[Path],
    strict_threshold: float = 0.88,
    review_threshold: float = 0.55,
) -> dict[str, Any]:
    """Find each benchmark prompt's closest lexical training prompt."""

    if not 0 <= review_threshold <= strict_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= review <= strict <= 1")
    tasks = benchmark_tasks(load_task_bank(task_bank))
    training_prompts = load_training_prompts(dataset_paths)
    rows = []
    for task in tasks:
        task_normalized = normalize_text(task.prompt)
        task_tokens = frozenset(task_normalized.split())
        lexical_candidates = []
        for training_prompt in training_prompts:
            union = task_tokens | training_prompt.tokens
            token_jaccard = (
                len(task_tokens & training_prompt.tokens) / len(union)
                if union
                else 1.0
            )
            lexical_candidates.append((token_jaccard, training_prompt))

        scored_candidates = []
        for token_jaccard, training_prompt in nlargest(
            50, lexical_candidates, key=lambda item: item[0]
        ):
            sequence_ratio = SequenceMatcher(
                None,
                task_normalized,
                training_prompt.normalized,
                autojunk=False,
            ).ratio()
            scored_candidates.append(
                (
                    max(token_jaccard, sequence_ratio),
                    {
                        "token_jaccard": round(token_jaccard, 6),
                        "sequence_ratio": round(sequence_ratio, 6),
                        "combined": round(max(token_jaccard, sequence_ratio), 6),
                    },
                    training_prompt,
                )
            )
        _, best_scores, best_prompt = max(scored_candidates, key=lambda item: item[0])
        exact = task_normalized == best_prompt.normalized
        strict_match = exact or best_scores["combined"] >= strict_threshold
        rows.append(
            {
                "task_id": task.id,
                "category": task.category,
                "benchmark_prompt": task.prompt,
                **best_scores,
                "exact_match": exact,
                "strict_match": strict_match,
                "manual_review": best_scores["combined"] >= review_threshold,
                **{
                    key: value
                    for key, value in asdict(best_prompt).items()
                    if key not in {"normalized", "tokens"}
                },
            }
        )

    thresholds = (0.5, 0.6, 0.7, 0.8, strict_threshold)
    return {
        "schema_version": 1,
        "scope": (
            "Lexical contamination screen across every user turn. Similar task "
            "coverage is expected and is not itself evidence of leakage."
        ),
        "benchmark_task_count": len(tasks),
        "training_conversation_files": [str(path) for path in dataset_paths],
        "training_user_turn_count": len(training_prompts),
        "sequence_prefilter_candidates_per_task": 50,
        "strict_threshold": strict_threshold,
        "review_threshold": review_threshold,
        "exact_match_count": sum(row["exact_match"] for row in rows),
        "strict_match_count": sum(row["strict_match"] for row in rows),
        "manual_review_count": sum(row["manual_review"] for row in rows),
        "combined_similarity_counts": {
            f"at_least_{threshold:.2f}": sum(
                row["combined"] >= threshold for row in rows
            )
            for threshold in thresholds
        },
        "closest_matches": sorted(
            rows, key=lambda row: (-row["combined"], row["task_id"])
        ),
    }


def write_report(report: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report["closest_matches"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, default=DEFAULT_TASK_BANK)
    parser.add_argument(
        "--dataset", type=Path, action="append", dest="datasets", default=[]
    )
    parser.add_argument("--strict-threshold", type=float, default=0.88)
    parser.add_argument("--review-threshold", type=float, default=0.55)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_paths = args.datasets or list(DEFAULT_DATASETS)
    report = audit_overlap(
        task_bank=args.task_bank,
        dataset_paths=dataset_paths,
        strict_threshold=args.strict_threshold,
        review_threshold=args.review_threshold,
    )
    write_report(report, args.output_json, args.output_csv)
    print(
        f"Audited {report['benchmark_task_count']} benchmark prompts against "
        f"{report['training_user_turn_count']} training user turns: "
        f"exact={report['exact_match_count']}, "
        f"strict={report['strict_match_count']}, "
        f"manual_review={report['manual_review_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

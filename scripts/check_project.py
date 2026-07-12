"""Lightweight project health check."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from kinyalm.data.examples import load_tsv_examples

    examples_path = ROOT / "docs" / "tokenizer" / "eval-examples.tsv"
    examples = load_tsv_examples(examples_path)
    task_bank_path = ROOT / "docs" / "evaluation" / "learning-task-bank.md"
    schema_path = ROOT / "docs" / "data" / "sft-data-schema.md"
    task_count, benchmark_count = count_task_bank_rows(task_bank_path)

    if task_count < 50:
        raise SystemExit(f"expected at least 50 tutor tasks, found {task_count}")
    if benchmark_count < 15:
        raise SystemExit(
            f"expected at least 15 benchmark-only tasks, found {benchmark_count}"
        )
    if not schema_path.exists():
        raise SystemExit(f"missing SFT schema: {schema_path}")

    print(f"Loaded {len(examples)} tokenizer evaluation examples.")
    print(
        "Loaded "
        f"{task_count} tutor evaluation tasks "
        f"({benchmark_count} benchmark-only)."
    )
    print(f"Found SFT schema at {schema_path.relative_to(ROOT)}.")
    print("First implementation target: tokenizer analysis on reviewed examples.")
    print("Reminder: keep raw training data and KILM experiments out of this repo.")
    return 0


def count_task_bank_rows(path: Path) -> tuple[int, int]:
    task_count = 0
    benchmark_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| T"):
            continue
        task_count += 1
        if "| benchmark-only |" in line:
            benchmark_count += 1
    return task_count, benchmark_count


if __name__ == "__main__":
    sys.exit(main())

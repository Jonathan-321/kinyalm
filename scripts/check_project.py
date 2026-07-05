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

    print(f"Loaded {len(examples)} tokenizer evaluation examples.")
    print("First implementation target: tokenizer analysis on reviewed examples.")
    print("Reminder: keep raw training data and KILM experiments out of this repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Interactive terminal chat for the pinned local Gemma 4 12B checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kinyalm.evaluation import load_bakeoff_config  # noqa: E402
from scripts.run_multilingual_bakeoff import (  # noqa: E402
    DEFAULT_CONFIG,
    MlxGenerator,
    resolve_runtime_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt", help="Run one prompt and exit")
    return parser.parse_args()


def conversation_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    """Build a complete prompt without mutating saved conversation history."""

    return [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_prompt},
    ]


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    config = load_bakeoff_config(args.config.resolve())
    candidate = resolve_runtime_candidates(config, ["gemma4-12b-it"], "mlx")[0]
    print(
        "Local Gemma 4 12B screening checkpoint "
        "(QAT-derived MLX 4/8-bit; no KinyaLM fine-tune)."
    )
    print("Loading the model...")
    generator = MlxGenerator(candidate, config.seed)
    history: list[dict[str, str]] = []

    def answer(user_prompt: str) -> str:
        result = generator.generate_messages(
            messages=conversation_messages(
                config.system_prompt,
                history,
                user_prompt,
            ),
            max_new_tokens=args.max_new_tokens,
            enable_thinking=config.enable_thinking,
        )
        print(
            f"\n[{result['output_tokens']} tokens, "
            f"{result['tokens_per_second']:.2f} tok/s, "
            f"finish={result['finish_reason']}]"
        )
        return str(result["response"])

    try:
        if args.prompt:
            print(answer(args.prompt))
            return 0

        print("Enter /reset to clear context or /quit to exit.")
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt == "/quit":
                break
            if prompt == "/reset":
                history.clear()
                print("Conversation cleared.")
                continue
            if not prompt:
                continue

            response = answer(prompt)
            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            )
            print(f"\nGemma 4: {response}")
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run an interactive Kakugo Kinyarwanda chat with MLX on Apple silicon."""

from __future__ import annotations

import argparse
import os
import platform
import re
import time
from typing import Any

MODEL_ID = "ptrdvn/kakugo-3B-kin"
MODEL_REVISION = "af4e61e47bdd026a73d9a37814f2df8d92f98013"
CONCISE_SYSTEM_PROMPT = "Be concise in your responses."
REASONING_SYSTEM_PROMPT = (
    "Before you respond, first think about your response and enclose your "
    "thinking process in <think> and </think> delimiters."
)
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def system_prompt(reasoning: bool) -> str:
    """Return the system prompt documented by the Kakugo model card."""
    if reasoning:
        return REASONING_SYSTEM_PROMPT
    return CONCISE_SYSTEM_PROMPT


def extract_visible_response(raw_response: str) -> tuple[str, str | None]:
    """Separate Kakugo's optional reasoning block from its visible answer."""
    text = raw_response.strip()
    reasoning_parts = [match.strip() for match in THINK_BLOCK.findall(text)]
    visible = THINK_BLOCK.sub("", text)

    if "<think>" in visible:
        visible, unfinished_reasoning = visible.split("<think>", 1)
        if unfinished_reasoning.strip():
            reasoning_parts.append(unfinished_reasoning.strip())

    reasoning = "\n\n".join(part for part in reasoning_parts if part) or None
    return visible.strip(), reasoning


def trim_history(
    messages: list[dict[str, str]], max_turns: int
) -> list[dict[str, str]]:
    """Keep the system prompt and the newest complete conversation turns."""
    if not messages or max_turns < 1:
        return messages[:1]
    return [messages[0], *messages[1:][-2 * max_turns :]]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat locally with ptrdvn/kakugo-3B-kin using MLX."
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--prompt", help="Run one prompt and exit")
    parser.add_argument("--max-tokens", type=positive_int, default=384)
    parser.add_argument("--history-turns", type=positive_int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only files already present in the Hugging Face cache",
    )
    return parser.parse_args()


def validate_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Kakugo MLX inference requires an Apple-silicon Mac.")


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, Any, float]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 0.0,
    )
    processors = [make_repetition_penalty(1.05)]

    started = time.perf_counter()
    response_parts: list[str] = []
    final_stats = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=processors,
    ):
        response_parts.append(response.text)
        final_stats = response
    elapsed = time.perf_counter() - started
    return "".join(response_parts), final_stats, elapsed


def print_response(
    raw_response: str,
    stats: Any,
    elapsed: float,
    show_reasoning: bool,
) -> str:
    visible, reasoning = extract_visible_response(raw_response)
    if show_reasoning and reasoning:
        print(f"\nKakugo reasoning:\n{reasoning}")

    if visible:
        print(f"\nKakugo: {visible}")
    else:
        print(
            "\nKakugo did not finish a visible answer. Increase "
            "--max-tokens or turn reasoning off."
        )

    if stats is not None:
        print(
            f"\n[{stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.1f} tok/s, "
            f"{stats.peak_memory:.1f} GB peak, {elapsed:.1f}s, "
            f"finish={stats.finish_reason}]"
        )
    return visible


def generate_turn(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    user_prompt: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], bool]:
    messages = trim_history(messages, args.history_turns)
    messages.append({"role": "user", "content": user_prompt})
    print("Generating...", flush=True)
    try:
        raw_response, stats, elapsed = generate_response(
            model, tokenizer, messages, args
        )
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
        return messages[:-1], False

    visible = print_response(
        raw_response,
        stats,
        elapsed,
        args.show_reasoning,
    )
    if not visible:
        return messages[:-1], False
    messages.append({"role": "assistant", "content": visible})
    return messages, True


def set_reasoning(
    command: str, current: bool
) -> tuple[bool, str | None]:
    parts = command.split()
    if len(parts) == 1:
        enabled = not current
    elif len(parts) == 2 and parts[1].lower() in {"on", "off"}:
        enabled = parts[1].lower() == "on"
    else:
        return current, "Use /reasoning on or /reasoning off."
    return enabled, None


def interactive_chat(
    model: Any, tokenizer: Any, args: argparse.Namespace
) -> None:
    reasoning = args.reasoning
    messages = [{"role": "system", "content": system_prompt(reasoning)}]
    print("Commands: /reset, /reasoning on|off, /show-reasoning, /help, /quit")

    while True:
        try:
            user_prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_prompt:
            continue
        if user_prompt == "/quit":
            return
        if user_prompt == "/help":
            print(
                "/reset clears history; /reasoning toggles Kakugo's experimental "
                "reasoning mode; /show-reasoning toggles whether hidden reasoning "
                "is printed; /quit exits."
            )
            continue
        if user_prompt == "/reset":
            messages = [
                {"role": "system", "content": system_prompt(reasoning)}
            ]
            print("Conversation cleared.")
            continue
        if user_prompt.startswith("/reasoning"):
            reasoning, error = set_reasoning(user_prompt, reasoning)
            if error:
                print(error)
                continue
            args.reasoning = reasoning
            messages = [
                {"role": "system", "content": system_prompt(reasoning)}
            ]
            state = "on" if reasoning else "off"
            print(f"Reasoning mode is {state}; conversation cleared.")
            continue
        if user_prompt == "/show-reasoning":
            args.show_reasoning = not args.show_reasoning
            state = "on" if args.show_reasoning else "off"
            print(f"Reasoning display is {state}.")
            continue

        messages, _ = generate_turn(
            model,
            tokenizer,
            messages,
            user_prompt,
            args,
        )


def main() -> None:
    args = parse_args()
    validate_platform()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import mlx.core as mx
    from mlx_lm import load

    mx.random.seed(args.seed)
    print(f"Loading {args.model} at {args.revision[:12]}...")
    started = time.perf_counter()
    model, tokenizer = load(args.model, revision=args.revision)
    print(f"Loaded in {time.perf_counter() - started:.1f}s.")
    print("Kakugo is an experimental baseline; verify important Kinyarwanda.")

    if args.prompt:
        messages = [
            {"role": "system", "content": system_prompt(args.reasoning)}
        ]
        generate_turn(model, tokenizer, messages, args.prompt, args)
        return
    interactive_chat(model, tokenizer, args)


if __name__ == "__main__":
    main()

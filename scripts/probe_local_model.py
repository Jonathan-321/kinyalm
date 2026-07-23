#!/usr/bin/env python3
"""Run a fixed set of prompts through a local MLX model and save the answers.

Use it twice to build a before/after comparison for a fine-tune:

    # BEFORE: base model, no adapter
    python scripts/probe_local_model.py \
        --model mlx-community/gemma-2-2b-it-4bit \
        --prompts-file data/sft/tessy-eval-prompts.txt \
        --output outputs/probe-before.jsonl

    # AFTER: same model + your fine-tuned LoRA adapter
    python scripts/probe_local_model.py \
        --model mlx-community/gemma-2-2b-it-4bit \
        --adapter outputs/gemma2b-tessy-lora \
        --prompts-file data/sft/tessy-eval-prompts.txt \
        --output outputs/probe-after.jsonl

Requires: pip install mlx-lm  (Apple Silicon only).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


def load_prompts(path: Path) -> list[str]:
    """Read prompts from a .txt (one per line) or a .jsonl (first user turn)."""
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if path.suffix == ".jsonl":
            row = json.loads(line)
            msgs = row.get("messages", [])
            user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            if user:
                prompts.append(user)
        else:
            prompts.append(line)
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None,
                        help="Path to a trained LoRA adapter (omit for base model).")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("mlx-lm is not installed. Run: pip install mlx-lm", file=sys.stderr)
        return 1

    prompts = load_prompts(Path(args.prompts_file))
    if not prompts:
        print("no prompts found", file=sys.stderr)
        return 1

    load_kwargs = {}
    if args.adapter:
        load_kwargs["adapter_path"] = args.adapter
    model, tokenizer = load(args.model, **load_kwargs)

    tag = f"{args.model}" + (f" + {args.adapter}" if args.adapter else " (base)")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for i, prompt in enumerate(prompts, start=1):
            messages = [{"role": "user", "content": prompt}]
            templated = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            answer = generate(
                model, tokenizer, prompt=templated,
                max_tokens=args.max_tokens, verbose=False,
            )
            handle.write(json.dumps(
                {"model": tag, "prompt": prompt, "answer": answer.strip()},
                ensure_ascii=False) + "\n")
            print(f"[{i}/{len(prompts)}] {prompt[:60]}...")

    print(f"\nSaved {len(prompts)} answers to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

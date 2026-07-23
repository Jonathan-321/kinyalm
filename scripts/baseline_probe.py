#!/usr/bin/env python3
"""Probe what a base model produces in Kinyarwanda BEFORE any fine-tuning.

Read-only: loads a model and generates greedily over a fixed set of
tutor-style prompts, then writes the completions to JSONL. Run it on the
same prompts before and after fine-tuning to get a real before/after.

Quantization is automatic (4-bit NF4 on CUDA, full precision on CPU), so a
2B model probes on a CPU laptop and a 9B probes on the rented GPU with the
same command:

    python scripts/baseline_probe.py --model google/gemma-2-2b-it \
        --output baseline-2b.jsonl
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

# Tutor-style prompts, one per SFT task type. English instructions asking
# for Kinyarwanda, chosen to expose code-switching, refusals, and
# hallucination. These are NOT the held-out benchmark prompts, so probing
# does not touch evaluation-only data.
DEFAULT_PROMPTS = [
    "How do I say good morning to someone in Kinyarwanda?",
    "Translate this sentence into Kinyarwanda: I am going to the market.",
    "What does the Kinyarwanda sentence 'Ndashaka amazi' mean in English?",
    "Briefly explain how noun-class prefixes like umu- and aba- work in Kinyarwanda.",
    "Is this correct Kinyarwanda: 'Ndi kwiga Ikinyarwanda'? If not, correct it.",
    "Give me five common Kinyarwanda words for family members with English meanings.",
    "Write one short beginner quiz question for practicing Kinyarwanda greetings.",
    "What is the Kinyarwanda word for 'photosynthesis'? If you are not sure, say so.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--output", required=True,
                        help="JSONL path for {prompt, completion} records")
    parser.add_argument("--prompts-file", default=None,
                        help="Optional text file, one prompt per line; "
                             "defaults to the built-in probe set")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable 4-bit quantization even on CUDA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.prompts_file:
        prompts = [line.strip() for line
                   in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    quantize = use_cuda and not args.no_quant
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"model: {args.model}")
    print(f"device: {'cuda' if use_cuda else 'cpu'}, 4-bit quantization: {quantize}")

    model_kwargs = {"dtype": dtype}
    # Gemma-2 soft-capping needs eager attention for correct outputs; other
    # models keep the Transformers default so the probe stays fast.
    if "gemma-2" in args.model.lower().replace("_", "-"):
        model_kwargs["attn_implementation"] = "eager"
    if quantize:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, start=1):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            print(f"\n[{index}/{len(prompts)}] {prompt}\n  -> {completion}")
            handle.write(json.dumps(
                {"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")

    print(f"\nbaseline probe written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

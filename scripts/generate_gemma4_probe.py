#!/usr/bin/env python3
"""Generate tutor answers from a base model or a base+LoRA adapter.

Mirrors the model-loading path in scripts/train_qlora.py (Gemma 4 loads
through AutoModelForMultimodalLM; other models through AutoModelForCausalLM),
so it runs the same Gemma 4 12B checkpoint the training uses. Writes
{"prompt", "completion"} JSONL compatible with scripts/compare_probes.py.

Run it twice for a before/after: once on the base model, once with --adapter,
then compare_probes.py renders the comparison.

    python scripts/generate_gemma4_probe.py \
        --model google/gemma-4-12B-it \
        --prompts-file configs/training/gemma4-eval-prompts.txt \
        --output base.jsonl

    python scripts/generate_gemma4_probe.py \
        --model google/gemma-4-12B-it \
        --adapter kinyalm/kinyalm-gemma-4-12b-experimental \
        --prompts-file configs/training/gemma4-eval-prompts.txt \
        --output adapter.jsonl
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

DEFAULT_SYSTEM_PROMPT = (
    "You are a bilingual Kinyarwanda-English language tutor. Answer the "
    "learner's request directly, accurately, and naturally. Follow any "
    "requested response language. If no response language is requested, use "
    "the language that best teaches the requested concept. State uncertainty "
    "instead of inventing a grammar rule, translation, cultural claim, or fact."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--adapter", default=None,
                        help="Optional PEFT/LoRA adapter repo id or local path")
    parser.add_argument("--prompts-file", required=True,
                        help="Text file with one prompt per line")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable 4-bit quantization even on CUDA")
    return parser.parse_args()


def resolve_attention_implementation(model: str) -> str | None:
    normalized = model.lower().replace("_", "-")
    if "gemma-4" in normalized or "gemma-2" in normalized or "gemma-3" in normalized:
        return "eager"
    return None


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    quantize = use_cuda and not args.no_quant
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    print(f"device: {'cuda' if use_cuda else 'cpu'}, dtype: {dtype}, 4-bit: {quantize}")

    revision_kwargs = {"revision": args.model_revision} if args.model_revision else {}
    model_kwargs = {"dtype": dtype}
    attn = resolve_attention_implementation(args.model)
    if attn:
        model_kwargs["attn_implementation"] = attn
    if quantize:
        from transformers import BitsAndBytesConfig

        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, **revision_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model, **revision_kwargs)

    if getattr(config, "model_type", "") == "gemma4_unified":
        from transformers import AutoModelForMultimodalLM as Loader
    else:
        Loader = AutoModelForCausalLM
    model = Loader.from_pretrained(
        args.model, config=config, **revision_kwargs, **model_kwargs
    )

    if args.adapter:
        from peft import PeftModel

        print(f"loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer, system_prompt: str, prompt: str, device):
    """Prepend the system instruction to the user turn (robust across templates)."""

    content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)


def main() -> int:
    args = parse_args()
    prompts = [line.strip() for line
               in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts_file}")
    print(f"{len(prompts)} prompts")

    import torch

    model, tokenizer = load_model_and_tokenizer(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", buffering=1) as handle:
        for index, prompt in enumerate(prompts, start=1):
            inputs = build_inputs(tokenizer, args.system_prompt, prompt, model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            handle.write(json.dumps(
                {"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(prompts)}] done")

    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the project QLoRA SFT training described in docs/model/sft-run-plan.md.

Defaults match the run plan's QLoRA starting settings, so the launch command
needs no hyperparameter flags:

    python scripts/train_qlora.py \
        --train-file data/sft/train.jsonl \
        --eval-file data/sft/validation.jsonl \
        --output-dir outputs/run-001

Every input row is validated against the project SFT schema before anything
is downloaded or trained. Rows must be split=train (or validation for the
eval file) and pass the review/source training gate.

Quantization is automatic: 4-bit NF4 on CUDA, full precision elsewhere, so
the same script smoke-tests on a CPU laptop with a tiny model:

    python scripts/train_qlora.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --train-file smoke.jsonl --output-dir outputs/smoke \
        --epochs 1 --max-seq-len 256 --grad-accum 1
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.sft import load_jsonl, validate_sft_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-prompts-file", default=None,
                        help="Optional text file, one prompt per line; greedy "
                             "samples are written to the output dir after training")
    # Defaults below mirror docs/model/sft-run-plan.md. Override only for
    # smoke tests or an approved plan change.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable 4-bit quantization even on CUDA")
    return parser.parse_args()


def load_split(path: str, allowed_splits: set[str]) -> list[dict]:
    """Load, validate, and split-filter one SFT JSONL file."""

    records = load_jsonl(path)
    results = validate_sft_records(records)
    failures = [result for result in results if not result.ok]
    if failures:
        for result in failures:
            for error in result.errors:
                print(f"{path} line {result.line_number}: {error}", file=sys.stderr)
        raise SystemExit(f"schema validation failed for {path}: "
                         f"{len(failures)} bad rows. Fix the data, not the script.")

    kept = [r for r in records if r["split"] in allowed_splits]
    dropped = len(records) - len(kept)
    if dropped:
        print(f"{path}: dropped {dropped} rows outside splits {sorted(allowed_splits)}")
    if any(r["split"] == "benchmark-only" for r in records):
        raise SystemExit(f"{path} contains benchmark-only rows; "
                         "benchmark prompts must never reach training.")
    if not kept:
        raise SystemExit(f"{path}: no usable rows after split filtering")
    return kept


def to_dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_list([{"messages": r["messages"]} for r in records])


def main() -> int:
    args = parse_args()

    train_records = load_split(args.train_file, {"train"})
    eval_records = load_split(args.eval_file, {"validation"}) if args.eval_file else None
    print(f"train rows: {len(train_records)}"
          + (f", validation rows: {len(eval_records)}" if eval_records else ""))

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    use_cuda = torch.cuda.is_available()
    quantize = use_cuda and not args.no_quant
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"device: {'cuda' if use_cuda else 'cpu'}, 4-bit quantization: {quantize}")

    model_kwargs = {
        "dtype": dtype,
        # Required for Gemma-2: logit soft-capping is incompatible with
        # flash/sdpa attention during training.
        "attn_implementation": "eager",
    }
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

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_len,
        gradient_checkpointing=use_cuda,
        optim="paged_adamw_8bit" if quantize else "adamw_torch",
        bf16=use_cuda,
        logging_steps=1,
        eval_strategy="epoch" if eval_records else "no",
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=to_dataset(train_records),
        eval_dataset=to_dataset(eval_records) if eval_records else None,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(args.output_dir)
    print(f"final train loss: {result.training_loss:.4f}")
    print(f"adapter saved to: {args.output_dir}")

    if args.sample_prompts_file:
        prompts = [line.strip() for line in
                   Path(args.sample_prompts_file).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        samples_path = Path(args.output_dir) / "samples.jsonl"
        trained = trainer.model
        trained.eval()
        with samples_path.open("w", encoding="utf-8") as handle:
            for prompt in prompts:
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(trained.device)
                with torch.no_grad():
                    output = trained.generate(
                        **inputs, max_new_tokens=200, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                prompt_length = inputs["input_ids"].shape[-1]
                completion = tokenizer.decode(
                    output[0][prompt_length:], skip_special_tokens=True
                )
                handle.write(json.dumps(
                    {"prompt": prompt, "completion": completion},
                    ensure_ascii=False) + "\n")
        print(f"samples written to: {samples_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

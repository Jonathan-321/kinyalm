# Tessy's Distillation-Queue Review Contribution

Owner: Tessy Mugisha

## What this is

My reviewed slice of the 1,000-row distillation queue (326 conversations,
all flagged **Keep** — I judged the originals natural and declined the
critic's revisions where I disagreed).

Two forms are committed:

1. **Review artifact (audit trail)** — `data/reviewed/tessy_distillation_queue.jsonl`
   The raw review export: one row per conversation with `my_flag`,
   `critic_feedback`, and `suggested_revision`. This is the record of *what I
   decided and why*, not training data.

2. **Trainable SFT files** — `data/sft/tessy-distill-review.train.jsonl`
   and `data/sft/tessy-distill-review.validation.jsonl`
   Produced by `scripts/convert_distillation_review_to_sft.py`. The multi-turn
   conversations are split into adjacent (user, assistant) pairs to match the
   first-run SFT schema (exactly two messages per row). Result: **561 rows
   (512 train / 49 validation)**, all passing `scripts/validate_sft_jsonl.py`.

## Regenerate the SFT files

```bash
python3 scripts/convert_distillation_review_to_sft.py \
    --review-jsonl data/reviewed/tessy_distillation_queue.jsonl \
    --out-prefix data/sft/tessy-distill-review \
    --reviewer "Tessy Mugisha" \
    --train-ratio 0.9

python3 scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.train.jsonl
python3 scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.validation.jsonl
```

## task_type mapping note

The distillation `task_family` labels were mapped onto the project's allowed
`task_type` set (see the table in the converter script). A few are approximate
(e.g. `pronunciation-and-orthography` -> `grammar-explanation`). The mapping is
explicit and auditable in the script; the team can refine labels without
regenerating the text.

## Run a local fine-tune (my M2 MacBook Air, 24 GB)

CPU smoke test (tiny model, proves the loop end to end — minutes, any laptop):

```bash
python3 scripts/train_qlora.py \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --train-file data/sft/tessy-distill-review.train.jsonl \
    --eval-file data/sft/tessy-distill-review.validation.jsonl \
    --output-dir outputs/tessy-smoke \
    --epochs 1 --max-seq-len 512 --batch-size 1 --grad-accum 1 \
    --sample-prompts-file data/sft/tessy-smoke-prompts.txt
```

Real local run on Apple Silicon uses `mlx-lm` instead (the repo's
`train_qlora.py` only 4-bit-quantizes on CUDA; on Mac it would run fp32 on CPU):

```bash
pip install mlx-lm
mlx_lm.lora --model google/gemma-2-2b-it --train \
    --data data/sft --iters 300 --batch-size 1
```

(Requires an HF token with accepted Gemma access.)

# Tessy's Distillation-Queue Review Contribution

Owner: Tessy Mugisha

## What this is

My reviewed slice of the 1,000-row distillation queue contains 326
conversations, all flagged **Keep** after fluent review. Of those, 288 also
received `Critic accepted`; those form the current trainable set. The remaining
38 received `Repair first` from the critic and remain in the audit artifact for
a second adjudication before training. This keeps the human decisions intact
without silently promoting rows where the two review signals disagree.

Two forms are committed:

1. **Review artifact (audit trail)** — `data/reviewed/tessy_distillation_queue.jsonl`
   The raw review export: one row per conversation with `my_flag`,
   `critic_feedback`, and `suggested_revision`. This is the record of *what I
   decided and why*, not training data.

2. **Trainable SFT files** — `data/sft/tessy-distill-review.train.jsonl`
   and `data/sft/tessy-distill-review.validation.jsonl`
   Produced by `scripts/convert_distillation_review_to_sft.py`. Each multi-turn
   conversation remains intact, and assignment happens at conversation level
   so related turns cannot leak across splits. Result: **288 conversations
   (258 train / 30 validation)**, all passing
   `scripts/validate_sft_jsonl.py`.

## Regenerate the SFT files

```bash
python3 scripts/convert_distillation_review_to_sft.py \
    --review-jsonl data/reviewed/tessy_distillation_queue.jsonl \
    --out-prefix data/sft/tessy-distill-review \
    --reviewer "Tessy Mugisha" \
    --train-ratio 0.9 \
    --mlx-data-dir data/mlx-data

python3 scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.train.jsonl
python3 scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.validation.jsonl
```

`data/mlx-data/` is ignored by Git and receives the filenames MLX-LM expects:
`train.jsonl` and `valid.jsonl`.

Do not use `--accept-disputed-keeps` unless the team has explicitly adjudicated
the 38 `Repair first` rows. The option exists to record that decision, not to
bypass it.

## task_type mapping note

The distillation `task_family` labels were mapped onto the project's allowed
`task_type` set (see the table in the converter script). Reading comprehension,
sentence generation, and pronunciation retain their dedicated task labels.
The mapping is explicit and auditable in the script.

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
pip install mlx-lm==0.31.3
python -m mlx_lm lora \
    --model mlx-community/gemma-2-2b-it-4bit \
    --train \
    --data data/mlx-data \
    --adapter-path outputs/gemma2b-tessy-lora \
    --iters 300 \
    --batch-size 1
```

This uses the public MLX conversion and the local staging files generated in
the previous step.

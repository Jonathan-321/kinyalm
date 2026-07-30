# Local 2B Fine-Tune — Before/After Result (Tessy)

Owner: Tessy Mugisha
Date: 2026-07-27
Machine: MacBook Air, Apple M2, 24 GB

## What ran

After confirming the 12B could not be fine-tuned locally (see
[`local-12b-notes.md`](local-12b-notes.md)), I ran the full pipeline on the
supported **`mlx-community/gemma-2-2b-it-4bit`** base using our reviewed data.

```bash
mlx_lm.lora --model mlx-community/gemma-2-2b-it-4bit --train \
  --data data/mlx-data --iters 300 --batch-size 1 --num-layers 8 \
  --max-seq-length 512 --adapter-path outputs/gemma2b-tessy-lora
```

- Data: 512 train / 49 validation (my reviewed distillation slice, SFT-converted)
- Trainable params: 3.2M (0.12% of the model, LoRA)
- **Validation loss: 5.48 → 2.77 → 2.61** across 300 iters
- **Peak memory: ~3 GB** (24 GB was never the limit for the 2B)
- Trained in a few minutes; adapter saved to `outputs/gemma2b-tessy-lora`

## Evidence status

These are the contributor's recorded results. The adapter, training log, exact
environment lock, and the local `outputs/probe-2b-before.jsonl` /
`outputs/probe-2b-after.jsonl` files were not committed, so another teammate
cannot yet reproduce the reported metrics exactly.

The 512/49 pair-level dataset used for this experiment is preserved in the
Hugging Face incoming area. It is not the current canonical training set:
GitHub contains the stricter 258/30 conversation-level split where 38
critic-disputed conversations remain withheld. Treat this run as useful
exploratory evidence, not as a final comparable bake-off result.

## What the before/after actually shows

The training succeeded mechanically (loss dropped, fit in memory), but the
**generated answers got worse, not better** — the fine-tuned model collapses
into repetition.

Example — "How do you say 'good morning, teacher' in Kinyarwanda?"
- Before: "**Ubuse abantu**" — coherent but wrong (not real Kinyarwanda).
- After: "Mbere y'umugoroba, Mbere y'umugoroba." — repeats itself, still wrong.

Example — "Translate: 'My sister is reading a book.'"
- Before: one wrong sentence, at least well-formed.
- After: "...y'isoko y'isoko y'isoko..." — degenerates into an endless loop.

## Why (plain terms)

1. The **2B base barely knows Kinyarwanda** — the "before" answers are already
   wrong (the "made-up language" we saw earlier).
2. Fine-tuning a small 4-bit model hard on 512 examples **overfits** — it
   memorizes patterns and loses the ability to form sentences. Lower loss on the
   training set does NOT mean better answers.

## Takeaway for the team

- The contributor reports that the full local loop ran: reviewed data, SFT
  conversion, LoRA fine-tune, and a before/after probe.
- The **2B is too weak a base** for quality Kinyarwanda, even after fine-tuning.
  This confirms the decision to fine-tune the **12B on the cloud GPU** (Bonheur's
  Lambda A100), where our data is already staged in the data-lake.
- If we ever want a usable 2B, next levers would be: fewer iters (try ~100),
  lower learning rate, and more/cleaner data — but the 12B is the real path.

# Gemma 4 12B Gate 2 Corrected-Objective Run — 2026-08-08

## Summary

Full one-epoch QLoRA fine-tune of Gemma 4 12B on the critic-accepted data,
using the corrected assistant-only loss objective. This is Gate 2 from the
[adapter recovery plan](../gemma4-adapter-recovery-plan.md): the corrected-
objective control that isolates the training-objective fix from the rejected
2026-08-03 run.

It is an experimental baseline, not a production tutor. The training data is
model-critic-accepted, not fluent-human-approved, so this measures training
lift, not release quality.

## What this run changes from 2026-08-03

The rejected 2026-08-03 run used two epochs at learning rate 2e-4 with no
assistant-only masking, and it collapsed into repetition. This run applies the
corrections now on main: assistant-only loss (`completion_only_loss=True`), one
epoch, and learning rate 5e-5. Same base model and same dataset revision, so the
objective change is isolated.

## Setup

| Item | Value |
| --- | --- |
| Model | `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Data | `kinyalm/kinyalm-data-lake` @ `754a58b021cfe1e505f432df0de45ce2f63a3b21` |
| Data tier | `experimental-critic-filtered`; not human-reviewed |
| Split | 776 train / 87 validation (1,395 / 144 supervised assistant turns) |
| Hardware | 1x A100 40 GB SXM4 (Lambda, Virginia) |
| Stack | Transformers 5.14.1, TRL 1.9.2, PyTorch 2.13.0 |
| Quantization | 4-bit NF4, bf16 compute |
| LoRA | r=16, alpha=32, dropout=0.05, attention + MLP projections |
| Schedule | 1 epoch, 175 steps, lr 5e-5 cosine, warmup ratio 0.03 |
| Loss scope | assistant-completions-only (corrected objective) |
| Seed | 42 |

## Results

| Metric | Value |
| --- | --- |
| Training runtime | 1,405 s (~23 min) |
| Final training loss | 1.816 (last epoch step ~1.09) |
| Validation loss | 1.354 |
| Validation token accuracy | 0.702 |
| Adapter size | ~250 MiB (`adapter_model.safetensors`, 262,373,216 bytes) |

Loss fell cleanly from ~6 to ~1 with no divergence. These numbers match the
2026-07-31 run (val loss ~1.30, token accuracy ~0.70), but this run has the
corrected assistant-only objective the earlier one lacked.

Published adapter (private, organization-only):
`kinyalm/kinyalm-gemma-4-12b-gate2-corrected`
Commit `cd88117b050506a6d3a94a23ab83d9523febccc8`.

## Open item to verify

The adapter here is ~250 MiB. The 2026-07-31 and 2026-08-03 adapters were
~126 MiB with the same rank and target modules. Worth confirming why this one is
larger (save dtype or saved modules) before comparing artifacts directly.

## Next steps

1. Base-vs-adapter eval: run this adapter and the unchanged base on the same
   30-prompt set (`scripts/generate_gemma4_probe.py` + `scripts/compare_probes.py`)
   to quantify the lift, not eyeball it.
2. Gate 3: freeze a fluent-human-approved split, fold in Bonheur's correction
   batch, and retrain with the same config.
3. Keep it reproducible: model and data revisions are pinned above and in the
   run's preflight manifest.
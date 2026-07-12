# Week 2 Update

Date: 2026-07-12

## Summary

The project has moved from setup into SFT readiness.

SFT means supervised fine-tuning: training a base model on reviewed
user/assistant examples so it behaves like a Kinyarwanda learning tutor.

## Completed

- Tessy added approval notes and attribution for the two approved
  DigitalUmuganda sources.
- Bonheur expanded the tokenizer evaluation set to 38 examples:
  - 37 confirmed examples,
  - 1 deliberate not-natural hyphen boundary case,
  - noun-class pairs,
  - verb morphology examples,
  - apostrophe and elision examples.
- Jonathan expanded the tutor benchmark to 50 prompts:
  - 26 prompts are marked `benchmark-only`,
  - benchmark rows must not be copied into SFT training data.
- Jonathan added the SFT JSONL schema and validator.
- Jonathan added an open Kinyarwanda benchmark shortlist covering Belebele,
  FLORES-200, IrokoBench, AfriSenti-SemEval, and KINNEWS.
- Jonathan added the data overhaul plan: 100 reviewed examples for the seed
  gate, then about 1,000 reviewed examples for a serious SFT run.
- Jonathan added OSCER smoke-test scripts and quickstart notes.
- Jonathan drafted the first SFT run plan.

## Current Gates

| Gate | Status | Notes |
| --- | --- | --- |
| Source approval | partial | TTS and MT sources approved; remaining sources need SFT-use decisions. |
| Tokenizer examples | ready for first analysis | Bonheur's examples are merged into this branch. |
| Tutor benchmark | ready | 50 prompts, with 26 held out for evaluation. |
| External benchmark list | ready for first loader | Benchmark sources are logged as evaluation-only. |
| SFT schema | ready | `docs/data/sft-data-schema.md` defines the JSONL format. |
| Seed SFT data | not ready | 100 reviewed examples still need to be authored and checked. |
| Useful SFT pack | not ready | About 1,000 reviewed examples are needed before a serious tutor SFT run. |
| Compute path | partial | OSCER scripts exist; login/authentication still needs to work. |

## Main Blockers

1. The team still needs reviewed SFT examples.
2. Bonheur still needs to finalize the base-model choice.
3. Tessy still needs to define which sources are SFT-usable, reference-only, or
   blocked.
4. OSCER login needs to be fixed before GPU jobs can run there.

## Next Week Target

By the next checkpoint, the team should have either:

1. a tiny QLoRA SFT run completed on reviewed seed conversations, or
2. a clear written blocker explaining what stopped training.

The highest-value next task is creating and reviewing the first 100
conversation examples in `data/sft/seed_conversations.jsonl`, while using
`docs/data/data-overhaul-plan.md` to scale toward 1,000 reviewed rows.

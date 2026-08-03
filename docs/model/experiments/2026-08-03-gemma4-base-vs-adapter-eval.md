# Gemma 4 12B: Base vs Fine-Tuned Adapter — 2026-08-03

## Summary

First side-by-side evaluation of the experimental adapter against the base
model. Both were run on the same 30-prompt tutor set on one Lambda A100 40 GB,
then compared. The adapter is more concise and more tutor-shaped and improves
on some tasks, but the specific errors already caught in fluent-speaker review
are still present — because those human corrections have not yet been retrained
in. This run is the "before" the next correction+retrain cycle measures against.

## Setup

| Item | Value |
| --- | --- |
| Base | `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Adapter | `kinyalm/kinyalm-gemma-4-12b-experimental` (trained 2026-07-31) |
| Prompts | `configs/training/gemma4-eval-prompts.txt` (30 tutor prompts) |
| Generation | `scripts/generate_gemma4_probe.py`, 4-bit NF4, greedy, 256 max new tokens |
| Hardware | 1x A100 40 GB, ~20 min, ~$0.70 |
| Raw outputs | `assets/2026-08-03-gemma4-base-vs-adapter/{base,adapter}.jsonl` |

Render the full side-by-side with:

```bash
python scripts/compare_probes.py \
  --before docs/model/experiments/assets/2026-08-03-gemma4-base-vs-adapter/base.jsonl \
  --after  docs/model/experiments/assets/2026-08-03-gemma4-base-vs-adapter/adapter.jsonl \
  --before-label "Base Gemma-4-12B-it" --after-label "Fine-tuned adapter" \
  --output base-vs-adapter.md --html
```

## What the adapter improved

- **Conciseness / register.** Base over-explains in English and pads every
  answer; the adapter answers directly, in the tutor register.
- **Translation.** "The children are playing outside near the house" — base:
  wordy, `Abana baturiza...` plus a long breakdown. Adapter:
  `Abana barimo gukina hanze hafi y'inzu.` (concise, natural verb `gukina`).
- **Elision correction.** For `...umuceri na ifi`, the base wrongly declares
  the sentence correct; the adapter catches the error and fixes it to
  `n'ifi`, explaining the vowel meeting.

## What is still wrong (feeds the next correction batch)

- **Polysemy (`gusoma`).** The adapter still gives only "reading" senses and
  misses the "kiss" meaning — the exact error corrected in fluent-speaker
  batch 1, which has not been retrained in yet.
- **Over-correction trap.** On a sentence that is already correct, the adapter
  produces a garbled "correction" (`Kosora ejo ihinduka ejo`) instead of
  saying there is no error. The base fails differently, inventing an
  adjective error.
- **Uncertainty.** On an unanswerable term (`photosynthesis`), the base
  hallucinates; the adapter gives a descriptive coinage but does not state
  uncertainty as the prompt requested.

## Reading

The fine-tune clearly shifted the model toward concise tutor behavior and
helped on several correction and translation tasks. It did not fix the errors
already documented in fluent-speaker review, which is expected: the adapter was
trained on synthetic critic-accepted data with the earlier objective, and the
eight human corrections in
`incoming/bonheurbyiringiro/gemma4-corrections-batch1/` are not in it yet.

## Next

1. Mine these 30 adapter outputs for the still-wrong answers → fluent-speaker
   correction batch 2.
2. Retrain the adapter with corrections folded in, using the assistant-only
   objective now on `main`.
3. Re-run this exact 30-prompt evaluation and compare against this baseline to
   measure the lift.

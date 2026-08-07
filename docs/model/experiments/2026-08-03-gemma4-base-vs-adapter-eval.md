# Gemma 4 12B: Base vs Fine-Tuned Adapter — 2026-08-03

## Summary

First side-by-side evaluation of the experimental adapter against the base
model. Both were run on the same 30-prompt tutor set on one Lambda A100 40 GB.
The adapter changed the response style and improved several individual answers,
but the raw outputs do not establish an overall quality improvement. They also
contain catastrophic repetition and elementary meaning errors. This checkpoint
therefore fails the demo gate and remains only a baseline for the corrected
training run.

## Setup

| Item | Value |
| --- | --- |
| Base | `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Adapter | `kinyalm/kinyalm-gemma-4-12b-experimental@feefb1e7ac359b60ca45af9db8fd883af8cac933` |
| Prompts | `configs/training/gemma4-eval-prompts.txt` (30 tutor prompts) |
| Generation | `scripts/generate_gemma4_probe.py`, 4-bit NF4, greedy, 256 max new tokens |
| Hardware | 1x A100 40 GB, ~20 min, ~$0.70 |
| Raw outputs | `assets/2026-08-03-gemma4-base-vs-adapter/{base,adapter}.jsonl` |

This was an unblinded engineering comparison, not a scored native-speaker
study. The generator concatenated the system instruction into the user message
instead of using a separate system role. It also saved only prompts and
completions, so finish reasons, resolved adapter revision, runtime versions,
and timing cannot be recovered from the raw files alone. Both candidates saw
the same construction, but this run should not be treated as final parity with
the production chat path.

Render the full side-by-side with:

```bash
python scripts/compare_probes.py \
  --before docs/model/experiments/assets/2026-08-03-gemma4-base-vs-adapter/base.jsonl \
  --after  docs/model/experiments/assets/2026-08-03-gemma4-base-vs-adapter/adapter.jsonl \
  --before-label "Base Gemma-4-12B-it" --after-label "Fine-tuned adapter" \
  --output base-vs-adapter.md --html
```

## Observed improvements

- **Conciseness / register.** Base over-explains in English and pads every
  answer; the adapter answers directly, in the tutor register.
- **Translation.** "The children are playing outside near the house" — base:
  wordy, `Abana baturiza...` plus a long breakdown. Adapter:
  `Abana barimo gukina hanze hafi y'inzu.` (concise, natural verb `gukina`).
- **Elision correction.** For `...umuceri na ifi`, the base wrongly declares
  the sentence correct; the adapter catches the error and fixes it to
  `n'ifi`, explaining the vowel meeting.

These are prompt-level observations. They are not an aggregate win rate because
the outputs were not randomized, blinded, or scored.

## Release-blocking failures

- **Catastrophic repetition.** The `urukundo` response repeats `rurushaho`
  until the 256-token generation limit. A direct PEFT run therefore reproduces
  the same failure class seen later in MLX.
- **Basic translation.** The adapter translates `ashonje` as "cold" instead
  of "hungry."
- **Noun and number errors.** It says the singular of `amaso` is `amaso` and
  changes `Abana bane` (four children) to `Abana babiri` (two children).
- **Tense explanation.** It presents `Nakora` as a completed past-tense form.
- **Task noncompliance.** The greeting quiz does not supply four greeting
  choices, and the noun-class quiz is unrelated and malformed.
- **Polysemy (`gusoma`).** It gives only reading-related senses and misses the
  "kiss" meaning already caught in fluent-speaker review.
- **Over-correction trap.** It produces `Kosora ejo ihinduka ejo` instead of
  explaining whether the supplied sentence is correct.
- **Uncertainty.** It invents a descriptive term for `photosynthesis` without
  stating uncertainty as requested.

## Reading

The fine-tune clearly shifted the model toward shorter, tutor-shaped responses.
That style change is not the same as a quality improvement. The adapter is
better on some rows, worse on others, and unusable on at least one row because
of repetition. The base is also inaccurate on many prompts, so beating selected
base answers would not by itself make the adapter good enough.

The direct PEFT repetition materially reduces the likelihood that MLX tensor
conversion is the primary cause of the collapse. An exact BF16-versus-MLX
parity run is still useful for implementation measurement, but it is no longer
a prerequisite for rejecting this adapter or starting the corrected control.

## Next

1. Keep these 30 rows as failure-mining input, not as a release score.
2. Run the one-epoch corrected-objective control on the immutable split under a
   new adapter repository or run ID.
3. Evaluate the unchanged base, corrected control, and later curated adapter on
   the held-out task bank with blinded native-speaker scoring.
4. Preserve this 30-prompt set as a secondary regression set and rerun it after
   every accepted training change.

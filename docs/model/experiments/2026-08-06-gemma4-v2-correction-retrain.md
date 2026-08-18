# Gemma 4 12B v2: Correction Retrain — 2026-08-06

## Summary

Closed the correct -> retrain -> measure loop. The v1 experimental adapter's
mistakes were reviewed by a fluent speaker, corrected, folded back into
training, and the retrained v2 adapter was evaluated on the same 30 prompts.
**Six of eight targeted errors are fixed, matching the corrections almost
verbatim.** This is the first end-to-end demonstration that human corrections
measurably improve the model.

## Method

- Reviewed v1's outputs and wrote **16 fluent-speaker corrections** (batches 1
  and 2), contributed to the data lake under
  `incoming/bonheurbyiringiro/gemma4-corrections-batch1|2/`.
- Folded them into the training set: 776 critic-accepted `experimental-train`
  rows + the 16 corrections **upweighted 4x** (64 instances, unique ids) =
  **840 train / 87 validation**. The two data tiers keep honest labels
  (experimental critic-accepted vs human-approved).
- This required a small `train_qlora.py` change (in this same PR): under
  `--experimental`, also load human-approved `train`/`validation` rows, so a
  retrain can mix the experimental baseline with corrections. Backward
  compatible: existing experimental packages contain only experimental rows.
- Retrained `google/gemma-4-12B-it` with the assistant-only recipe now on main
  (lr 5e-5, 1 epoch, eager attention) on one Lambda A100 40 GB (~40 min).

## Results

| Metric | v1 (2026-07-31) | v2 (this run) |
| --- | --- | --- |
| Recipe | full-sequence, lr 2e-4, 2 epochs | assistant-only, lr 5e-5, 1 epoch |
| Train rows | 776 | 840 (776 + 16 corrections x4) |
| Final eval loss | 1.30 | 1.34 |
| Eval token accuracy | ~0.70 | ~0.71 |

v2 adapter (private): `kinyalm/kinyalm-gemma-4-12b-experimental-v2`.
Raw v1/v2 outputs on the 30-prompt set:
`assets/2026-08-06-gemma4-v1-vs-v2/adapter-v{1,2}.jsonl`.

Eval loss is not the headline here; the targeted before/after is.

## The lift (v1 -> v2 on corrected prompts)

Fixed, matching the corrections:

| Prompt | v1 | v2 |
| --- | --- | --- |
| "Good evening, did you sleep well?" | `Mwaramutse ... wakoze neza` | `Mwiriwe, waraye neza?` |
| "Please help me carry this heavy bag" | garbled meta-question | actual translation |
| `Umwana ... ashonje` | "cold" | "hungry" |
| urukundo/urwango | `kurushaho` repeat loop | clean structured definition |
| `amaso` singular/plural | both "amaso" | singular `ijisho` |
| respond to elder's "urakoze" | wrong response | `Murakoze` |

Partial or not moved:

- **Over-correction trap** (a sentence that is already correct): v2 now
  acknowledges the flagged word is fine but still hunts for an error instead of
  answering "nta kosa." Better, not fixed. A single upweighted example did not
  fully retrain this behavior.
- **`gusoma` = kiss**: did not surface on this eval prompt, which asks about
  usage *contexts* and was read as reading contexts. The corrected prompt used
  different wording ("can it have more than one meaning"). This is a
  prompt-generalization gap, not a plain miss.

## Reading

Targeted human corrections, even ~16 of them, produce a clear and specific
lift on exactly the items corrected. The two that did not fully move point at
the next iteration: cover prompt-phrasing variants, and add more examples for
subtle behaviors like refusing to over-correct.

## Next

1. Correction batch 3 focused on phrasing variants and more over-correction
   traps.
2. Continue accumulating fluent-human-approved rows so the base training tier
   improves, not just the correction patches.
3. Re-run the 30-prompt eval after each retrain to keep measuring the lift.

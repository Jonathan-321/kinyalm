# Gemma 4 12B QLoRA Full Run — 2026-07-31

## Summary

The first full (multi-epoch) QLoRA fine-tune of Gemma 4 12B on the KinyaLM
critic-accepted data completed on a rented Lambda A100 40 GB. This is the
follow-up to the one-step infrastructure smoke in
[`../gemma4-qlora-smoke-2026-07-30.md`](../gemma4-qlora-smoke-2026-07-30.md):
the smoke proved the pipeline loads and steps; this run actually trains.

It is an **experimental baseline**, not a production tutor. The training data is
model-critic-accepted, not fluent-human-approved, so the result is meant to
measure the training lift and expose failure modes, not to ship.

## Setup

| Item | Value |
| --- | --- |
| Model | `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Data | `kinyalm/kinyalm-data-lake` critic-accepted package |
| Split | 776 experimental-train / 87 experimental-validation |
| Hardware | 1x A100 40 GB SXM4 (Lambda), CUDA 12.8, driver 580 |
| Stack | Transformers 5.14.1, TRL 1.9.2, PEFT 0.20.0 |
| Quantization | 4-bit NF4, bf16 compute |
| LoRA | r=16, alpha=32, dropout=0.05, attention + MLP projections |
| Schedule | 2 epochs (194 steps), lr 2e-4 cosine, warmup 0.03 |
| Attention | eager (required for the Gemma family) |
| Launch | `MODEL_PROFILE=gemma4 ALLOW_EXPERIMENTAL_FULL_RUN=1 scripts/cloud/submit_lambda_job.sh` |

## Results

| Metric | Value |
| --- | --- |
| Training runtime | 1569 s (~26 min) |
| Final training loss | 1.60 (trended to ~1.0 by end of epoch 2) |
| Validation loss | 1.298 |
| Validation token accuracy | 0.697 |
| Adapter size | ~126 MB |

Loss fell cleanly from the ~7.4 seen in the one-step smoke to ~1.0 with no
divergence. Token accuracy reached roughly 0.70 on the held-out validation
rows.

Published adapter (private, organization-only):
`kinyalm/kinyalm-gemma-4-12b-experimental`.

Total cost was approximately one instance-hour, about **$2**.

## Quality read

A fluent Kinyarwanda speaker reviewed the 12 post-training sample generations.

What improved clearly: the adapter produces **fluent, tutor-shaped
Kinyarwanda**, a large step up from the base model. Examples that read well:

- Defining *ubupfura* (courtesy/dignity) with two example sentences.
- A correct distinction between *kubona* (to get/perceive) and *kureba*
  (to look with the eyes), with an example.

What is still wrong: the model makes **real accuracy errors** despite fluent
surface form.

- On a sentence-correction task it declared *"Ejo nzagiye ku ishuri"* correct
  and then produced a circular explanation, rather than fixing the tense.
- It translated "introduce myself" as *nsubiramo* ("I repeat") instead of a
  correct form such as *kwivuga* / *kwibwira*.

This pattern — fluent form, imperfect accuracy — is consistent with training on
synthetic, model-critic-accepted data. The model learned the tutor register
well; correctness needs fluent-human-approved training rows.

## Next steps

1. Native-speaker evaluation: base Gemma 4 12B vs this adapter on the same
   prompts, to quantify the lift rather than eyeball it
   (`scripts/baseline_probe.py` + `scripts/compare_probes.py`).
2. Retrain on the fluent-human-approved tier as reviewed rows accumulate in the
   data lake, then compare against this experimental baseline.
3. Keep the run reproducible: model and data revisions are pinned above and in
   the run's preflight manifest.

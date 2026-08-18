# KinyaLM Roadmap

**Updated:** August 17, 2026

## Current Position

The initial class project is complete as a research and demonstration pipeline:

- Track A reproduced tokenizer, causal-language-model training, checkpointing,
  loss, perplexity, and generation on small Kinyarwanda experiments.
- Track B produced reviewed data workflows, reproducible Gemma 4 QLoRA runs,
  published adapter artifacts, blinded evaluation, MLX conversion, and a local
  streaming browser interface.
- The 1,560 cumulative-step targeted continuation is the best KinyaLM adapter,
  but the unchanged base still leads the current 150-prompt development screen.
- Model-assisted results have identified useful targeted gains and clear
  regressions; a new hidden native-reviewed final benchmark is still required.

The next phase is not simply "train longer." It is to freeze reliable evidence,
repair text-model regressions, and add modalities through separately testable
components.

## Priority 1: Clean Research Release

Deliverables:

- one current README and reproducible local launcher;
- a complete experiment ledger with immutable model and dataset revisions;
- preserved teammate reports and contribution history;
- a release-candidate data card and model card;
- no unresolved pull-request conflicts or duplicated superseded workflows;
- passing tests for every supported data, training, evaluation, and serving
  path.

Exit condition: a new contributor can reproduce the local base-versus-adapter
comparison without relying on private instructions.

## Priority 2: Final Text Evaluation

1. Create a hidden prompt set that was not used for training or development.
2. Cover conversation, uncertainty, multi-turn memory, correction, both
   translation directions, grammar, and hallucination resistance.
3. Run unchanged base and selected checkpoints with identical prompts, system
   messages, decoding, and runtime.
4. Blind model identity and collect native-speaker scores plus exact rewrites.
5. Report paired outcomes, category scores, repetition, and reviewer agreement.

Exit condition: the project can make a supported claim about whether an adapter
beats the base, where it helps, and where it regresses.

## Priority 3: Text Adapter Recovery

Use the final development evidence to build a small correction mixture rather
than adding generic volume:

- retain targeted morphology, correction, and Kinyarwanda-to-English gains;
- add high-quality conversation, uncertainty, greetings, context retention,
  and English-to-Kinyarwanda examples;
- remove template phrases that correlate with repetition;
- compare preserved intermediate checkpoints, not only the final loss;
- promote only after native review on the hidden set.

Exit condition: one adapter improves the paired native-reviewed score without a
material repetition regression.

## Priority 4: Speech Input

Build the first multimodal extension as a modular Kinyarwanda ASR benchmark and
push-to-talk interface. Compare Omnilingual ASR and MMS on at least 100
native-reviewed recordings, then expose an editable transcript before sending
text to KinyaLM.

Exit condition: ASR word and character error rates, latency, subgroup errors,
and 100 reviewer decisions are recorded with pinned model revisions.

## Priority 5: Spoken And Visual Lessons

After speech input passes:

- add a research TTS baseline with explicit license boundaries;
- build a limited pronunciation-retry workflow;
- create a grounded image/OCR benchmark for signs, worksheets, and common
  objects;
- add visual Q&A only after extracted text and visible evidence can be audited.

The full component choices, licenses, interfaces, and gates are in the
[multimodal expansion roadmap](project/multimodal-expansion-roadmap.md).

## Release Rule

Every new capability must include:

- a pinned model and data revision;
- source and redistribution records;
- a held-out evaluation set;
- machine metrics and native human review;
- latency and resource measurements;
- failure examples and an explicit promotion decision.

Features that work in a demo but do not meet these conditions remain labeled as
research prototypes.

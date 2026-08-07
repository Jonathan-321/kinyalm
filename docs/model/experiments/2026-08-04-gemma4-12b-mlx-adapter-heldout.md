# Gemma 4 12B: Held-Out MLX Base vs Adapter Check

**Date:** 2026-08-04

**Decision:** fail; do not use this adapter for the demo

**Native-speaker blind review:** pack ready; scoring pending

**Exact BF16 parity run:** optional follow-up

## Question

Does the rejected Gemma 4 12B adapter degrade the pinned local base when both
receive the same held-out prompts through the same MLX runtime?

This comparison holds the local model checkpoint, chat template, system prompt,
task order, seed, decoding, and token budget constant. The only experimental
difference is whether the converted adapter is attached.

## Immutable Inputs

| Input | Revision or setting |
| --- | --- |
| Source base | `google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Local MLX base | `mlx-community/gemma-4-12B-it-qat-4bit@e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` |
| Adapter | `kinyalm/kinyalm-gemma-4-12b-experimental@feefb1e7ac359b60ca45af9db8fd883af8cac933` |
| MLX conversion | 656 of 656 tensors converted; source SHA-256 `508ff76bcfa124c75678a0abc4887668ff1850d4b9e74a9a21642bb0582dca23` |
| Backend | MLX-LM 0.31.3 |
| Candidates | Unchanged local base and the same base with the converted adapter |
| Tasks | First 10 held-out tasks selected in task-bank order |
| Decoding | Greedy, thinking disabled, maximum 768 new tokens |
| Seed | 20260721 |

The run used the same system prompt and task selection implemented by
`scripts/run_multilingual_bakeoff.py`. The local adapter command is preserved
in `docs/model/gemma4-adapter-recovery-plan.md`.

## Runtime Result

| Metric | Unchanged base | Base with adapter |
| --- | ---: | ---: |
| Tasks completed | 10 / 10 | 10 / 10 |
| Output tokens | 3,648 | 1,795 |
| Generation time | 614.35 seconds | 338.63 seconds |
| Mean generation speed | 6.06 tokens/second | 5.68 tokens/second |
| Peak unified memory | 11.73 GB | 11.95 GB |
| Responses stopped normally | 10 | 8 |
| Responses truncated at 768 tokens | 0 | 2 |

The base was substantially more verbose, but verbosity is separate from task
correctness. It completed every response normally and preserved the meanings
of the elementary translation prompts that the adapter reversed.

## Failure Evidence

The adapter violated the recovery plan's acceptance criteria before subjective
scoring:

- `T008`, a two-example vocabulary request, repeated one bilingual sentence
  until the 768-token limit.
- `T010`, a school-vocabulary lesson, repeated `bagengeye` until the 768-token
  limit.
- `T016` translated `I am a student.` as `Ndi umwarimu.`, which means that the
  speaker is a teacher rather than a student.
- `T018` rendered `My name is Aline.` as `Ndi Aline.` instead of using the
  expected name construction.
- `T011` claimed there was an apostrophe to remove from `kwiga`, although the
  supplied sentence contained no apostrophe.
- `T005` did not produce the requested student-teacher exchange.

One response showed useful uncertainty behavior: `T015` requested the missing
learner sentence instead of inventing it. That isolated success does not offset
the repetition and elementary translation failures.

On the exact same prompts, the unchanged base produced the requested exchange
for `T005`, translated `I am a student.` as `Ndi umunyeshuri.` for `T016`, and
translated `My name is Aline.` as `Izina ryanjye ni Aline.` for `T018`. The
base still contains questionable Kinyarwanda and over-explanation, so this is
evidence of adapter degradation rather than evidence that the base is ready for
release.

## Interpretation

This is not a verbosity problem that should be repaired with a shorter output
limit. The loops and meaning reversals show that the adapter itself is unsafe
for the demo. A decoding cap would only hide the failure sooner.

The result is consistent with the earlier local chat and the direct PEFT
30-prompt A100 run. Because the local base does not reproduce the two loops and
gets the simple translations right, the shared MLX checkpoint is not the cause
of these regressions. Adding the adapter causes them.

An exact same-prompt BF16 comparison could still measure backend parity, but it
is not required to reject the old adapter or to start the corrected-objective
control.

## Local Artifacts

The raw files remain under `outputs/adapter-parity/` until they are published
to the evaluation area of the data lake. Their SHA-256 hashes are:

| Artifact | SHA-256 |
| --- | --- |
| Unchanged-base JSONL | `2e39fb1059a1df84a62ef38412055305bf2a330f6e97495ff7a3a714e97b5214` |
| Adapter JSONL | `1a8d489f843998a9e2ce250337c82b8d0d5a753eecbf1b81bec50e23fa85b47f` |
| 20-row blind-review CSV | `2aa9e918e5ccb7c601761745ff3c7d8564f9d7757690855d118747c60413415f` |

The review CSV hides model identity and randomizes answer order. The separate
key must remain private until native-speaker scoring is complete.

## Next Decision

1. Have fluent reviewers score the 20-row local blind pack.
2. Run the corrected assistant-only one-epoch control on the frozen split.
3. Evaluate the corrected control against the unchanged base with these same
   tasks before promoting any checkpoint.
4. Run exact BF16 parity only if the corrected PEFT and MLX outputs later
   disagree materially.

Do not add training steps to revision `feefb1e7ac35`. The next training artifact
must use the corrected assistant-completion objective and a new repository or
run identifier.

# Gemma 4 12B Experimental QLoRA Run

Run date: 2026-08-03

## Outcome

The complete two-epoch Gemma 4 QLoRA run finished, evaluated all validation
rows, generated samples, published its adapter, and terminated the Lambda
instance. The optimization result is valid and reproducible. The adapter is not
ready to present as a language-quality improvement: its first 12 saved samples
still contain serious Kinyarwanda tutoring failures, and no blinded comparison
against the unchanged base has been scored.

## Reproducible setup

| Field | Value |
| --- | --- |
| Run ID | `gemma4-12b-experimental-20260803T024131Z` |
| Project commit | `d1d7a48e118c5ae08dd6f5f962eee2704b3cc460` |
| Base model | `google/gemma-4-12B-it` |
| Base revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Dataset | `kinyalm/kinyalm-data-lake` |
| Dataset revision | `754a58b021cfe1e505f432df0de45ce2f63a3b21` |
| Data tier | `experimental-critic-filtered` |
| Human reviewed | No |
| Split | 776 train + 87 validation |
| GPU | Lambda 1x A100 40 GB SXM4 |
| Quantization | 4-bit NF4 with BF16 computation |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Schedule | 2 epochs, 194 optimizer steps, learning rate `2e-4`, warmup ratio 0.03 |
| Sequence/batch | 1,024 tokens, batch 1, gradient accumulation 8 |
| Seed | 42 |

## Optimization result

| Metric | Result |
| --- | ---: |
| Training runtime | 1,572 seconds (26.2 minutes) |
| Final training loss | 1.6264 |
| Final validation loss | 1.3250 |
| Final validation mean token accuracy | 0.6943 |
| Validation runtime | 6.0029 seconds |
| Recorded processed tokens | 306,840 |
| Adapter size | 131,235,784 bytes (about 126 MiB) |
| Generated samples | 12 |
| Training-only cost floor | About `$0.87` at `$1.99/hour` |

These metrics show that the adapter learned to predict tokens in this
critic-filtered split. They do not show that it became a more accurate or
natural Kinyarwanda tutor.

## Preliminary sample inspection

This is an engineering spot-check, not the planned blinded native-speaker
evaluation. The saved samples expose several high-severity problems:

| Prompt capability | Observed issue |
| --- | --- |
| Define `ubupfura` | Gives an incorrect, incoherent definition and examples |
| Correct `Ejo nzagiye ku ishuri` | Repeats the original sentence and calls it correct |
| Explain `kubona` versus `kureba` | Starts the distinction but ends with malformed explanations |
| Translate “introduce myself politely” | Translates it as learning how to repeat oneself |
| Beginner conversation | Produces a third-person description instead of starting a conversation |
| `mu`, `ku`, and `i` exercise | Falls into a long `yari` repetition loop |
| Noun-class agreement | Gives a questionable singular agreement form |
| Respectful “come here” | Supplies unnatural or incorrect phrases |
| Polysemy of `gusoma` | Misses the relevant second sense and invents an explanation |
| Ambiguous tense clarification | Produces a malformed question instead of clarifying past versus future |

The English translation of `Nubwo imvura yagwaga, twakomeje urugendo` is the
clearest usable response in the sample set. One good answer is not enough to
offset the systematic failures above.

## Evidence and publication

Private adapter repository:
<https://huggingface.co/kinyalm/kinyalm-gemma-4-12b-experimental>

Published revision: `feefb1e7ac359b60ca45af9db8fd883af8cac933`

The local evidence bundle is stored under:
`docs/model/experiments/runs/gemma4-12b-experimental-20260803T024131Z/`

It contains the dataset manifest, preflight configuration, metadata, training
log, trainer state, adapter configuration, environment record, and all 12
samples. The 126 MiB adapter weight file remains on Hugging Face rather than in
Git.

The Lambda console showed **No running instances** at 2026-08-03 03:22 UTC.

## Decision

Keep this adapter as the controlled experimental checkpoint. Do not call it the
final KinyaLM model. The next quality gate is:

1. generate identical held-out prompts with the unchanged base and this adapter;
2. randomize and blind the answer labels;
3. collect two fluent-speaker scores per answer;
4. report wins, ties, losses, regressions, and reviewer agreement;
5. use those failures to revise the human-approved data;
6. retrain the final adapter only after freezing that approved split.

# Gemma 4 12B Experimental QLoRA Run

Run date: 2026-08-03

## Outcome

The complete two-epoch Gemma 4 QLoRA run finished, evaluated all validation
rows, generated samples, published its adapter, and terminated the Lambda
instance. The optimization result is reproducible, but the checkpoint is now
rejected for demonstration. Its first 12 CUDA samples contained serious
Kinyarwanda tutoring failures, and a later native-speaker local session exposed
severe response collapse across unrelated prompts.

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

## Live native-speaker rejection

The locally converted adapter was tested through the browser interface. It
answered `Uri nde?` with a malformed identity response and then converged on
the same sentence for unrelated conversation, identity, and factual prompts:

> `Ndakora imyitozo y'ubyandikwa.`

The sentence appears zero times in the frozen training split, system prompt,
and project code. This is therefore not direct memorization of one row. It is a
generation collapse severe enough to reject the checkpoint without waiting for
a full blind study. The session is useful failure evidence, but it is not a
quantitative base-versus-adapter score.

## Root-cause audit

- The original PEFT checkpoint already produced a repetition loop on Lambda,
  so MLX is not the primary source of the failure.
- All 656 PEFT tensors were preserved exactly by the MLX conversion after the
  required transpose; maximum absolute tensor difference was `0.0`.
- The original trainer passed a conversational `messages` dataset to TRL
  without assistant-only masking. In the pinned TRL version, this computes loss
  over the full chat, including user prompts.
- The run used two epochs at `2e-4` while adapting all attention and MLP
  projections on data that the manifest explicitly marks as not human-reviewed
  and not production-eligible.
- The local MLX checkpoint uses a different quantization scheme from the CUDA
  NF4 training base. Output parity still needs a controlled A100-versus-MLX
  check, but that residual risk does not explain the failures already saved on
  Lambda.

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

Keep this adapter only as a rejected, controlled experimental artifact. Do not
add steps, present it as KinyaLM, or use it as the starting point for preference
optimization. The recovery plan is:

1. verify direct PEFT versus MLX output parity on identical prompts;
2. rerun one controlled experiment with assistant-only loss, one epoch, and
   learning rate `5e-5`;
3. freeze and train a fluent-human-approved split;
4. randomize and blind all base-versus-adapter labels;
5. report wins, ties, losses, regressions, and reviewer agreement by task
   family.

The detailed sequence is in
[`gemma4-adapter-recovery-plan.md`](../gemma4-adapter-recovery-plan.md).

# KinyaLM Continuation Screening: Team Report

**Date:** August 11, 2026

**Status:** complete model-assisted development screen; native review pending

## Result in One Paragraph

The targeted continuation is the strongest KinyaLM adapter we have trained so
far. On the same 150 prompts, it passed 28 model-assisted reviews, compared with
24 for both the source step-780 adapter and the control continuation. It also
beat the unchanged Gemma base in the areas we explicitly targeted: morphology
and grammar, sentence correction, and Kinyarwanda-to-English translation.
However, it did not beat Gemma overall: the base passed 33 prompts, five more
than the targeted adapter, and repeated much less often. The next goal is
therefore to preserve the targeted gains without damaging conversation,
uncertainty handling, greetings, or context retention.

## Comparable Results

All four systems answered the same 150 prompts with the same system message,
greedy decoding, 256-token limit, base checkpoint, and MLX runtime. Model names
were hidden from the judge.

| Candidate | Passes | Pass rate | Kinyarwanda correctness | Repetition |
| --- | ---: | ---: | ---: | ---: |
| Unchanged Gemma 4 12B | **33/150** | **22.00%** | **2.53/5** | **4.67%** |
| Targeted continuation | 28/150 | 18.67% | 2.15/5 | 18.67% |
| Control continuation | 24/150 | 16.00% | 2.02/5 | 20.67% |
| Source step-780 adapter | 24/150 | 16.00% | 2.03/5 | 20.67% |

The targeted continuation improved on the source and control adapters by four
passes, or 2.67 percentage points. Against the base, it recovered 16 prompts
that Gemma failed, but regressed on 21 prompts that Gemma passed. Its net result
was therefore five fewer passes. The paired 95% confidence interval was
`-11.33` to `+4.67` percentage points, so this screen does not establish a
statistically reliable overall difference.

## Where Targeting Helped

| Capability | Base | Targeted | Change |
| --- | ---: | ---: | ---: |
| Morphology and grammar | 10% | **20%** | **+10 pp** |
| Sentence correction | 10% | **25%** | **+15 pp** |
| Kinyarwanda to English | 40% | **60%** | **+20 pp** |

These are real development signals: the continuation data changed the behavior
in the intended direction, and the targeted curriculum outperformed training
longer on the original mixture. They are not yet final product claims because
each category is small and the current scores come from a model judge.

## Where It Still Regressed

The targeted adapter fell behind the base on ambiguity and uncertainty
(`20%` versus `80%`), greetings (`10%` versus `40%`), reading and multi-turn
consistency (`0%` versus `20%`), dialogue (`0%` versus `10%`), and English-to-
Kinyarwanda translation (`10%` versus `20%`). It also repeated on 28 of 150
prompts, compared with seven for the base. These failures explain why lower
validation loss and visible task-specific gains did not become an overall win.

The first live comparison after this screen confirmed the risk. For the prompt
`Kosora iyi nteruro kandi usobanure mu Cyongereza: Nmeze neza, wowe umeze
gute?`, the base corrected `Nmeze` to `Nimeze` and explained the change. The
targeted adapter exhausted its 192-token budget repeating `yose`. This is one
smoke result rather than a score, but it is a clear release blocker and a useful
native-review example for the next correction set.

## Reproducibility

- Base: `mlx-community/gemma-4-12B-it-qat-4bit@e70c6b3ba097`
- Targeted adapter: [kinyalm/kinyalm-gemma-4-12b-continuation-targeted-lr2e6](https://huggingface.co/kinyalm/kinyalm-gemma-4-12b-continuation-targeted-lr2e6)
- Adapter revision: `274504f7bcd029230ea6dfe04bc251964642bbda`
- Converted adapter SHA-256: `30fd3c2796843e9f38c42614a3401d66a4d904f1ab770fbf8454a245c913e544`
- Judge: Gemini 3.1 Pro, 600 of 600 blinded rows completed
- Evaluation role: development screening, not the final hidden test

The local chat now exposes `Targeted`, `Base`, and `Compare` modes. `Compare`
runs both versions sequentially from one resident MLX model, with the same
prompt, conversation settings, and decoding. The unchanged base is obtained by
setting the loaded LoRA contribution to zero; a recorded benchmark response was
reproduced exactly before this path was enabled.

## Team Decision

Keep the targeted continuation as the best adapter candidate, but do not call it
better than Gemma overall. Use the 10-prompt local comparison for native review,
record exact corrections, and build the next targeted mixture around the 21
regressions while retaining the 16 recovered behaviors. Promotion still needs a
new hidden native-reviewed evaluation set that was never used to shape training.

# Open Kinyarwanda Benchmark Candidates

This project needs two evaluation layers:

1. project-specific tutor prompts in `docs/evaluation/learning-task-bank.md`,
2. external open benchmarks that let us compare against known tasks.

Do not train on benchmark test sets. Treat these as evaluation-only unless the
source license and split design explicitly allow training.

## Recommended First Benchmarks

| Benchmark | Task | Kinyarwanda Coverage | License / Access | Project Use | Priority |
| --- | --- | --- | --- | --- | --- |
| Belebele | multiple-choice reading comprehension | `kin_Latn`, 900 rows | CC-BY-SA-4.0 on Hugging Face | Evaluate reading comprehension and answer selection. | high |
| FLORES-200 | machine translation | `kin_Latn` | CC-BY-SA-4.0 | Evaluate English <-> Kinyarwanda translation with chrF/spBLEU. | high |
| AfriXNLI | natural language inference | `kin`, 450 validation + 600 test rows | Apache-2.0 on Hugging Face | Test reasoning over Kinyarwanda sentence pairs. | high |
| AfriMGSM | grade-school math reasoning | `kin`, 8 train + 250 test rows | Apache-2.0 on Hugging Face | Test whether the model can follow Kinyarwanda math prompts. | medium |
| AfriMMLU | multiple-choice knowledge QA | `kin`, 608 rows | Apache-2.0 on Hugging Face | Test Kinyarwanda multiple-choice QA across five subjects. | medium |
| AfriSenti-SemEval | sentiment classification | `kin` included among 14 languages | CC-BY-4.0 repo license | Test sentiment classification, not tutor ability. | medium |
| KINNEWS | news topic classification | 21,268 Kinyarwanda news articles, 14 classes | MIT repo license | Test text classification and domain robustness. | medium |

## Why These Matter

### Belebele

Belebele is useful because it is not just translation. It tests whether a model
can read a passage and choose the right answer from four options. For a tutor,
this is closer to comprehension than raw next-token loss.

Use:

- multiple-choice accuracy,
- refusal/format compliance,
- answer explanation quality if we ask the model to explain after choosing.

### FLORES-200

FLORES-200 is useful for controlled translation evaluation. It includes
Kinyarwanda as `kin_Latn`, so we can evaluate both directions:

- English to Kinyarwanda,
- Kinyarwanda to English.

Use:

- chrF as the first metric,
- spBLEU only if the tokenizer setup is correct,
- fluent-speaker review for a small sample because automatic MT metrics can
  miss learner-facing quality problems.

### IrokoBench: AfriXNLI, AfriMGSM, AfriMMLU

IrokoBench is useful because it moves beyond shallow text classification. Its
Kinyarwanda subsets cover:

- inference,
- math reasoning,
- multiple-choice knowledge QA.

These should be held out from SFT training and used as model capability checks.

### AfriSenti-SemEval

AfriSenti gives a Kinyarwanda sentiment task. It is not directly a tutor task,
but it can reveal whether the model can classify short, social-media-style
Kinyarwanda text.

Use carefully because the source domain is Twitter/X-style text, which is not
the same as classroom language.

### KINNEWS

KINNEWS is a Kinyarwanda news classification benchmark with published
baselines. It is not a tutor benchmark, but it is useful for checking whether
representations understand news-domain Kinyarwanda enough to classify topics.

Use:

- accuracy,
- macro-F1 if class balance is uneven,
- no direct tutor fine-tuning unless the team separately approves the source
  and domain fit.

## Not A Kinyarwanda Benchmark

MasakhaNEWS is valuable for African news classification, but the Hugging Face
subset list includes `run` for Rundi rather than `kin` for Kinyarwanda. Do not
count it as a Kinyarwanda benchmark unless a Kinyarwanda subset is added later.

## First Implementation Plan

1. Add source-log entries for the benchmark sources.
2. Keep the structured benchmark manifest in
   `configs/evaluation/kinyarwanda_benchmarks.json` valid.
3. Start with Belebele and AfriXNLI because they are easy to score with
   accuracy.
4. Add FLORES after the translation evaluator has chrF working.
5. Keep all benchmark examples out of `data/sft/seed_conversations.jsonl`.

## Source Links

- Belebele: https://github.com/facebookresearch/belebele
- Belebele on Hugging Face: https://huggingface.co/datasets/facebook/belebele
- FLORES-200: https://github.com/facebookresearch/flores/tree/main/flores200
- FLORES on Hugging Face: https://huggingface.co/datasets/facebook/flores
- IrokoBench collection: https://huggingface.co/collections/masakhane/irokobench
- AfriXNLI: https://huggingface.co/datasets/masakhane/afrixnli
- AfriMGSM: https://huggingface.co/datasets/masakhane/afrimgsm
- AfriMMLU: https://huggingface.co/datasets/masakhane/afrimmlu
- AfriSenti-SemEval: https://github.com/afrisenti-semeval/afrisent-semeval-2023
- KINNEWS: https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus

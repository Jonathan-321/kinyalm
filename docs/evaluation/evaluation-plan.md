# Evaluation Plan

## Tokenizer Evaluation

Metrics:

- average tokens per word,
- tokens per character,
- fragmentation of common prefixes and noun classes,
- treatment of apostrophes, hyphens, punctuation, and accents,
- examples of good and bad segmentations.

Starter examples:

- greetings,
- family vocabulary,
- school phrases,
- common verbs,
- noun-class examples,
- short learner sentences.

## LM Evaluation

Metrics:

- validation loss,
- perplexity,
- sample quality,
- repetition rate,
- memorization checks,
- comparison across training sizes.

Interpretation rule:

```text
Loss going down means the training loop works. It does not mean the model is a
good tutor.
```

## Tutor Evaluation

Human-scored rubric for each answer:

| Dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Kinyarwanda correctness | wrong or misleading | partly correct | correct and natural |
| Beginner clarity | confusing | understandable | clear and learner-friendly |
| Grammar explanation quality | wrong rule or no rule | partly useful | accurate and well scoped |
| Cultural/register appropriateness | awkward or risky | acceptable | context-aware |
| Helpfulness | does not answer | partially helps | directly useful |
| Uncertainty behavior | overconfident | mixed | clearly states uncertainty |

Reviewer fields:

- prompt ID,
- model or system version,
- answer text,
- Kinyarwanda correctness score,
- beginner clarity score,
- grammar explanation score,
- cultural/register note,
- hallucination or unsupported-claim flag,
- final pass/fail decision,
- reviewer notes.

Review rules:

1. Score what the answer actually says, not what the model probably meant.
2. Mark `not sure` rather than guessing when a grammar or register question is
   uncertain.
3. Fail answers that give confident cultural claims without support.
4. Fail answers that are mostly English when the prompt asks for Kinyarwanda
   output, unless the prompt explicitly asks for explanation in English.
5. Keep benchmark prompts separate from SFT training rows.

## Starter Tutor Tasks

Use the 50-prompt task bank:

```text
docs/evaluation/learning-task-bank.md
```

## Error Categories

- wrong grammar rule,
- wrong translation,
- hallucinated cultural claim,
- too much English,
- too much Kinyarwanda for beginner level,
- overconfident uncertainty,
- unsupported answer.

## First Demo Evaluation

For the first SFT-readiness demo:

1. Pick 10 `benchmark-only` prompts from at least five categories.
2. Run the same prompts against the baseline system and the SFT candidate, if
   an SFT candidate exists.
3. Have a fluent speaker review answers without looking at training examples.
4. Report average scores and the three most important failure modes.
5. Do not claim tutor quality from loss or perplexity alone.

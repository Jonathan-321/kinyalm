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

Human-scored rubric:

| Dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Correctness | wrong or misleading | partly correct | correct |
| Clarity | confusing | understandable | clear and learner-friendly |
| Cultural appropriateness | awkward or risky | acceptable | context-aware |
| Helpfulness | does not answer | partially helps | directly useful |
| Uncertainty behavior | overconfident | mixed | clearly states uncertainty |

## Starter Tutor Tasks

- Explain a basic greeting.
- Correct a simple learner sentence.
- Generate a five-question vocabulary quiz.
- Explain a noun-class example.
- Translate a short classroom phrase.
- Give a polite version of a phrase.
- Explain when a phrase should not be used.

## Error Categories

- wrong grammar rule,
- wrong translation,
- hallucinated cultural claim,
- too much English,
- too much Kinyarwanda for beginner level,
- overconfident uncertainty,
- unsupported answer.

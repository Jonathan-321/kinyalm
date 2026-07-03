# Tokenizer Analysis Plan

## Goal

Evaluate whether a tokenizer handles Kinyarwanda text in a way that is compact,
consistent, and explainable.

This does not prove the language model is good. It only tells us whether the
first text-to-token layer is reasonable enough to train on.

## Starter Examples

The starter examples live in:

```text
docs/tokenizer/eval-examples.tsv
```

Every example is marked `needs-review` until a fluent speaker or teacher checks
it. Do not use these examples as authoritative lesson content yet.

## Metrics

Track:

- words,
- tokens,
- tokens per word,
- characters,
- tokens per character,
- longest tokenized example,
- examples with heavy fragmentation.

## Review Questions

For each tokenizer:

1. Does it split common prefixes too aggressively?
2. Does it handle punctuation without creating strange fragments?
3. Does it keep common words reasonably compact?
4. Does it behave differently on names, classroom phrases, and morphology-heavy
   words?
5. Are the results explainable in the final report?

## First Comparison

Compare at least two tokenizers:

1. the team-built BPE tokenizer,
2. an existing multilingual tokenizer, such as one from a base model used for
   the tutor prototype.

The team-built tokenizer can be small. The point is to analyze behavior, not to
claim production quality.

## Current Scaffold

Implemented project helpers:

- `src/kinyalm/data/examples.py`
- `src/kinyalm/tokenization/metrics.py`
- `tests/test_tokenizer_metrics.py`
- `scripts/check_project.py`

These helpers evaluate tokenizer outputs. They do not implement BPE.

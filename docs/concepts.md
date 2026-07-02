# Concept Map

## How The Project Teaches The Class Ideas

```text
Kinyarwanda text
→ cleaning and licensing decisions
→ tokenizer training
→ token IDs
→ Transformer LM
→ logits
→ cross-entropy loss
→ optimizer step
→ checkpointed training
→ validation loss/perplexity
→ sampling
→ tutor prototype and human evaluation
```

## Tokenizer

The tokenizer turns raw text into token IDs. For Kinyarwanda, this is not just a
plumbing detail because morphology matters: prefixes, noun classes, agreement,
apostrophes, and compound forms can be fragmented badly by an English-heavy
tokenizer.

First analysis questions:

- How many tokens per word do we get?
- Which common prefixes or suffixes are split awkwardly?
- Do common classroom phrases stay reasonably compact?
- How are punctuation, apostrophes, hyphens, and accents handled?

## Language Model

The language model predicts the next token. A small from-scratch model will not
be a strong tutor by itself, but it proves the team understands the training
pipeline.

Core pieces:

- embeddings,
- attention,
- feed-forward layers,
- normalization,
- residual connections,
- logits,
- cross-entropy,
- optimizer,
- checkpointing.

## Tutor Prototype

The tutor should be useful even if the from-scratch LM is small. The first MVP
should use retrieval over approved lesson notes and examples, because that gives
more control over correctness than asking a tiny model to invent grammar
explanations.

Tutor tasks:

- explain grammar,
- teach vocabulary by theme,
- generate practice exercises,
- gently correct learner sentences,
- show cultural/register context,
- say when it is uncertain.

## Evaluation

This project needs three evaluation tracks:

- tokenizer evaluation,
- LM evaluation,
- tutor evaluation.

The tutor should be scored by humans on correctness, clarity, cultural
appropriateness, helpfulness, and uncertainty behavior.

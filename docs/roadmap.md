# Roadmap

## Project Window

June 30 - August 14, 2026

## North Star

Build a Kinyarwanda language learning project that is honest in scope:

```text
from-scratch LM components for learning + tutor prototype for usefulness
```

The from-scratch model may be small and limited. That is acceptable. The
deliverable should still show that the team understands the tokenizer, model,
training loop, data limits, evaluation, and risks.

## Week 1: Repo, Proposal, Resources, Data Inventory

Focus:

- set up the repo,
- assign owners,
- confirm project scope with the professor,
- inventory possible data sources,
- identify constraints early.

Deliverables:

- project charter,
- team roles,
- weekly operating rhythm,
- data source log,
- compute plan,
- risk register,
- initial reading schedule.

Decision point:

```text
Professor feedback on scope, licensing, evaluation, and compute expectations.
```

## Week 2: Tokenizer And Data Cleaning

Focus:

- collect a small permissioned Kinyarwanda corpus,
- clean and document it,
- implement or adapt the project BPE tokenizer path,
- analyze tokenization quality for Kinyarwanda morphology.

Deliverables:

- cleaned sample corpus,
- BPE tokenizer train/encode/decode/save/load,
- tokenizer evaluation report,
- examples of good and bad segmentations.

Decision point:

```text
Choose vocabulary size, cleaning rules, and minimum viable corpus.
```

## Week 3: Transformer LM Sanity Check

Focus:

- get a small decoder-only Transformer training run working on a tiny corpus,
- verify loss decreases,
- save and resume checkpoints.

Deliverables:

- tiny training config,
- loss curve,
- checkpoint/resume proof,
- sampling examples,
- notes on model size and compute limits.

Decision point:

```text
Confirm the model size that fits the available compute.
```

## Week 4: Kinyarwanda LM Baseline

Focus:

- train the tokenizer and small LM on the cleaned Kinyarwanda corpus,
- evaluate honestly.

Deliverables:

- validation loss/perplexity,
- sample generations,
- repetition and memorization checks,
- error analysis reviewed by fluent/native speakers if available.

Decision point:

```text
Prioritize better data, model scaling, or tutor prototype reliability.
```

## Week 5: Tutor MVP

Focus:

- build a retrieval-first tutor using approved grammar notes and examples,
- create 20-30 curated teaching tasks,
- run first human evaluation.

Deliverables:

- simple tutor interface,
- curated knowledge base,
- evaluation prompts,
- first pass/fail review.

Decision point:

```text
Pick final demo path: RAG-only, fine-tuned, or hybrid.
```

## Week 6: Evaluation, Polish, Documentation

Focus:

- finish the reproducibility path,
- write data card and model card,
- freeze features,
- fix the highest-impact demo failures.

Deliverables:

- data card,
- model card,
- evaluation results,
- limitations and safety notes,
- cleaned scripts and notebooks.

Decision point:

```text
Freeze features by August 3 and prioritize reliability.
```

## Final Sprint: August 10 - August 14

Focus:

- final presentation,
- demo,
- report,
- reproducibility check.

Deliverables:

- slides,
- final demo or recorded fallback,
- final report,
- links to shareable artifacts.

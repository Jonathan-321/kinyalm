# Team Execution Plan

## Purpose

This repository is for team-facing project work.

Use it for:

- shared planning,
- source review,
- tokenizer examples and analysis,
- approved-corpus preparation,
- evaluation rubrics,
- project reports,
- reproducible scripts the whole team can run.

Keep separate:

- private scratch experiments,
- toy-only training loops,
- unreviewed data downloads,
- large outputs,
- one-person notebooks that are not ready for review.

## Current State

Done:

- public GitHub repo,
- collaborators added,
- owners assigned,
- Week 1 issues created,
- Hugging Face candidate datasets listed,
- first-pass source metadata review generated,
- tokenizer evaluation examples added,
- tokenizer metrics scaffold added,
- basic tests passing.

Current checks:

```bash
python3 scripts/check_project.py
PYTHONPATH=src python3 -m pytest tests/test_tokenizer_metrics.py -q
python3 scripts/review_hf_sources.py --out docs/data/huggingface-source-review.md
```

## Main Rule

Do not train on any dataset until it has a clear source-log status.

Allowed now:

- inspect dataset cards and metadata,
- record licenses and provenance,
- validate sample words/sentences with speakers,
- build tokenizer-analysis scripts,
- write evaluation prompts,
- define model configs.

Not allowed yet:

- train on blocked sources,
- commit raw data,
- scrape lesson pages without permission,
- fine-tune a tutor model,
- claim model quality from toy outputs or training loss alone.

## Immediate Workstreams

### Data: Tessy

Goal:

```text
Turn source candidates into approved, reference-only, blocked, or rejected.
```

Inputs:

- `docs/data/source-log.md`
- `docs/data/huggingface-source-review.md`
- `docs/data/source-review-checklist.md`

Next actions:

1. Confirm visible licenses from dataset cards.
2. Check whether each source allows training, redistribution, and quoted
   examples in the final report.
3. Add non-Hugging Face sources if useful.
4. Mark blocked sources clearly.
5. Mirror spreadsheet decisions back into the repo.

First useful outcome:

```text
One small source or source subset is approved for tokenizer experiments.
```

### Tokenizer: Bonheur

Goal:

```text
Make tokenizer behavior visible before model training.
```

Inputs:

- `docs/tokenizer/eval-examples.tsv`
- `docs/tokenizer/tokenizer-analysis-plan.md`
- `src/kinyalm/tokenization/metrics.py`
- CS336 Assignment 1 tokenizer requirements

Next actions:

1. Review and correct the 20 starter examples.
2. Add morphology-heavy examples if needed.
3. Confirm which examples need speaker validation.
4. Build a small tokenizer-analysis report or notebook.
5. Compare team BPE behavior against a multilingual tokenizer once BPE exists.

First useful outcome:

```text
A tokenizer-analysis report showing tokens per word, tokens per character,
and fragmentation examples.
```

### Modeling: Bonheur

Goal:

```text
Define the smallest realistic LM run before writing a large training path.
```

Inputs:

- `docs/model/compute-inventory.md`

Next actions:

1. Collect compute inventory.
2. Propose tiny CPU and small-GPU model configs.
3. Define validation split expectations.
4. Decide what counts as a successful sanity run.

First useful outcome:

```text
A tiny model config that the team agrees can run on available hardware.
```

### Evaluation: Tessy

Goal:

```text
Make success measurable before the model exists.
```

Inputs:

- `docs/evaluation/evaluation-plan.md`
- `docs/evaluation/learning-task-bank.md`

Next actions:

1. Tighten the human scoring rubric.
2. Review the 20 starter learning tasks.
3. Define failure categories for grammar, translation, and uncertainty.
4. Decide which prompts need fluent-speaker review.

First useful outcome:

```text
A small evaluation set the team can use for Track A samples or any tutor
fallback.
```

### Coordination And Docs: Jonathan

Goal:

```text
Keep the project moving without scope creep.
```

Next actions:

1. Keep GitHub issues current.
2. Record decisions in the repo.
3. Keep professor feedback visible once received.
4. Make sure blocked data stays blocked.
5. Keep benchmark sources separate from SFT training data.

First useful outcome:

```text
A weekly update that says what is approved, what is blocked, and what the team
can implement next.
```

## Decision Gates

### Gate 1: Data Gate

Pass condition:

```text
At least one small corpus is approved for tokenizer experiments.
```

If this fails, use only speaker-written examples and reference-only notes until
permission is clear.

Current status: Digital Umuganda TTS sentence text is approved for tokenizer
analysis and small-LM experiments. Digital Umuganda MT is approved for
Kinyarwanda-side import with attribution. After fixing cp1252 decoding, the MT
source can be prepared as 44,527 clean lines with zero replacement characters.

### Gate 2: Tokenizer Gate

Pass condition:

```text
BPE tokenizer round-trips text and has a clear analysis report.
```

Current status: the repo has 38 tokenizer evaluation examples, including
noun-class, verb morphology, apostrophe, and elision cases. Bonheur should
review the example set, flag bad or unnatural examples, and summarize the
tokenizer risks before further scale-up.

If this fails, fix tokenizer behavior before model training.

### Gate 3: Tiny LM Gate

Pass condition:

```text
Tiny model trains reproducibly with decreasing validation loss.
```

If this fails, debug data, batching, tokenizer, or model config before scaling.

Current status: the project has SFT schema, validation code, OSCER smoke-test
scripts, and a first QLoRA run plan. The missing inputs are reviewed SFT
examples, final base-model choice, and working GPU access. Early feasibility
work showed that decreasing validation loss is not enough; tutor usefulness
still needs held-out prompts and fluent-speaker review.

### Gate 4: Usefulness Gate

Pass condition:

```text
Outputs are good enough under human review to support learning tasks.
```

If this fails, keep Track B as a retrieval-first tutor fallback.

## What We Should Do Next

The next team-facing move is:

```text
Build the reviewed SFT seed pack and keep external benchmarks held out.
```

Concretely:

1. Jonathan drafts 100 schema-valid SFT rows marked `needs-review`.
2. Tessy reviews source status and language quality before rows enter training.
3. Bonheur finalizes the base-model recommendation and QLoRA constraints.
4. Jonathan keeps the 1,000-example target and benchmark separation visible in
   the task board.

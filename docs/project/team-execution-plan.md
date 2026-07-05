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

The runnable sandbox now lives outside this repo:

```text
https://github.com/Jonathan-321/kilm
```

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
- claim model quality from toy/sandbox outputs.

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
5. Keep KILM sandbox updates separate from this repo unless there is a
   team-ready decision to record.

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

### Gate 2: Tokenizer Gate

Pass condition:

```text
BPE tokenizer round-trips text and has a clear analysis report.
```

Current status: KILM has a toy BPE path that round-trips text, writes tokenizer
analysis reports, and runs through the tiny LM loop. This gate is not complete
for the class project until the same analysis is rerun on approved Kinyarwanda
text.

If this fails, fix tokenizer behavior before model training.

### Gate 3: Tiny LM Gate

Pass condition:

```text
Tiny model trains reproducibly with decreasing validation loss.
```

If this fails, debug data, batching, tokenizer, or model config before scaling.

Current status: KILM can prepare a corpus, run explicit train/validation
training, save/resume checkpoints, sample from checkpoints, and write run
reports. This gate is still toy-only until an approved corpus is available.

### Gate 4: Usefulness Gate

Pass condition:

```text
Outputs are good enough under human review to support learning tasks.
```

If this fails, keep Track B as a retrieval-first tutor fallback.

## What We Should Do Next

The next team-facing move is:

```text
Data approval + tokenizer analysis.
```

Concretely:

1. Tessy confirms which source can be used first.
2. Bonheur validates tokenizer examples.
3. Jonathan keeps the issue board and docs synchronized.
4. Once a source is approved, create a tiny processed sample and run the
   tokenizer-analysis scaffold on it.

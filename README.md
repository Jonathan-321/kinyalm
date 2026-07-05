# Kinyarwanda Language Learning LM

This repo is for a class project: build the core language-modeling pipeline from
scratch while also delivering a small, useful Kinyarwanda learning tutor demo.

The working plan is:

```text
Track A: from-scratch LM learning pipeline
Track B: practical Kinyarwanda tutor MVP
```

Track A follows the spirit of Stanford CS336: tokenizer, decoder-only
Transformer, loss, optimizer, training loop, checkpointing, sampling, and
evaluation. Track B uses retrieval over approved Kinyarwanda learning material
first, with fine-tuning only if time, data, and compute make that realistic.

## Start Here

Week 1 is not about training a model yet. It is about making the project
workable.

1. Read [project-charter.md](docs/project/project-charter.md).
2. Read the full [master-plan.md](docs/project/master-plan.md).
3. Follow the [team-execution-plan.md](docs/project/team-execution-plan.md).
4. Check team ownership in [roles.md](docs/team/roles.md).
5. Work through [week-1-kickoff.md](docs/team/week-1-kickoff.md).
6. Record every data source in [source-log.md](docs/data/source-log.md).
7. Keep risks visible in [constraints-and-risks.md](docs/project/constraints-and-risks.md).

## Contributors

- Jonathan Muhire ([@Jonathan-321](https://github.com/Jonathan-321))
- Tessy Mugisha ([@TessyMugisha](https://github.com/TessyMugisha))
- Bonheur Byiringiro ([@BonheurByiringiro](https://github.com/BonheurByiringiro))

## Repository Layout

```text
configs/                  # training/eval config files
coursework/cs336/         # official CS336 assignment repos as submodules
data/
  raw/                    # original downloaded or received data, not committed
  interim/                # intermediate cleaned data, not committed by default
  processed/              # final small shareable samples and manifests
  external/               # third-party references, tracked by source log
docs/
  data/                   # source log, data card template
  evaluation/             # evaluation rubric and test prompt plan
  project/                # charter, roadmap, constraints
  team/                   # roles, weekly operating plan
experiments/              # local experiment outputs
logs/                     # local training/eval logs
notebooks/                # analysis notebooks
scripts/                  # project helper scripts
src/kinyalm/              # project package code
tests/                    # project tests
```

Large datasets, checkpoints, and logs should not be committed unless we
explicitly decide they are small, licensed, and useful to share.

## First Technical Milestone

The first technical milestone is:

```text
Train and evaluate a Kinyarwanda-aware BPE tokenizer on a small, documented
sample corpus.
```

That includes:

- a source log for the corpus,
- a Hugging Face source review,
- a cleaning note,
- tokenizer train/encode/decode/save/load behavior,
- tokens-per-word and fragmentation analysis,
- examples involving prefixes, noun classes, apostrophes, hyphens, and common
  Kinyarwanda words.

## First Runnable Check

Run the project health check:

```bash
python3 scripts/check_project.py
```

Run the starter tokenizer metric tests:

```bash
PYTHONPATH=src python3 -m pytest tests/test_tokenizer_metrics.py -q
```

These checks do not train a tokenizer or use unapproved data. They only verify
that the starter example set and tokenizer-analysis helpers are wired correctly.

Review Hugging Face candidate metadata:

```bash
python3 scripts/review_hf_sources.py --out docs/data/huggingface-source-review.md
```

## Team Execution

The team-facing next steps live here:

```text
docs/project/team-execution-plan.md
```

The immediate focus is:

```text
data approval + tokenizer analysis
```

## KILM Sandbox

The tiny end-to-end LM sandbox now lives in its own repo:

https://github.com/Jonathan-321/kilm

That repo contains approved-corpus fetching, corpus preparation, character and
BPE tokenizer paths, morphology-focused tokenizer examples, explicit
train/validation runs, learning-rate scheduling, gradient clipping, interval
checkpoints, sampling, run reports, review packets, CI, and tests. Its first
approved-data smoke baseline uses the Digital Umuganda TTS sentence text:
1,000 prepared lines split into 900 train / 100 validation lines.

This repo keeps only the planning record and Track A gates so the sandbox can
evolve without turning `kinyalm` into an experiment dump.

## CS336 Boundary

The CS336 assignment repos include their own AI-assistance policy. We will use
them for learning and reference, but we will not turn this repo into a solution
dump for course assignments.

Useful links:

- CS336 course page: https://cs336.stanford.edu/
- Assignment 1: https://github.com/stanford-cs336/assignment1-basics
- KinyaBERT paper: https://arxiv.org/abs/2203.08459

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

1. Read [project-charter.md](/Users/jonathanmuhire/Documents/RL/docs/project/project-charter.md).
2. Assign owners using [roles.md](/Users/jonathanmuhire/Documents/RL/docs/team/roles.md).
3. Work through [week-1-kickoff.md](/Users/jonathanmuhire/Documents/RL/docs/team/week-1-kickoff.md).
4. Record every data source in [source-log.md](/Users/jonathanmuhire/Documents/RL/docs/data/source-log.md).
5. Keep risks visible in [constraints-and-risks.md](/Users/jonathanmuhire/Documents/RL/docs/project/constraints-and-risks.md).

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
- a cleaning note,
- tokenizer train/encode/decode/save/load behavior,
- tokens-per-word and fragmentation analysis,
- examples involving prefixes, noun classes, apostrophes, hyphens, and common
  Kinyarwanda words.

## CS336 Boundary

The CS336 assignment repos include their own AI-assistance policy. We will use
them for learning and reference, but we will not turn this repo into a solution
dump for course assignments.

Useful links:

- CS336 course page: https://cs336.stanford.edu/
- Assignment 1: https://github.com/stanford-cs336/assignment1-basics
- KinyaBERT paper: https://arxiv.org/abs/2203.08459

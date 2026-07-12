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
first, with supervised fine-tuning only after source approval, seed examples,
and evaluation prompts are ready.

## Start Here

The current project push is SFT readiness: define the conversation format,
prepare held-out evaluation prompts, confirm source permissions, and prove the
compute path before training.

1. Read [project-charter.md](docs/project/project-charter.md).
2. Read the full [master-plan.md](docs/project/master-plan.md).
3. Follow the [team-execution-plan.md](docs/project/team-execution-plan.md).
4. Use the [two-week SFT execution plan](docs/team/two-week-sft-execution-plan.md).
5. Check team ownership in [roles.md](docs/team/roles.md).
6. Record every data source in [source-log.md](docs/data/source-log.md).
7. Follow the SFT schema in [sft-data-schema.md](docs/data/sft-data-schema.md).
8. Keep benchmark sources separate with
   [open-kinyarwanda-benchmarks.md](docs/evaluation/open-kinyarwanda-benchmarks.md).
9. Use [data-overhaul-plan.md](docs/data/data-overhaul-plan.md) to scale from
   the seed dataset toward a useful SFT pack.
10. Keep risks visible in [constraints-and-risks.md](docs/project/constraints-and-risks.md).

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
  sft/                    # small reviewed SFT JSONL files only
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
that the starter example set, tutor benchmark, SFT schema, and tokenizer-analysis
helpers are wired correctly.

Validate a future SFT seed file:

```bash
python3 scripts/validate_sft_jsonl.py data/sft/seed_conversations.jsonl
```

Review Hugging Face candidate metadata:

```bash
python3 scripts/review_hf_sources.py --out docs/data/huggingface-source-review.md
```

External benchmark metadata lives in:

```text
configs/evaluation/kinyarwanda_benchmarks.json
```

## Team Execution

The team-facing next steps live here:

```text
docs/project/team-execution-plan.md
```

The immediate focus is:

```text
source approval + tokenizer analysis + benchmark separation + SFT data scale-up
```

## Data And Evaluation Direction

The repo now treats data as the main project bottleneck.

Current SFT status:

- 50 tutor evaluation prompts exist, with 26 held out for benchmarking.
- A JSONL schema and validator exist for future SFT examples.
- No reviewed SFT conversation dataset is committed yet.
- The first seed gate is 100 reviewed examples.
- The first serious tutor SFT target is about 1,000 reviewed examples.

External benchmark candidates are tracked in
[open-kinyarwanda-benchmarks.md](docs/evaluation/open-kinyarwanda-benchmarks.md)
and logged in [source-log.md](docs/data/source-log.md). These rows are
evaluation-only unless the team explicitly approves a training split. Do not
copy benchmark prompts or benchmark test rows into `data/sft/`.

## CS336 Boundary

The CS336 assignment repos include their own AI-assistance policy. We will use
them for learning and reference, but we will not turn this repo into a solution
dump for course assignments.

Useful links:

- CS336 course page: https://cs336.stanford.edu/
- Assignment 1: https://github.com/stanford-cs336/assignment1-basics
- KinyaBERT paper: https://arxiv.org/abs/2203.08459

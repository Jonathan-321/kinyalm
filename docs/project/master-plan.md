# Master Plan

## Project Name

KinyaLM: Kinyarwanda Language Learning LM

## Goal

Build a class project that proves two things:

1. We understand the language-modeling stack from the inside.
2. We can turn that understanding into a useful Kinyarwanda learning assistant
   prototype.

The project has two tracks:

```text
Track A: from-scratch LM pipeline
Track B: practical tutor MVP
```

Track A is the learning track. Track B is the usefulness track.

## Final Deliverables

By August 14, the project should have:

- a working GitHub repo,
- documented data sources and permissions,
- a Kinyarwanda tokenizer analysis,
- a small from-scratch LM baseline,
- validation loss/perplexity results,
- sample generations,
- a retrieval-first tutor MVP,
- 20-30 reviewed tutor tasks,
- a data card,
- a model card,
- final report and presentation.

## Core Constraint

The hard part is not just implementing model code. The hard part is making the
project honest:

- data must be usable and permissioned,
- model quality must be reported honestly,
- tutor answers must be reviewed by humans,
- final success must not depend on large compute.

## Repo And Platform Setup

GitHub repo:

https://github.com/Jonathan-321/kinyalm

Visibility:

```text
public
```

Contributors:

- Jonathan Muhire ([@Jonathan-321](https://github.com/Jonathan-321))
- Tessy Mugisha ([@TessyMugisha](https://github.com/TessyMugisha))
- Bonheur Byiringiro ([@BonheurByiringiro](https://github.com/BonheurByiringiro))

Hugging Face starting point:

https://huggingface.co/datasets?search=kinyarwanda

Repo structure:

```text
configs/                  experiment configs
coursework/cs336/         official CS336 submodules
data/                     raw/interim/processed/external data placeholders
docs/                     planning, data, evaluation, and team docs
experiments/              local experiment outputs
logs/                     training/eval logs
notebooks/                analysis notebooks
scripts/                  helper scripts
src/kinyalm/              project package
tests/                    project tests
```

## Workstreams

### 1. Project Coordination

Owner: Jonathan Muhire

Responsibilities:

- run short check-ins,
- keep issues current,
- send professor updates,
- prevent scope creep,
- keep weekly demos small and real.

Week 1 output:

- professor email sent,
- roles assigned,
- first milestone confirmed.

### 2. Data

Owner: Tessy Mugisha

Responsibilities:

- find candidate Kinyarwanda sources,
- review licenses and permissions,
- maintain `docs/data/source-log.md`,
- separate raw, interim, and processed data,
- write the data card.

Week 1 output:

- candidate source list,
- license status for each source,
- blocked sources clearly marked.

### 3. Tokenizer

Owner: Bonheur Byiringiro

Responsibilities:

- understand CS336 BPE requirements,
- choose tokenizer evaluation examples,
- train tokenizer on small approved corpus,
- analyze Kinyarwanda fragmentation.

Week 1 output:

- 10-20 tokenizer analysis examples,
- tokenizer evaluation plan.

First technical milestone:

```text
Kinyarwanda-aware BPE tokenizer analysis on a small documented corpus.
```

### 4. Modeling

Owner: Bonheur Byiringiro

Responsibilities:

- define tiny model configs,
- implement or connect the decoder-only LM path,
- run sanity training,
- track validation loss/perplexity,
- save and resume checkpoints.

Week 1 output:

- tiny CPU-friendly config assumptions,
- compute inventory.

### 5. Tutor MVP

Owner: Jonathan Muhire

Responsibilities:

- define learning assistant behavior,
- organize approved grammar notes,
- build retrieval-first tutor flow,
- prepare demo tasks.

Week 1 output:

- 20 draft tutor tasks,
- list of required lesson materials.

### 6. Evaluation

Owner: Tessy Mugisha

Responsibilities:

- define tokenizer metrics,
- define LM metrics,
- create human tutor rubric,
- collect pass/fail review notes,
- track failure categories.

Week 1 output:

- first human scoring rubric,
- starter evaluation prompts.

### 7. Documentation

Owner: Jonathan Muhire

Responsibilities:

- keep README and docs current,
- write weekly updates,
- collect results,
- prepare report and slides.

Week 1 output:

- first weekly update draft,
- final report outline started.

## Week-By-Week Plan

### Week 1: Setup, Scope, Data Reality

Goal:

```text
Make the project runnable, assigned, and honest.
```

Tasks:

- assign roles,
- send professor email,
- create GitHub issues,
- inventory Hugging Face and other data sources,
- check licenses,
- choose tokenizer examples,
- define tiny model assumptions,
- draft tutor tasks and eval rubric.

Demo:

- repo walkthrough,
- data source log,
- risk register,
- tokenizer examples.

Do not demo:

- model quality,
- fine-tuning,
- polished UI.

### Week 2: Tokenizer And Data Cleaning

Goal:

```text
Train and evaluate a tokenizer on a small approved corpus.
```

Tasks:

- prepare local sample corpus,
- document cleaning rules,
- train BPE tokenizer,
- run encode/decode checks,
- compute tokens per word/character,
- inspect morphology-heavy examples.

Demo:

- tokenizer analysis,
- good/bad segmentations,
- source and cleaning notes.

### Week 3: Tiny LM Sanity Check

Goal:

```text
Prove the model training loop works.
```

Tasks:

- define tiny Transformer config,
- run training on tiny corpus,
- track train/validation loss,
- checkpoint and resume,
- generate samples.

Demo:

- decreasing loss curve,
- checkpoint proof,
- early samples with honest limitations.

### Week 4: Kinyarwanda LM Baseline

Goal:

```text
Train the small LM on Kinyarwanda text and evaluate honestly.
```

Tasks:

- train tokenizer + LM on cleaned Kinyarwanda corpus,
- measure validation loss/perplexity,
- inspect repetition and memorization,
- collect speaker feedback if possible.

Demo:

- baseline report,
- sample generations,
- error analysis.

### Week 5: Tutor MVP

Goal:

```text
Build the useful demo path.
```

Tasks:

- build retrieval-first assistant,
- add approved lesson notes,
- run 20-30 learning tasks,
- collect human scores,
- fix highest-impact failures.

Demo:

- tutor answering curated tasks,
- human review notes,
- limitations.

### Week 6: Evaluation And Documentation

Goal:

```text
Freeze features and make the project reproducible.
```

Tasks:

- finish data card,
- finish model card,
- finalize evaluation results,
- clean scripts and notebooks,
- prepare final report and slides.

Demo:

- final candidate demo,
- reproducibility walkthrough.

### Final Sprint: Presentation

Goal:

```text
Submit a project that is clear, honest, and demoable.
```

Tasks:

- final slides,
- live demo or recorded fallback,
- final report,
- artifact links.

## Immediate Next Decisions

Before implementation, decide:

1. Have Tessy and Bonheur accepted the GitHub collaborator invites?
2. Does the starter role split work for everyone's availability?
3. Which data sources can we use for training?
4. Which sources are reference-only?
5. What compute can the team access?
6. What does the professor expect from the final demo?

## Team Execution Plan

Day-to-day project work is tracked here:

```text
docs/project/team-execution-plan.md
```

The immediate team-facing priority is:

```text
data approval + tokenizer analysis
```

## First Implementation Target

Once the planning decisions are made, implement only this first:

```text
Tokenizer analysis on a small, documented, permission-reviewed Kinyarwanda
sample corpus.
```

Everything else depends on that data foundation.

## Evaluation And Data Scale Direction

The main bottleneck is no longer "can we sketch a training loop?" It is whether
the project has enough reviewed data and honest evaluation to support a useful
Kinyarwanda tutor demo.

Current direction:

- use approved sources only after source-log review,
- keep external benchmarks separate from SFT examples,
- start with a 100-row reviewed SFT seed file,
- scale toward about 1,000 reviewed examples before treating SFT results as
  meaningful,
- judge tutor outputs with held-out prompts and fluent-speaker review.

Open benchmark candidates are tracked in:

```text
docs/evaluation/open-kinyarwanda-benchmarks.md
```

The data scale-up plan is tracked in:

```text
docs/data/data-overhaul-plan.md
```

Early feasibility work showed an important lesson: validation loss can improve
while generated Kinyarwanda is still not learner-ready. The project should not
claim tutor quality from training loss alone.

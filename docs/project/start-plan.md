# Start Plan

## What We Do First

Start with project infrastructure and data reality, not model training.

The first implementation target is:

```text
Kinyarwanda tokenizer analysis on a small, documented corpus.
```

That is the right first step because it forces the team to confront the real
constraints:

- Do we have usable Kinyarwanda text?
- Are we allowed to use it?
- Does BPE split Kinyarwanda morphology reasonably?
- What examples will make the final presentation concrete?

## Day 1-2

- Assign team roles.
- Send professor email.
- Fill the first data source log entries.
- Pick tokenizer analysis examples.
- Confirm local environment: Python, `uv`, Git, and repo access.

## Day 3-4

- Download or collect only approved/investigating sample data.
- Write cleaning notes.
- Create a tiny corpus file locally.
- Define tokenizer evaluation metrics.

## Day 5

- Demo the repo, source log, constraints, and tokenizer-analysis plan.
- Do not demo model quality yet.

## First Implementation Sequence

1. Data source log.
2. Small local sample corpus.
3. Text cleaning script.
4. Tokenizer train/encode/decode path.
5. Tokenizer analysis report.
6. Tiny LM config only after tokenizer data is stable.

## What Counts As Progress

Good Week 1 progress looks like this:

- the repo is understandable,
- everyone has an owner area,
- blocked data sources are marked blocked,
- the professor has the proposal,
- the team can explain the first tokenizer experiment.

It does not require a trained model.

# Week 2 Task Plan

Dates: July 6 - July 12, 2026

Main repo: `Jonathan-321/kinyalm`

Local folder: `/Users/jonathanmuhire/Documents/RL`

## Week 2 Goal

Move the project from planning into the SFT path.

SFT means supervised fine-tuning: we train a strong base language model on
example Kinyarwanda user/assistant conversations so it learns the behavior we
want from the tutor.

By the end of this week, the team should have:

- approved sources clearly logged,
- a first SFT conversation dataset format,
- a held-out Kinyarwanda tutor benchmark,
- an external Kinyarwanda benchmark shortlist,
- a staged plan for scaling from 100 reviewed examples toward about 1,000,
- a QLoRA training configuration ready for GPU use,
- a smoke-test path that proves the training script can start without crashing.

## Operating Rules

- Do not commit raw datasets, model weights, checkpoints, or training logs.
- Do not fine-tune on any source until it is logged in
  `docs/data/source-log.md` with a clear license or permission status.
- Every task must leave a reviewable artifact: a file, table, script, checklist,
  or short decision note.
- The target demo path is SFT-first: conversation data, fine-tuning,
  held-out evaluation, and review.
- RL or preference optimization comes later only if the SFT model already gives
  useful answers.

## This Week's Team Deliverables

| Deliverable | Owner | Backup | Due | Done means |
| --- | --- | --- | --- | --- |
| Source approval update | Tessy Mugisha | Jonathan Muhire | Jul 10 | `docs/data/source-log.md` marks each priority source as approved, reference-only, blocked, investigate, or rejected. |
| SFT source decision | Tessy Mugisha | Jonathan Muhire | Jul 10 | Candidate conversation, translation, and lesson-example sources are marked as usable for SFT, human-reference-only, permission-needed, or blocked. |
| Tutor benchmark bank | Jonathan Muhire | Tessy Mugisha | Jul 10 | `docs/evaluation/learning-task-bank.md` has at least 50 tutor prompts grouped by task type. |
| External benchmark shortlist | Jonathan Muhire | Bonheur Byiringiro | Jul 12 | `docs/evaluation/open-kinyarwanda-benchmarks.md` lists open Kinyarwanda evaluation sets and keeps them out of training. |
| SFT data scale plan | Jonathan Muhire / Tessy Mugisha | Bonheur Byiringiro | Jul 12 | `docs/data/data-overhaul-plan.md` defines the 100-row seed gate and the 1,000-row useful target. |
| Fluent-speaker rubric | Tessy Mugisha | Bonheur Byiringiro | Jul 11 | `docs/evaluation/evaluation-plan.md` explains how reviewers score Kinyarwanda answers. |
| Tokenizer morphology packet | Bonheur Byiringiro | Jonathan Muhire | Jul 11 | `docs/tokenizer/eval-examples.tsv` contains real morphology examples and a short summary of what good tokenization should preserve. |
| QLoRA SFT run plan | Bonheur Byiringiro / Jonathan Muhire | Tessy Mugisha | Jul 11 | `docs/model/sft-run-plan.md` names the base model, dataset format, training settings, expected GPU needs, and fallback settings. |
| Week 2 update | Jonathan Muhire | Tessy Mugisha | Jul 12 | `docs/weekly/week-2-update.md` summarizes SFT readiness, blockers, and the Week 3 run target. |

## Tessy: Data And Evaluation

### W2-DATA-001: Finish Priority Source Review

What:

Update `docs/data/source-log.md` for the priority datasets and learning
resources.

How:

- Check the dataset card or source page.
- Record the visible license, terms, or permission status.
- Mark each source with one status: `approved`, `reference-only`, `blocked`,
  `investigate`, or `rejected`.
- Add a short note about privacy, bias, classroom usefulness, and whether the
  source contains conversational examples.

When:

Finish the first pass by July 10.

Acceptance criteria:

- No teammate has to guess whether a source can be used.
- Any unclear source is blocked or marked reference-only.
- SFT-usable sources are clearly separated from sources that are only useful
  for background reading.

### W2-DATA-002: Define SFT-Usable Sources

What:

Add an `SFT Use Decision` section to `docs/data/source-log.md`.

How:

For each candidate source, mark one of:

- `can train on`,
- `can convert into prompt/answer pairs`,
- `human reference only`,
- `needs professor permission`,
- `do not use`.

When:

Finish by July 10, before dataset formatting starts.

Acceptance criteria:

- The team knows which sources can become training examples.
- The team knows which sources can only help humans write examples.
- No source enters the SFT dataset without a recorded decision.

### W2-EVAL-001: Prepare Review Instructions

What:

Update `docs/evaluation/evaluation-plan.md` with a simple guide for fluent
speaker review.

How:

Add instructions for scoring:

- Kinyarwanda correctness from 1-5,
- answer usefulness from 1-5,
- grammar explanation quality from 1-5,
- register and cultural notes,
- hallucination or unsupported-claim flag,
- `not sure` option when the reviewer is uncertain.

When:

Draft by July 11.

Acceptance criteria:

- A reviewer can score model answers without needing a live explanation of the
  whole project.
- The rubric works for translation, grammar help, correction, and beginner
  conversation prompts.

## Bonheur: Tokenizer And SFT Training Plan

### W2-TOK-001: Review Kinyarwanda Morphology Examples

What:

Update `docs/tokenizer/eval-examples.tsv`.

How:

Review examples for:

- noun classes,
- common prefixes like `umu-`, `aba-`, `iki-`, `umu-`, `ku-`,
- verbs such as `ndiga`, `turiga`, `gukora`,
- apostrophes like `n'ubwo`,
- classroom sentences,
- beginner phrases a tutor would actually use.

When:

Finish by July 10.

Acceptance criteria:

- Weak or unnatural examples are replaced or marked clearly.
- At least 15 examples are confirmed or improved.
- Anything uncertain remains marked `needs-speaker-review`.

### W2-TOK-002: Draft Tokenizer And Morphology Summary

What:

Create `docs/tokenizer/week-2-tokenizer-summary.md`.

How:

Summarize:

- what tokenizer behavior we want for Kinyarwanda,
- which morphology examples are most important,
- what tokens-per-word and fragmentation mean,
- what would count as acceptable behavior for SFT.

When:

Draft by July 11.

Acceptance criteria:

- Includes at least 10 concrete Kinyarwanda examples.
- Explains why bad fragmentation can make fine-tuning harder.
- Stays tied to the tutor use case, not a generic tokenizer lecture.

### W2-MODEL-001: Draft QLoRA SFT Run Plan

What:

Create `docs/model/sft-run-plan.md`.

How:

Specify:

- base model candidate,
- why we are using an instruct-capable base model,
- SFT JSONL message format,
- QLoRA settings,
- expected GPU memory,
- batch size and fallback batch size,
- checkpoint and evaluation cadence,
- what counts as a successful first run.

When:

Draft by July 11.

Acceptance criteria:

- Ends with one concrete Week 3 training target.
- Includes fallback settings if the first GPU run runs out of memory.
- Explains the difference between base-model ability and SFT behavior in plain
  language.

## Jonathan: SFT Dataset, Benchmark, And Coordination

### W2-SFT-001: Create The SFT Dataset Schema

What:

Create `docs/data/sft-data-schema.md`.

How:

Define the exact JSONL format:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source":"...","review_status":"..."}
```

Include rules for:

- source tracking,
- reviewer status,
- language quality notes,
- duplicate removal,
- keeping validation examples separate from training examples.

When:

Draft by July 10.

Acceptance criteria:

- A teammate can format new examples without asking for the schema again.
- The schema keeps training and evaluation data separate.

### W2-SFT-002: Seed The First Conversation Set

What:

Create a first reviewed seed set for SFT planning.

How:

Add or prepare at least 100 starter examples covering:

- English-to-Kinyarwanda translation,
- Kinyarwanda-to-English translation,
- grammar explanation,
- sentence correction,
- vocabulary teaching,
- beginner dialogue,
- quiz generation,
- refusal or uncertainty when the prompt is unclear.

When:

Finish by July 11.

Acceptance criteria:

- The examples follow the schema in `docs/data/sft-data-schema.md`.
- At least 30 examples are marked for fluent-speaker review.
- No example is used for training unless its source is approved or manually
  authored by the team.

### W2-EVAL-002: Expand Tutor Benchmark To 50 Prompts

What:

Update `docs/evaluation/learning-task-bank.md`.

How:

Add enough tasks to reach at least 50 total prompts. Include:

- translation,
- grammar explanation,
- sentence correction,
- vocabulary teaching,
- quiz generation,
- beginner dialogue,
- uncertainty behavior,
- culturally sensitive phrasing.

When:

Finish by July 10.

Acceptance criteria:

- Tasks are grouped by category.
- Each task has a review focus.
- At least 15 tasks are held out from training and marked as evaluation-only.

### W2-COORD-001: Keep The Board Current

What:

Update `TASKS.md` after each check-in.

How:

- Update statuses.
- Add blockers with exact next action.
- Keep the Friday demo scope visible.

When:

After every team check-in.

Acceptance criteria:

- The task board reflects current SFT work, not an old project path.

## Friday Demo

Show:

1. Source status table with SFT use decisions.
2. SFT data schema and at least 100 seed examples or drafted examples.
3. 50-prompt tutor benchmark with held-out examples marked.
4. Tokenizer morphology packet with at least 10 concrete examples.
5. QLoRA SFT run plan with Week 3 training target and fallback settings.

## Do Not Start This Week

- RL or preference optimization.
- Polished UI work.
- Fine-tuning on unapproved datasets.
- Scraping copyrighted lesson pages.
- Training on data that has not been logged.

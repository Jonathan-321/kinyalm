# KinyaLM Two-Week SFT Execution Plan

Dates: July 8 - July 21, 2026

Repo: `Jonathan-321/kinyalm`

Local folder: `/Users/jonathanmuhire/Documents/RL`

## Goal

In the next two weeks, the team should move from planning to an actual SFT
experiment.

SFT means supervised fine-tuning: we train a strong base model on example
Kinyarwanda user/assistant conversations so it learns how the tutor should
answer.

By July 21, the project should have one of these two outcomes:

1. Best case: a seed SFT adapter trained, benchmarked, reviewed, and ready for
   the 1,000-example data expansion.
2. Blocked case: a clear written blocker showing exactly what stopped training
   and what decision or resource is needed next.

## The Two-Week Rule

The plan is not to spend two weeks debating architecture. The plan is:

- Week 1: prepare approved SFT data, held-out evaluation prompts, and the run
  configuration.
- Week 2: package the dataset, launch the first fine-tune, run benchmark
  prompts, review outputs, and prepare the next data fix.

The 100-example seed is only a gate for the pipeline and review process. A
serious tutor run should aim for about 1,000 reviewed examples after this.

Every task below has a file path, a person, and a visible output.

## Team Roles

| Person | Main Job For These Two Weeks | What They Should Not Get Distracted By |
| --- | --- | --- |
| Jonathan Muhire | Own SFT dataset format, seed examples, benchmark prompts, coordination, and integration. | Polishing UI, rebuilding the whole repo structure, or waiting for perfect data before making the first dataset. |
| Tessy Mugisha | Own source approval, language-quality review, and fluent-speaker scoring instructions. | Approving unclear data just to move fast. If the license or permission is unclear, mark it blocked or human-reference-only. |
| Bonheur Byiringiro | Own tokenizer examples, base-model choice, QLoRA settings, compute assumptions, and training-readiness notes. | Tiny toy model work that does not help the first SFT run. |

## Shared File Outputs

These are the files the team should produce or update.

| File | Owner | Purpose |
| --- | --- | --- |
| `docs/data/source-log.md` | Tessy | Shows which sources can be trained on, which are reference-only, and which are blocked. |
| `docs/data/sft-data-schema.md` | Jonathan | Defines the exact JSONL format for conversation examples. |
| `data/sft/seed_conversations.jsonl` | Jonathan | First 100-example reviewed seed set. Do not commit if it includes unapproved source material. |
| `docs/data/data-overhaul-plan.md` | Jonathan / Tessy | Plan for scaling from the seed set toward about 1,000 reviewed SFT examples. |
| `docs/evaluation/open-kinyarwanda-benchmarks.md` | Jonathan | Open external benchmark shortlist; keep these rows out of SFT training. |
| `docs/evaluation/learning-task-bank.md` | Jonathan | Held-out tutor benchmark prompts. These must not be copied into training data. |
| `docs/evaluation/evaluation-plan.md` | Tessy | Human review rubric for scoring model answers. |
| `docs/tokenizer/eval-examples.tsv` | Bonheur | Real Kinyarwanda words/sentences to check tokenizer behavior. |
| `docs/tokenizer/week-2-tokenizer-summary.md` | Bonheur | Short explanation of tokenizer risks for Kinyarwanda SFT. |
| `docs/model/sft-run-plan.md` | Bonheur + Jonathan | Base model, QLoRA settings, batch size, checkpoint plan, and OOM fallback. |
| `docs/weekly/week-2-update.md` | Jonathan | Status update for professor/team after the first SFT-readiness sprint. |

## Week 1: July 8 - July 12

Week 1 is about getting the SFT run ready. The team should end the week with
data decisions, a schema, seed examples, benchmark prompts, and a training run
plan.

### Jonathan: SFT Dataset And Benchmark

#### Task 1: Create The SFT Data Schema

What to do:

Create `docs/data/sft-data-schema.md`.

How to do it:

1. Define the format every training example must follow:

   ```json
   {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source":"manual","review_status":"needs-review","split":"train"}
   ```

2. Explain each field:
   - `messages`: the actual user prompt and assistant answer.
   - `source`: where the example came from, such as `manual`, `approved-dataset-name`, or `team-authored`.
   - `review_status`: `needs-review`, `reviewed`, `rejected`, or `approved`.
   - `split`: `train`, `validation`, or `benchmark-only`.

3. Add rules:
   - benchmark examples cannot be used for training,
   - examples from unclear sources cannot be used for training,
   - each example should have one user message and one assistant message for the first run,
   - assistant answers should be useful, direct, and beginner-friendly.

Done means:

- A teammate can open the schema and format examples without asking you what
  the JSON should look like.

#### Task 2: Draft 100 Seed Conversation Examples

What to do:

Prepare `data/sft/seed_conversations.jsonl`.

How to do it:

1. Start with manually authored examples so licensing does not block the first
   draft.
2. Use the schema from `docs/data/sft-data-schema.md`.
3. Cover at least these categories:
   - 15 greeting and beginner conversation examples,
   - 15 English-to-Kinyarwanda translation examples,
   - 15 Kinyarwanda-to-English translation examples,
   - 15 grammar explanation examples,
   - 15 sentence correction examples,
   - 10 vocabulary teaching examples,
   - 10 quiz-generation examples,
   - 5 uncertainty examples where the tutor should say it is not sure.
4. Mark every example as `needs-review` unless Tessy or another fluent speaker
   has checked it.
5. Keep the examples short and clean. The goal is a useful first run, not a
   giant messy dataset.

Done means:

- There are at least 100 valid JSONL lines.
- The examples cover more than translation.
- No benchmark-only prompt is copied into the training set.
- The team knows this is the seed gate, not the final useful SFT scale.

#### Task 3: Expand The Held-Out Benchmark To 50 Prompts

What to do:

Update `docs/evaluation/learning-task-bank.md`.

How to do it:

1. Keep the existing starter tasks.
2. Add enough new prompts to reach at least 50.
3. Add a clear `benchmark-only` note to prompts that must never enter training.
4. Use categories:
   - greeting,
   - translation,
   - grammar explanation,
   - sentence correction,
   - vocabulary,
   - quiz generation,
   - dialogue,
   - uncertainty behavior.
5. For each row, include what the reviewer should focus on.

Done means:

- The file has at least 50 prompts.
- At least 15 prompts are explicitly marked as held-out evaluation prompts.

#### Task 4: Keep The Board Current

What to do:

Update `TASKS.md` after each check-in.

How to do it:

1. Move tasks between `todo`, `in-progress`, `blocked`, `review`, and `done`.
2. If something is blocked, write the exact blocker and who can unblock it.
3. Keep the Week 2 training target visible.

Done means:

- A teammate can open `TASKS.md` and know what is happening today.

### Tessy: Data Approval And Human Review

#### Task 1: Finish Source Approval Decisions

What to do:

Update `docs/data/source-log.md`.

How to do it:

1. For each source, check the dataset card or source page.
2. Record the license or permission status.
3. Add an `SFT Use Decision` note for each useful source:
   - `can train on`,
   - `can convert into prompt/answer pairs`,
   - `human reference only`,
   - `needs professor permission`,
   - `do not use`.
4. If a source is unclear, do not approve it. Mark it `blocked` or
   `human reference only`.
5. Add a short note for privacy and quality concerns. Chat data needs extra
   caution.

Done means:

- The team can tell which sources are allowed for SFT without guessing.

#### Task 2: Review The First 100 Seed Examples

What to do:

Review `data/sft/seed_conversations.jsonl` after Jonathan drafts it.

How to do it:

1. Read each example as if it were a beginner tutor answer.
2. Mark each one:
   - `approved`,
   - `needs-fix`,
   - `rejected`,
   - `not-sure`.
3. Look for:
   - incorrect Kinyarwanda,
   - unnatural phrasing,
   - too much English,
   - answer that is too advanced for a beginner,
   - unsupported cultural claims,
   - unclear or overconfident explanations.
4. Write the exact fix when possible.

Done means:

- At least 50 examples are approved or have clear fix notes.
- Bad examples are not silently left in the training set.

#### Task 3: Strengthen The Human Review Rubric

What to do:

Update `docs/evaluation/evaluation-plan.md`.

How to do it:

1. Keep the current rubric dimensions.
2. Add a simple 1-5 scoring guide for:
   - correctness,
   - clarity,
   - usefulness,
   - cultural/register appropriateness,
   - uncertainty behavior.
3. Add a `not sure` option so reviewers are not forced to guess.
4. Add one example of a good review note and one example of a bad review note.

Done means:

- A fluent speaker can score model outputs without needing a meeting first.

### Bonheur: Tokenizer And SFT Run Readiness

#### Task 1: Improve Tokenizer Morphology Examples

What to do:

Update `docs/tokenizer/eval-examples.tsv`.

How to do it:

1. Review the existing rows.
2. Add or improve examples for:
   - noun-class prefixes like `umu-`, `aba-`, `iki-`, `ibi-`,
   - verbs like `ndiga`, `turiga`, `gukora`,
   - apostrophes like `n'ubwo`,
   - classroom phrases,
   - common beginner tutor phrases.
3. Mark each example as:
   - `confirmed`,
   - `needs-review`,
   - `replace`,
   - `not-natural`.

Done means:

- At least 15 examples are confirmed or improved.
- Any uncertain Kinyarwanda stays marked for review.

#### Task 2: Write The Tokenizer Summary

What to do:

Create `docs/tokenizer/week-2-tokenizer-summary.md`.

How to do it:

1. Explain why Kinyarwanda morphology matters for tokenization.
2. Show at least 10 real examples from `docs/tokenizer/eval-examples.tsv`.
3. Explain tokens-per-word in plain language: lower is usually easier for the
   model, but bad merges can still hurt meaning.
4. Explain what tokenizer behavior is good enough for the first SFT run.

Done means:

- The summary connects tokenizer behavior to the tutor model, not just abstract
  tokenizer metrics.

#### Task 3: Draft The QLoRA SFT Run Plan

What to do:

Create `docs/model/sft-run-plan.md`.

How to do it:

1. Pick the first base model candidate.
2. Explain why the model is a good SFT base.
3. Define the initial settings:
   - LoRA rank,
   - LoRA alpha,
   - target modules,
   - learning rate,
   - batch size,
   - gradient accumulation,
   - max sequence length,
   - checkpoint cadence,
   - evaluation cadence.
4. Add an OOM fallback. OOM means out of memory: the GPU could not fit the
   batch/model at the chosen settings.
5. Add the exact first-run success criteria:
   - training starts,
   - loss logs appear,
   - checkpoint saves,
   - sample generation works,
   - benchmark script can run after training.

Done means:

- Jonathan can use the plan to launch training without asking what settings to
  use.

## Week 1 Check-In Targets

| Date | What Must Be Ready |
| --- | --- |
| Jul 9 | Source-log review started, SFT schema drafted, tokenizer examples being reviewed. |
| Jul 10 | Source decisions mostly filled, 50 benchmark prompts drafted, schema usable. |
| Jul 11 | 100 seed examples drafted, QLoRA run plan drafted, review rubric updated. |
| Jul 12 | Team review: decide whether Week 2 training can start or name exact blockers. |

## Week 2: July 13 - July 21

Week 2 is about running the first real SFT experiment and learning from it.

### Jonathan: Package Data, Run Evaluation, Coordinate The Experiment

#### Task 1: Package Train And Validation Files

What to do:

Create processed SFT files from the seed examples.

How to do it:

1. Put train examples and validation examples in separate files:
   - `data/sft/processed/train.jsonl`
   - `data/sft/processed/validation.jsonl`
2. Keep benchmark-only prompts out of both files.
3. Check every JSONL line has:
   - one `user` message,
   - one `assistant` message,
   - a source,
   - a review status,
   - a split.
4. Remove exact duplicate prompts.
5. Create a short note in `docs/data/sft-dataset-v1-notes.md` explaining:
   - number of train examples,
   - number of validation examples,
   - source types,
   - known weaknesses.

Done means:

- Bonheur has clean files to train on.

#### Task 2: Prepare The Benchmark Output Table

What to do:

Create `eval_results/sft_week2_review.tsv` after the first model generates
answers.

How to do it:

1. Use the held-out prompts from `docs/evaluation/learning-task-bank.md`.
2. For each prompt, record:
   - prompt ID,
   - category,
   - prompt,
   - model answer,
   - correctness score,
   - clarity score,
   - usefulness score,
   - reviewer notes.
3. Leave score columns blank until Tessy reviews.

Done means:

- The team has one table where outputs and review notes live together.

#### Task 3: Write The Week 2 Update

What to do:

Create `docs/weekly/week-2-update.md`.

How to do it:

Include:

- what data was approved,
- how many SFT examples exist,
- whether training launched,
- what broke if it did not launch,
- sample outputs or where they are stored,
- next fix for Week 3.

Done means:

- The professor can understand progress without reading every repo file.

### Tessy: Review Training Data And Model Outputs

#### Task 1: Review The Train/Validation Split

What to do:

Check `data/sft/processed/train.jsonl` and
`data/sft/processed/validation.jsonl`.

How to do it:

1. Spot-check at least 30 training examples.
2. Spot-check at least 15 validation examples.
3. Confirm validation examples are not copied from training.
4. Mark examples that should be removed before training.

Done means:

- The first SFT run is not obviously poisoned by bad or duplicated examples.

#### Task 2: Score First Model Outputs

What to do:

Fill in review columns in `eval_results/sft_week2_review.tsv`.

How to do it:

1. Score each answer using `docs/evaluation/evaluation-plan.md`.
2. Add short notes for the main failure.
3. Use consistent tags:
   - `wrong-translation`,
   - `bad-grammar`,
   - `too-much-english`,
   - `too-advanced`,
   - `hallucination`,
   - `good`.
4. Pick the 10 worst outputs and explain what kind of training example would
   help fix them.

Done means:

- Week 3 data fixes are based on actual model mistakes.

### Bonheur: Launch And Monitor The First SFT Run

#### Task 1: Final Training Readiness Check

What to do:

Use `docs/model/sft-run-plan.md` to verify settings before launch.

How to do it:

1. Confirm the base model name.
2. Confirm the training and validation file paths.
3. Confirm output directory naming.
4. Confirm GPU memory and batch size.
5. Confirm checkpoint interval.
6. Confirm sample-generation interval.

Done means:

- The training command can be run without missing decisions.

#### Task 2: Launch The First SFT Run

What to do:

Start the first QLoRA SFT training run.

How to do it:

1. Use the settings in `docs/model/sft-run-plan.md`.
2. Save logs under `logs/`.
3. Save adapter outputs under an ignored model-output folder.
4. Watch the first 50-100 steps for:
   - out-of-memory errors,
   - data formatting errors,
   - loss becoming `nan`,
   - checkpoint save failures.
5. If the run OOMs, halve the batch size and restart.
6. If the data format fails, give Jonathan the exact bad line number or error.

Done means:

- Either a run is training, or the blocker is written clearly enough for the
  team to fix.

#### Task 3: Save First Training Notes

What to do:

Add training notes to `docs/model/sft-week2-training-notes.md`.

How to do it:

Record:

- model name,
- GPU type,
- batch size,
- effective batch size,
- learning rate,
- max sequence length,
- number of examples,
- checkpoint path,
- final or current loss,
- any crash and fix.

Done means:

- The run is reproducible enough for the report.

## Week 2 Check-In Targets

| Date | What Must Be Ready |
| --- | --- |
| Jul 13 | Train/validation JSONL files prepared and spot-checked. |
| Jul 14 | Training settings finalized and first run ready to launch. |
| Jul 15 | First run launched or exact blocker written down. |
| Jul 16 | First sample outputs generated, even if rough. |
| Jul 17 | Benchmark table created with model outputs. |
| Jul 18 | Tessy review notes added to the benchmark table. |
| Jul 19 | Top failure categories identified. |
| Jul 20 | Dataset fixes drafted for weak categories. |
| Jul 21 | Decision: run second SFT pass, collect more examples, or fix training blocker. |

## What Counts As Real Progress

The team should not count vague planning as progress. Count these:

- source decisions written in `docs/data/source-log.md`,
- valid SFT JSONL examples,
- held-out benchmark prompts,
- a run plan with concrete QLoRA settings,
- training logs,
- checkpoints or adapter outputs,
- benchmark outputs,
- fluent-speaker review notes,
- a clear next data fix.

## What Not To Do Yet

- Do not start RL.
- Do not build a polished UI.
- Do not train on unclear source data.
- Do not mix benchmark-only prompts into training.
- Do not spend Week 2 only writing theory notes. The goal is an actual SFT run.

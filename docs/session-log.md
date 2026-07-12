# Session Log

## 2026-06-29

Started the workspace around the full Stanford CS336 course rather than only
Assignment 1.

Decision:

```text
CS336 first, PatchLab later.
```

Reason:

CS336 teaches the inner model and training stack. PatchLab will make more sense
after the transformer, systems, data, and alignment pieces are grounded.

Initial setup:

- Added official CS336 assignment repos as submodules under `coursework/cs336`.
- Added a root README for the local workspace.
- Added roadmap and concept notes.
- Added an Assignment 1 start plan that keeps AI help in teaching-assistant
  mode instead of solution-generation mode.

Next step:

You run Assignment 1's tests once to verify the environment, then begin
implementing the adapters one concept at a time.

## 2026-07-02

Pivoted the workspace from a general CS336 learning wrapper into the concrete
Kinyarwanda Language Learning LM class project.

Decision:

```text
CS336 remains the learning backbone, but the repo's main project is now the
Kinyarwanda LM/tutor deliverable.
```

Initial project setup:

- Added project charter, roadmap, start plan, constraints/risk register, and
  professor email draft.
- Added team role matrix and Week 1 kickoff checklist.
- Added task board for the first week.
- Added data source log, data card template, evaluation plan, and weekly update
  template.
- Added root Python project config and package/test placeholders.

Next step:

Assign owners, send the professor email, and begin data-source review before
training or scraping anything.

Update:

- Made the GitHub repo public.
- Invited Tessy Mugisha and Bonheur Byiringiro as collaborators with write
  access.
- Updated contributors, handles, and starter role ownership across the project
  docs.
- Created the remaining Week 1 GitHub issues and routed pending-owner issues
  with comments until collaborator invitations are accepted.

## 2026-07-03

Started the first implementation milestone without touching CS336 assignment
solutions.

Added:

- reviewed-before-use tokenizer example set,
- tokenizer analysis plan,
- tokenizer metric utilities,
- project example loader,
- lightweight project health check,
- starter tests for tokenizer metrics.

Verified:

- `python3 scripts/check_project.py`
- `PYTHONPATH=src python3 -m pytest tests/test_tokenizer_metrics.py -q`

Exploration update:

- Added a Hugging Face metadata review script that does not download dataset
  files.
- Generated `docs/data/huggingface-source-review.md`.
- Updated the source log so datasets with no visible license stay blocked for
  training until review is complete.

Feasibility update:

- Toy and approved-corpus runs proved that the basic tokenizer, batching,
  training, checkpointing, sampling, and reporting mechanics can be wired up.
- Digital Umuganda TTS sentence text can be prepared as 3,922 unique lines for
  tokenizer and small-LM experiments.
- The Digital Umuganda MT Kinyarwanda side needs cp1252 decoding before clean
  UTF-8 processing; after decoding, it yields 44,527 clean lines with zero
  replacement characters.
- Early sample checks showed the central lesson for this repo: lower validation
  loss is not proof of useful learner-facing Kinyarwanda.
- This repo keeps team-ready decisions, gates, source review, benchmark
  separation, and reproducible project scripts rather than experiment dumps.

Team execution update:

- Added `docs/project/team-execution-plan.md`.
- Added `docs/data/source-review-checklist.md`.
- Added `docs/model/compute-inventory.md`.
- Added `docs/evaluation/learning-task-bank.md`.
- Kept the immediate repo focus on data approval and tokenizer analysis.

## 2026-07-12

Week 2 SFT-readiness update:

- Merged Bonheur's tokenizer evaluation work into the active SFT-readiness
  branch:
  - 38 tokenizer examples,
  - 37 confirmed examples,
  - one deliberate not-natural hyphen boundary check,
  - noun-class, verb morphology, and apostrophe/elision cases.
- Added `docs/data/sft-data-schema.md` for reviewed conversation JSONL.
- Added `src/kinyalm/data/sft.py` and `scripts/validate_sft_jsonl.py` so future
  SFT seed files can be checked before training.
- Expanded `docs/evaluation/learning-task-bank.md` from 20 to 50 prompts, with
  26 marked `benchmark-only`.
- Updated `docs/evaluation/evaluation-plan.md` with fluent-speaker scoring
  fields and first-demo review rules.
- Added `docs/model/sft-run-plan.md` to define QLoRA readiness, fallback
  settings, smoke tests, and Week 3 target.
- Added `data/sft/README.md` and `data/sft/.gitkeep`.
- Added `docs/weekly/week-2-update.md`.
- Updated `scripts/check_project.py` to fail if the SFT schema is missing or
  the tutor benchmark falls below 50 prompts / 15 held-out prompts.

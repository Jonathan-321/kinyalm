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

Track A sandbox update:

- Moved the runnable sandbox into standalone repo
  `https://github.com/Jonathan-321/kilm`.
- KILM contains the review-only toy corpus, character and BPE tokenizer paths,
  tiny causal Transformer, runner, run notes, and tests.
- Verified the BPE path on the toy corpus: validation loss dropped from 4.3828
  to 1.8230, and token count dropped from 558 character tokens to 386 BPE
  tokens.
- Added KILM corpus preparation, explicit train/validation corpus support,
  checkpoint save/resume, checkpoint sampling, run reports, comparison reports,
  Makefile shortcuts, and GitHub Actions CI.
- Prepared-split toy smoke run works with separate train/validation files:
  BPE validation loss moved from 4.3952 to 3.7307 on a tiny 2-line validation
  split. This is a wiring check, not model-quality evidence.
- Added the approved-corpus baseline path in KILM:
  - Digital Umuganda TTS sentence text imports into ignored local data.
  - The full local TTS import prepares into 3,922 unique lines, split into
    3,530 train / 392 validation lines.
  - A 512-vocab BPE tokenizer compresses the full prepared text from 340,384
    char tokens to 134,090 BPE tokens.
  - Morphology-focused tokenizer examples now produce a reviewable split
    report.
  - The 20-step sanity run moved validation perplexity from 602.1208 to
    522.7661.
  - The `small` MPS baseline moved validation perplexity from 605.7486 to
    137.0228 over 200 steps.
  - Run metadata records learning-rate schedule, gradient clipping, checkpoint
    interval, data card, model card, and sample-review TSV.
- This repo now keeps only the planning record and Track A gates.

Team execution update:

- Added `docs/project/team-execution-plan.md`.
- Added `docs/data/source-review-checklist.md`.
- Added `docs/model/compute-inventory.md`.
- Added `docs/evaluation/learning-task-bank.md`.
- Kept the immediate repo focus on data approval and tokenizer analysis.

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

- Added a separate `sandbox/track_a` guide and review-only toy corpus.
- Added a character-level tokenizer and tiny causal Transformer sandbox.
- Added `scripts/run_track_a_sandbox.py` to run the full toy loop end to end.
- Added tests for tokenizer round-tripping and model forward-pass shape.

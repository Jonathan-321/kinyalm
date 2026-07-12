# Task Board

Use this as the starter board until the team moves tasks into GitHub Issues.

## Week 1: Required

| ID | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| W1-001 | Confirm team members and role owners | Jonathan Muhire | todo | Review `docs/team/roles.md` with Tessy and Bonheur. |
| W1-002 | Send proposal email to professor | Jonathan Muhire | todo | Use `docs/project/professor-email.md`. |
| W1-003 | Inventory Kinyarwanda data sources | Tessy Mugisha | in-progress | Digital Umuganda TTS and MT sources are approved; Tessy should add non-HF sources and SFT-use decisions. |
| W1-004 | Check source licenses and permissions | Tessy Mugisha | in-progress | Digital Umuganda TTS/MT passed the first gate; remaining sources need approved, reference-only, blocked, investigate, or rejected status. |
| W1-005 | Pick 10-20 tokenizer analysis examples | Bonheur Byiringiro | done | 38 examples in `docs/tokenizer/eval-examples.tsv`: 37 confirmed, 1 deliberate not-natural boundary check. |
| W1-006 | Read CS336 tokenizer requirements | Bonheur Byiringiro | todo | Do not copy solutions into this repo. |
| W1-007 | Define SFT base-model and compute assumptions | Bonheur Byiringiro | todo | Pick the likely instruct base model, GPU need, and fallback batch size. |
| W1-008 | Draft 20 tutor/SFT evaluation tasks | Jonathan Muhire | done | Expanded `docs/evaluation/learning-task-bank.md` to 50 prompts with held-out benchmark prompts. |
| W1-009 | Draft human scoring rubric | Tessy Mugisha | in-progress | Rubric exists; review `docs/evaluation/evaluation-plan.md` and `docs/evaluation/learning-task-bank.md`. |
| W1-010 | Prepare first weekly update | Jonathan Muhire | done | Initial update exists in `docs/weekly/week-1-update.md`. |

## Week 1: Nice To Have

| ID | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| W1-011 | Create repo labels | Jonathan Muhire | done | `data`, `tokenizer`, `model`, `tutor`, `eval`, `docs`, `blocked`. |
| W1-012 | Add starter tokenizer-analysis scaffold | Bonheur Byiringiro | in-progress | Starter examples, metrics helpers, and tests are in place; notebook/report can build on them. |
| W1-013 | Create shared data spreadsheet | Tessy Mugisha | todo | Mirror important entries back into repo docs. |
| W1-014 | Collect compute inventory | Bonheur Byiringiro | in-progress | Template added in `docs/model/compute-inventory.md`. |
| W1-015 | Use team execution plan for next work | Jonathan Muhire | done | Added `docs/project/team-execution-plan.md`. |
| W1-016 | Record two-week SFT execution plan | Jonathan Muhire | done | Added `docs/team/two-week-sft-execution-plan.md` and aligned `docs/team/week-2-task-plan.md`. |

## Week 2: Required

Detailed owner notes live in `docs/team/week-2-task-plan.md`.

| ID | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| W2-001 | Finish priority source review | Tessy Mugisha | todo | Update `docs/data/source-log.md`; mark sources approved, reference-only, blocked, investigate, or rejected. |
| W2-002 | Define SFT-usable sources | Tessy Mugisha | todo | Add an `SFT Use Decision` section to `docs/data/source-log.md`. |
| W2-003 | Prepare fluent-speaker review instructions | Tessy Mugisha | todo | Update `docs/evaluation/evaluation-plan.md` with scoring instructions. |
| W2-004 | Review tokenizer morphology examples | Bonheur Byiringiro | todo | Update `docs/tokenizer/eval-examples.tsv`; replace weak or unnatural examples. |
| W2-005 | Draft tokenizer and morphology summary | Bonheur Byiringiro | todo | Create `docs/tokenizer/week-2-tokenizer-summary.md`. |
| W2-006 | Write QLoRA SFT run plan | Bonheur Byiringiro / Jonathan Muhire | in-progress | Draft added in `docs/model/sft-run-plan.md`; Bonheur still needs to fill the final base-model decision. |
| W2-007 | Expand tutor benchmark to 50 tasks | Jonathan Muhire | done | `docs/evaluation/learning-task-bank.md` now has 50 prompts and 26 benchmark-only prompts. |
| W2-008 | Define SFT dataset schema and seed examples | Jonathan Muhire | in-progress | Schema and validator added; 100 reviewed seed examples still need authoring and fluent-speaker review. |
| W2-009 | Keep Week 2 board current | Jonathan Muhire | in-progress | Board updated for the SFT-readiness branch. |
| W2-010 | Prepare Week 2 written update | Jonathan Muhire | todo | Create `docs/weekly/week-2-update.md` by Jul 12. |

## Week 2: Friday Demo Target

| ID | Demo Item | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| W2-D01 | Source status table | Tessy Mugisha | todo | Show what is approved, blocked, reference-only, or SFT-usable. |
| W2-D02 | Tokenizer examples and fragmentation notes | Bonheur Byiringiro | todo | Show at least 8 concrete Kinyarwanda examples. |
| W2-D03 | Tutor benchmark bank | Jonathan Muhire | done | 50 prompts, categories, and held-out examples are in `docs/evaluation/learning-task-bank.md`. |
| W2-D04 | SFT run plan | Jonathan Muhire / Bonheur Byiringiro | in-progress | Schema, validator, QLoRA draft, and OSCER smoke scripts exist; final base model and seed data are still pending. |

## Do Not Start Yet

- Fine-tuning on unapproved datasets.
- RL or preference optimization.
- Scraping ambiguous copyrighted sources.
- Building a polished UI.
- Training on data that has not been logged.

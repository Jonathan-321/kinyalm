# Task Board

Use this as the starter board until the team moves tasks into GitHub Issues.

## Week 1: Required

| ID | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| W1-001 | Confirm team members and role owners | Jonathan Muhire | todo | Review `docs/team/roles.md` with Tessy and Bonheur. |
| W1-002 | Send proposal email to professor | Jonathan Muhire | todo | Use `docs/project/professor-email.md`. |
| W1-003 | Inventory Kinyarwanda data sources | Tessy Mugisha | in-progress | Digital Umuganda TTS and MT sources are now logged as approved for KILM; Tessy should add any non-HF sources. |
| W1-004 | Check source licenses and permissions | Tessy Mugisha | in-progress | Digital Umuganda TTS/MT passed first approved-data gate; Tessy should add attribution/domain notes for final cards. |
| W1-005 | Pick 10-20 tokenizer analysis examples | Bonheur Byiringiro | in-progress | KILM has 17 morphology/orthography examples; Bonheur should validate or replace weak examples. |
| W1-006 | Read CS336 tokenizer requirements | Bonheur Byiringiro | todo | Do not copy solutions into this repo. |
| W1-007 | Define tiny model config assumptions | Bonheur Byiringiro | todo | CPU-friendly first. |
| W1-008 | Draft 20 tutor MVP tasks | Jonathan Muhire | todo | Grammar, vocabulary, correction, quizzes. |
| W1-009 | Draft human scoring rubric | Tessy Mugisha | in-progress | Rubric exists; review `docs/evaluation/evaluation-plan.md` and `docs/evaluation/learning-task-bank.md`. |
| W1-010 | Prepare first weekly update | Jonathan Muhire | done | Initial update exists in `docs/weekly/week-1-update.md`. |

## Week 1: Nice To Have

| ID | Task | Owner | Status | Notes |
| --- | --- | --- | --- | --- |
| W1-011 | Create repo labels | Jonathan Muhire | done | `data`, `tokenizer`, `model`, `tutor`, `eval`, `docs`, `blocked`. |
| W1-012 | Add starter tokenizer-analysis scaffold | Bonheur Byiringiro | in-progress | Starter examples, metrics helpers, and tests are in place; notebook/report can build on them. |
| W1-013 | Create shared data spreadsheet | Tessy Mugisha | todo | Mirror important entries back into repo docs. |
| W1-014 | Collect compute inventory | Bonheur Byiringiro | in-progress | Template added in `docs/model/compute-inventory.md`. |
| W1-015 | Run Track A sandbox and interpret results | Jonathan Muhire | done | Moved to standalone `Jonathan-321/kilm` repo; use it as a guide, not final model evidence. |
| W1-016 | Use team execution plan for next work | Jonathan Muhire | done | Added `docs/project/team-execution-plan.md`. |
| W1-017 | Run longer approved-data KILM baseline | Jonathan Muhire | done | 10k continuation moved validation perplexity from 139.1711 to 59.5324, but sample quality still failed. |
| W1-018 | Review KILM generated samples | Tessy Mugisha | blocked | Current sample is marked `failed-smoke`; do not spend fluent-review time until samples are recognizably Kinyarwanda. |
| W1-019 | Plan next KILM scale change | Jonathan Muhire / Bonheur Byiringiro | todo | Compare larger context/model and more approved text; do not just repeat the same small run forever. |

## Do Not Start Yet

- Large model training.
- Fine-tuning a base model.
- Scraping ambiguous copyrighted sources.
- Building a polished UI.
- Training on data that has not been logged.

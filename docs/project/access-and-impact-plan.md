# Access And Impact Plan

## Current Access State

The KinyaLM datalake is public-gated:

```text
https://huggingface.co/datasets/kinyalm/kinyalm-data-lake
```

Anyone can view the dataset page. Files require a logged-in Hugging Face account
and gated access acceptance.

This is the fastest useful access mode for review. It lets teammates and fluent
speakers get the files without waiting for a private repo invite.

## Impact Target

The next milestone is not more raw data. It is reviewed, trainable data.

Batch 001 has 286 draft SFT rows. The first measurable target is:

```text
100 reviewed rows
75 approved rows
1 approved train/validation pack
1 QLoRA smoke run using only approved rows
1 reviewer notes summary
```

## Access Funnel

| Stage | Owner | Output |
| --- | --- | --- |
| Share link | Jonathan | Tessy, Bonheur, and extra reviewers have the HF link |
| Accept gated access | Each reviewer | Reviewer can download the TSV |
| Stage new data | Contributor | Files packaged under `incoming/<username>/<batch-id>/` |
| Submit contribution | Contributor | HF PR or GitHub issue with source/permission notes |
| Triage contribution | Jonathan, data owner | Accepted files move to draft/review folders |
| Review rows | Tessy, Bonheur, extra reviewers | TSV rows have status, scores, and notes |
| Promote approved rows | Jonathan | `train.jsonl`, `validation.jsonl`, and promotion summary |
| Train smoke run | Bonheur | First small SFT run on approved-only rows |
| Error review | Tessy, Jonathan | Failure tags and next data-generation targets |

## Write Access Plan

The shared `kinyalm` Hugging Face organization and public-gated datalake are
active. Regular contributors should receive organization repository write
access. Outside contributors and teammates waiting for an invite should use HF
pull requests.

Remaining inputs for teammate invitations:

```text
Tessy HF username
Bonheur HF username
```

All uploads, including direct organization uploads, must go through `incoming/`
and include a `CONTRIBUTION.md` file.

## Reviewer Throughput

Use small review blocks:

```text
20 rows per review pass
30 minutes per pass
one task type per pass when possible
```

This makes it easier to catch patterns like unnatural greetings, bad grammar
explanations, or translations that are technically correct but not useful for a
beginner.

## Public Impact Without Losing Control

Good public-facing signal:

- public GitHub repo,
- public-gated HF datalake,
- clear reviewer guide,
- clear training gate,
- transparent row counts,
- eventual approved mini dataset with a data card.

Do not publish as a final training dataset until:

- source status is clean,
- reviewer status is clean,
- failure notes are summarized,
- benchmark-only rows are excluded,
- the data card says exactly what the data can and cannot be used for.

## Success Metrics

Track these weekly:

| Metric | Why it matters |
| --- | --- |
| reviewers with HF access | access is not real until people can open files |
| rows reviewed | tells whether the team is moving |
| rows approved | tells whether we can train |
| needs-fix rows | tells what to repair or regenerate |
| rejected rows | protects model quality |
| approved rows by task type | prevents one-category overfitting |
| first approved train/validation pack | turns review into training input |
| held-out eval failures | tells whether the model helps learners |

## 15-Agent Parallel Execution Map

Use these lanes when spinning up parallel agents. Each lane has a separate
responsibility to avoid duplicated work.

If the environment limits active agents, run these in waves. Start with access,
review workflow, Batch 001 coverage, review rubric, source licensing, and SFT
schema/promotion safety.

| Agent | Lane | Output |
| ---: | --- | --- |
| 1 | HF access QA | Verify public-gated access path and list failure modes |
| 2 | Reviewer onboarding QA | Check whether a new reviewer can follow the docs |
| 3 | Batch 001 coverage | Summarize row counts by task type and missing categories |
| 4 | Review rubric | Tighten scoring and failure tags |
| 5 | Data source licensing | Update source-log risk and approval status |
| 6 | SFT schema | Check that generated rows match schema and promotion needs |
| 7 | Promotion pipeline | Test approved-row promotion on a tiny fixture |
| 8 | Evaluation bank | Expand held-out learner tasks without leaking benchmarks |
| 9 | Kinyarwanda benchmarks | Verify which open benchmarks are evaluation-only |
| 10 | Tokenizer analysis | Prepare morphology-heavy tokenizer examples |
| 11 | Base-model bake-off | Compare 7B, 9B, 14B, and 30B-plus options by license and Kinyarwanda risk |
| 12 | QLoRA run readiness | Check config, GPU assumptions, and smoke-run command |
| 13 | Cloud/HPC readiness | Confirm OSCER/cloud setup and storage paths |
| 14 | Community contribution | Draft issue templates and reviewer invite flow |
| 15 | Weekly impact report | Turn progress into a professor/team update |

The main agent should keep ownership of integration, Git history, and final
training gates.

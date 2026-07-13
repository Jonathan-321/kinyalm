# HF Datalake Reviewer Onboarding

This guide is for teammates and Kinyarwanda speakers reviewing Batch 001.

## Access

Open:

```text
https://huggingface.co/datasets/kinyalm/kinyalm-data-lake
```

Sign in to Hugging Face and accept the gated access prompt. The dataset page is
public, but file access requires a logged-in HF account.

Access is per Hugging Face user. Automatic approval should unlock the files
immediately in the browser after you accept the prompt. Browser access is enough
for review; command-line access needs a separate HF token and is not required.

If access fails:

1. confirm you are signed in to Hugging Face,
2. refresh the dataset page,
3. tell Jonathan the exact error message.

## First Review Target

Start with Batch 001:

```text
review/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.review.tsv
```

Batch 001 has 286 draft rows. The first impact target is:

```text
100 reviewed rows, with at least 75 approved rows.
```

That is enough to create the first small approved SFT pack and run a real
training smoke test.

## Review Columns

Open the review TSV in a spreadsheet editor. Only edit these columns:

| Column | What to enter |
| --- | --- |
| `review_status` | `approved`, `needs-fix`, `rejected`, `not-sure`, or leave `needs-review` for untouched rows |
| `correctness_1_5` | 1 means wrong, 5 means fully correct |
| `naturalness_1_5` | 1 means unnatural, 5 means natural Kinyarwanda |
| `helpfulness_1_5` | 1 means not useful for a learner, 5 means clearly useful |
| `failure_tags` | comma-separated problem tags when the row is not approved |
| `reviewer` | your name or GitHub username |
| `reviewer_notes` | short fix note, correction, or reason |

Do not edit `id`, `task_type`, `user_prompt`, or `assistant_response`. If the
answer needs different wording, mark `review_status` as `needs-fix` and put the
exact suggested replacement in `reviewer_notes`. The current promotion script
does not use edited `assistant_response` text.

Literal `\n` in the TSV means a line break. It is not a Kinyarwanda character
problem.

Only rows marked exactly `approved` are promoted by:

```bash
python3 scripts/promote_reviewed_sft.py
```

Approved rows must have correctness, naturalness, and helpfulness scores of 4
or 5, a reviewer name, and no failure tags.

Use these comma-separated failure tags when helpful:

```text
wrong-translation, wrong-grammar, awkward-kinyarwanda, unclear, too-advanced,
too-much-english, too-much-kinyarwanda, unsupported-claim,
hallucinated-culture, overconfident-uncertainty, bad-register, unsafe,
source-risk, benchmark-leak, privacy-risk, format-error, duplicate, off-task
```

## Approval Rule

Mark a row `approved` only if all of these are true:

- Kinyarwanda is correct enough for a beginner learning tool.
- English explanation or translation is accurate.
- The answer is natural, not just technically understandable.
- The row does not teach a bad habit.
- Scores are usually 4 or 5 for correctness, naturalness, and helpfulness.

Use `needs-fix` when the idea is useful but the wording needs correction.
Use `rejected` when the row should not train the model.
Use `not-sure` when a fluent speaker or source check is needed.
Leave `needs-review` only for rows you did not review.

## How To Return Review

1. Download the review TSV from Hugging Face.
2. Open it in Google Sheets, Excel, LibreOffice, or another spreadsheet editor.
3. Fill only the review columns listed above.
4. Save or export it as TSV or tab-separated text.
5. Send the edited TSV back to Jonathan.

Do not rename columns. Do not send a screenshot as the only review artifact.

## Suggested Split

To move quickly:

| Reviewer | First block |
| --- | --- |
| Tessy | data rows 1-100, spreadsheet rows 2-101 |
| Bonheur | data rows 101-200, spreadsheet rows 102-201 |
| Jonathan or extra reviewer | data rows 201-286 and conflict checks |

After the first pass, Jonathan promotes only approved rows into
`~/KinyaLMData/approved/`.

## What Not To Do

- Do not train from the `draft` folder.
- Do not mark uncertain rows as approved.
- Do not copy benchmark-only prompts into training rows.
- Do not rewrite large sections without leaving a reviewer note.

The goal is not to approve everything. The goal is to find the rows that are
actually safe and useful enough to train on.

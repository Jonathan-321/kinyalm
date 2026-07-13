# Access Message Templates

Use these messages when inviting teammates or outside Kinyarwanda reviewers.

## Tessy And Bonheur

```text
Hey, the KinyaLM review datalake is ready here:

https://huggingface.co/datasets/kinyalm/kinyalm-data-lake

Please sign in to Hugging Face and accept the gated access prompt. Start with
the Batch 001 review TSV:

review/sft-drafts-2026-07-13-batch-001/sft-drafts-2026-07-13-batch-001.review.tsv

Review status values are: approved, needs-fix, rejected, not-sure.

When done, send the edited TSV back to Jonathan. Do not rename columns, and do
not edit id, task_type, user_prompt, or assistant_response directly.

The first target is 100 reviewed rows with at least 75 approved rows. The guide
is in docs/team/hf-datalake-reviewer-onboarding.md.
```

## Extra Kinyarwanda Reviewer

```text
Hi, we are building a small Kinyarwanda learning tutor project and need fluent
speaker review before any data is used for training.

The draft review dataset is here:

https://huggingface.co/datasets/kinyalm/kinyalm-data-lake

It is public-gated, so you can sign in to Hugging Face and accept access. We are
not asking you to approve everything. The useful review is: which rows are
correct, natural, and helpful for a beginner, and which ones need fixing or
should be rejected.

When done, send the edited TSV back to Jonathan as TSV/tab-separated text.
```

## Public Project Update

```text
KinyaLM now has a public-gated Hugging Face review datalake:

https://huggingface.co/datasets/kinyalm/kinyalm-data-lake

Batch 001 has 286 draft SFT examples. They are not approved training data yet.
The next milestone is fluent-speaker review, then promotion of approved rows
into a small train/validation pack.
```

## Access Help Reply

```text
If the HF page opens but files are blocked, make sure you are signed in and have
accepted the gated access prompt on the dataset page. If it still fails, send a
screenshot or the exact error text so we can debug it.
```

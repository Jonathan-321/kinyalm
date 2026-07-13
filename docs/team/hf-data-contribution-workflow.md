# HF Data Contribution Workflow

Use this when Tessy, Bonheur, or another teammate needs to add data to the
KinyaLM Hugging Face datalake.

## Current Reality

The current datalake is:

```text
https://huggingface.co/datasets/Jonnyyy/kinyalm-data-lake
```

It is public-gated. Teammates can read and download after accepting access, but
they do not automatically get write access.

There are two contribution paths:

| Path | Best for | Who merges |
| --- | --- | --- |
| HF pull request | fastest without shared write access | Jonathan |
| Shared HF organization | direct team uploads | org members with write/admin |

## Path A: HF Pull Request

Use this now if teammates do not have write access to the dataset repo.

1. Open the dataset:

   ```text
   https://huggingface.co/datasets/Jonnyyy/kinyalm-data-lake
   ```

2. Sign in and accept gated access.
3. Go to `Community`.
4. Create a pull request or PR branch.
5. Add files under `incoming/<github-or-hf-username>/<batch-id>/`.
6. Include a `CONTRIBUTION.md` file.
7. Do not modify existing approved or review files unless the PR is specifically
   for reviewer corrections.

Suggested upload layout:

```text
incoming/tessy/batch-002/
  CONTRIBUTION.md
  data.jsonl
  source-notes.md
  review-plan.md
```

Jonathan reviews the PR, checks source status, and moves accepted files into the
right datalake folder.

## Path B: Shared HF Organization

Use this once the team wants direct write access.

1. Create a Hugging Face organization, for example:

   ```text
   kinyalm-team
   ```

2. Invite Tessy and Bonheur with write access.
3. Transfer or mirror the dataset to:

   ```text
   kinyalm-team/kinyalm-data-lake
   ```

4. Update repo defaults:

   ```bash
   python3 scripts/upload_hf_datalake.py \
     --repo-id kinyalm-team/kinyalm-data-lake \
     --commit-message "Sync datalake to team org"
   ```

Direct write access should still use the same folder rules. Team uploads go to
`incoming/` first. Jonathan or the data owner promotes clean files after source
and review checks.

## Contributor Rules

Every contribution needs:

- source owner or author,
- license or permission status,
- whether it can be used for training,
- whether it can be redistributed,
- whether it contains private/sensitive data,
- row count if known,
- reviewer or fluent-speaker plan,
- intended folder: draft, review, approved, benchmark-only, or source-only.

Do not upload:

- private chats,
- personal data,
- benchmark test rows for SFT,
- scraped material without permission,
- raw files with unclear license,
- model-generated rows marked as approved.

## Minimal CONTRIBUTION.md

```markdown
# Contribution

Contributor:
HF username:
GitHub username:
Batch ID:

## Files

- data.jsonl
- source-notes.md

## Source

Where did this come from?

## Permission

Can this be used for training? yes/no/unknown
Can this be redistributed? yes/no/unknown
License or permission note:

## Review

Who should review this?
What should reviewers check?

## Safety

Does it contain private or sensitive data? yes/no/unknown
Does it include benchmark-only rows? yes/no/unknown
```

## Merge Rule

Adding data to HF does not make it trainable.

Data becomes trainable only after it passes:

1. source approval,
2. fluent-speaker or task-owner review,
3. schema validation,
4. promotion into train/validation by project scripts.

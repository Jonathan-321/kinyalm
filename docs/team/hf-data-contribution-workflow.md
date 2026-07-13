# HF Data Contribution Workflow

Use this when Tessy, Bonheur, or another teammate needs to add data to the
KinyaLM Hugging Face datalake.

## Current Reality

The current datalake is:

```text
https://huggingface.co/datasets/kinyalm/kinyalm-data-lake
```

It is public-gated and owned by the `kinyalm` Hugging Face organization.
Teammates can read and download after accepting access. Direct uploads require
organization membership with repository write access.

There are two contribution paths:

| Path | Best for | Who merges |
| --- | --- | --- |
| Organization upload | regular team contributions | data owner or maintainer |
| HF pull request | contributors without organization write access | org maintainer |

## Path A: Organization Upload

Use this for Tessy, Bonheur, and other regular contributors after they join the
`kinyalm` organization.

1. Ask an organization owner to add the contributor's exact Hugging Face
   username with repository write access.
2. Open the dataset:

   ```text
   https://huggingface.co/datasets/kinyalm/kinyalm-data-lake
   ```

3. Add files under `incoming/<hf-username>/<batch-id>/`.
4. Include a `CONTRIBUTION.md` file.
5. Do not modify existing approved or review files unless the upload is specifically
   for reviewer corrections.

Suggested upload layout:

```text
incoming/tessy/batch-002/
  CONTRIBUTION.md
  data.jsonl
  source-notes.md
  review-plan.md
```

The data owner checks source status and moves accepted files into the right
datalake folder.

## Path B: HF Pull Request

Use this for outside contributors or while a teammate is waiting for an
organization invitation.

1. Sign in and accept gated access.
2. Go to `Community` and open a pull request.
3. Add files under `incoming/<hf-username>/<batch-id>/`.
4. Include `CONTRIBUTION.md` with source, permission, review, and safety notes.
5. Ask an organization maintainer to review and merge the pull request.

Both paths use the same folder rules. Direct write access never bypasses
`incoming/`, source review, or the training promotion gate.

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

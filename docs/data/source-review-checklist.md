# Source Review Checklist

Use this checklist before any data source moves from `blocked` or
`investigate` to `approved`.

## Required Fields

For every source, confirm:

- source name,
- source URL or location,
- creator or publisher,
- original data source if different from uploader,
- license,
- whether training use is allowed,
- whether redistribution is allowed,
- whether quoted examples are allowed in the report,
- known domain bias,
- known quality concerns,
- whether a data card exists.

## Status Rules

### `approved`

Use only when:

- license or permission is clear,
- intended use is clear,
- attribution requirements are understood,
- source quality is good enough for the specific use.

### `reference-only`

Use when:

- the source can inform humans,
- but training, redistribution, or copying examples is not clearly allowed.

### `blocked`

Use when:

- no visible license exists,
- source provenance is unclear,
- privacy/consent risk is present,
- the source is copyrighted and permission is not granted,
- the source has quality problems that would mislead the model.

### `rejected`

Use when:

- the source cannot be used for this project,
- or the quality/risk tradeoff is not worth further work.

## First-Pass Notes From Hugging Face Review

Likely deeper-review candidates:

- DigitalUmuganda Kinyarwanda-English MT dataset: visible `cc-by-4.0`.
- mbazaNLP Kinyarwanda-English parallel dataset: visible `cc-by-4.0`, but
  religious-source bias is documented.
- Mikecyane Kinyarwanda chat: visible `apache-2.0`, but privacy/consent review
  is required because the visible file is a WhatsApp chat.

Blocked until license/provenance review:

- mbazaNLP Kinyarwanda language model dataset.
- RogerB Kinyarwanda Wikipedia snapshot.
- DigitalUmuganda Common Voice Kinyarwanda text dataset.
- saillab Alpaca Kinyarwanda cleaned.
- ChrisToukmaji Kinyarwanda instruction tuning.

## Approval Note Template

```text
Source:
Reviewer:
Date:
Decision: approved / reference-only / blocked / rejected
Allowed use:
License:
Attribution requirement:
Redistribution allowed:
Training allowed:
Report examples allowed:
Risks:
Next action:
```

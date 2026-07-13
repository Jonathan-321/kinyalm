---
name: SFT row review
about: Report a Batch 001 row that needs approval, fixing, or rejection
title: "SFT row review: "
labels: data, review
---

## Row ID

Example: `sftdraft-b001-0001-grammarexplanation`

## Recommended status

Choose one:

- [ ] approved
- [ ] needs-fix
- [ ] rejected
- [ ] not-sure

## Scores

Approved rows should have all three scores at 4 or 5.

| Score | Value |
| --- | --- |
| correctness_1_5 |  |
| naturalness_1_5 |  |
| helpfulness_1_5 |  |

## Failure tags

Use comma-separated tags when helpful:

```text
wrong-translation, wrong-grammar, awkward-kinyarwanda, unclear, too-advanced,
too-much-english, too-much-kinyarwanda, unsupported-claim,
hallucinated-culture, overconfident-uncertainty, bad-register, unsafe,
source-risk, benchmark-leak, privacy-risk, format-error, duplicate, off-task
```

Leave this blank for approved rows.

## Notes or correction

Write the suggested correction or reason here.

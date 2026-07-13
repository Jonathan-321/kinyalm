# SFT Draft Batches

This file tracks generated SFT candidate batches. These are not approved
training datasets until fluent-speaker review promotes individual rows.

## Batch 001

| Field | Value |
| --- | --- |
| Dataset ID | `sft-drafts-2026-07-13-batch-001` |
| Status | draft, needs review |
| Rows | 286 |
| JSONL | `/Users/jonathanmuhire/KinyaLMData/drafts/sft-drafts-2026-07-13-batch-001.jsonl` |
| Review TSV | `/Users/jonathanmuhire/KinyaLMData/reviewed/sft-drafts-2026-07-13-batch-001.review.tsv` |
| JSONL Manifest | `data/manifests/sft-drafts-2026-07-13-batch-001.json` |
| Review Manifest | `data/manifests/sft-drafts-2026-07-13-batch-001-review-sheet.json` |
| Can Train? | no |
| Next Action | Upload JSONL and review TSV to Google Drive review queue. |

Promotion command after review:

```bash
python3 scripts/promote_reviewed_sft.py \
  --draft-jsonl /Users/jonathanmuhire/KinyaLMData/drafts/sft-drafts-2026-07-13-batch-001.jsonl \
  --review-tsv /Users/jonathanmuhire/KinyaLMData/reviewed/sft-drafts-2026-07-13-batch-001.review.tsv \
  --out-dir /Users/jonathanmuhire/KinyaLMData/approved/sft-approved-batch-001
```

Category counts:

| Task Type | Rows |
| --- | ---: |
| culture-register | 5 |
| dialogue | 35 |
| grammar-explanation | 45 |
| greeting | 18 |
| quiz-generation | 20 |
| sentence-correction | 45 |
| translation-en-rw | 34 |
| translation-rw-en | 34 |
| uncertainty | 5 |
| vocabulary | 45 |

Review rule:

```text
Rows with split=draft or review_status=needs-review must not train.
```

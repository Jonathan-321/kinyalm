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
| Upload Package | `/Users/jonathanmuhire/KinyaLMData/packages/sft-drafts-2026-07-13-batch-001-review-package.zip` |
| Package Manifest | `data/manifests/sft-drafts-2026-07-13-batch-001-review-package.json` |
| HF Data Lake | `https://huggingface.co/datasets/kinyalm/kinyalm-data-lake` |
| HF Upload Manifest | `data/manifests/hf-datalake-sft-drafts-2026-07-13-batch-001.json` |
| Can Train? | no |
| Next Action | Review the first 100 rows and promote only rows that pass the fluent-speaker gate. |

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

## Batch 002

| Field | Value |
| --- | --- |
| Dataset ID | `sft-drafts-2026-07-13-batch-002` |
| Status | draft, needs review |
| Rows | 714 |
| Generation Profile | `data/sft/draft-profiles/useful-gap-v1.yaml` |
| JSONL | `/Users/jonathanmuhire/KinyaLMData/drafts/sft-drafts-2026-07-13-batch-002.jsonl` |
| Review TSV | `/Users/jonathanmuhire/KinyaLMData/reviewed/sft-drafts-2026-07-13-batch-002.review.tsv` |
| Review Shards | Three non-overlapping TSVs, 238 rows each |
| JSONL Manifest | `data/manifests/sft-drafts-2026-07-13-batch-002.json` |
| Review Manifest | `data/manifests/sft-drafts-2026-07-13-batch-002-review-sheet.json` |
| Upload Package | `/Users/jonathanmuhire/KinyaLMData/packages/sft-drafts-2026-07-13-batch-002-review-package.zip` |
| Package Manifest | `data/manifests/sft-drafts-2026-07-13-batch-002-review-package.json` |
| HF Data Lake | `https://huggingface.co/datasets/kinyalm/kinyalm-data-lake` |
| HF Upload Manifest | `data/manifests/hf-datalake-sft-drafts-2026-07-13-batches-001-002.json` |
| Can Train? | no |
| Next Action | Complete first-pass shard review, then have a fluent speaker approve or reject each candidate. |

Category counts:

| Task Type | Rows |
| --- | ---: |
| culture-register | 25 |
| dialogue | 85 |
| grammar-explanation | 115 |
| greeting | 62 |
| quiz-generation | 60 |
| sentence-correction | 115 |
| translation-en-rw | 66 |
| translation-rw-en | 66 |
| uncertainty | 25 |
| vocabulary | 95 |

## Combined Draft Pool

Batch 001 and Batch 002 contain exactly 1,000 draft rows in the category mix
defined by `docs/data/data-overhaul-plan.md`. Generation and schema validation
are complete, but the useful milestone remains **1,000 reviewed rows**, not
1,000 generated rows.

Batch 002 was checked for exact prompt/answer collisions against the corrected
Batch 001, and no collisions remain. Batch 001's four duplicate quiz rows were
replaced, and each quiz answer now matches the question count requested by its
prompt.

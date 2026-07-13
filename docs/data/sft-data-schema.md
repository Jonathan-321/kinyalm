# SFT Data Schema

SFT means supervised fine-tuning: training a base model on examples of the
conversation behavior we want.

This project should not fine-tune on raw scraped text. The first SFT dataset
should be a small JSONL file of reviewed tutor conversations, with every line
traceable to a source decision in `docs/data/source-log.md`.

## File Location

Use this path for the first local dataset:

```text
data/sft/seed_conversations.jsonl
```

Do not commit the file until the examples are small, intentionally shareable,
and reviewed. Raw exports, model-generated drafts, and large generated files
should stay local.

## JSONL Record

Each line is one JSON object:

```json
{"id":"sft-greeting-001","task_type":"greeting","split":"draft","source":"team-authored","source_status":"team-authored","review_status":"needs-review","language_mix":"kinyarwanda+english","messages":[{"role":"user","content":"Teach me how to greet someone in Kinyarwanda."},{"role":"assistant","content":"Muraho means hello. You can use it as a general polite greeting."}],"reviewer_notes":""}
```

## Required Fields

| Field | Type | Required Values / Rule |
| --- | --- | --- |
| `id` | string | Stable unique ID, lowercase with hyphens or numbers. |
| `task_type` | string | One of the task types below. |
| `split` | string | `draft`, `train`, `validation`, or `benchmark-only`. |
| `source` | string | `team-authored`, `manual`, or a source name from `source-log.md`. |
| `source_status` | string | `approved`, `team-authored`, `manual`, `reference-only`, `blocked`, or `investigate`. |
| `review_status` | string | `needs-review`, `approved`, `needs-fix`, `rejected`, or `not-sure`. |
| `language_mix` | string | `kinyarwanda`, `english`, or `kinyarwanda+english`. |
| `messages` | array | For the first run, exactly two messages: one `user`, one `assistant`. |
| `reviewer_notes` | string | Empty string is allowed; reviewers should fill this when they change status. |

## Task Types

Use these first so evaluation stays organized:

- `greeting`
- `translation-en-rw`
- `translation-rw-en`
- `grammar-explanation`
- `sentence-correction`
- `vocabulary`
- `quiz-generation`
- `dialogue`
- `uncertainty`
- `culture-register`

## Training Gate

A row can be used for `train` or `validation` only when all of these are true:

1. `review_status` is `approved`.
2. `source_status` is `approved`, `team-authored`, or `manual`.
3. The source is documented in `docs/data/source-log.md`, unless it is fully
   team-authored.
4. The row is not copied from `docs/evaluation/learning-task-bank.md` if that
   task is marked `benchmark-only`.

Rows with `split=draft` are for authoring and review only. They must not enter
a training run.

Rows with `review_status` of `needs-review`, `needs-fix`, `rejected`, or
`not-sure` must not enter a training run.

Rows with `source_status` of `blocked`, `investigate`, or `reference-only` must
not enter a training run. `reference-only` can help humans write new examples,
but the source text itself should not be converted directly into training data.

## First Seed Dataset Target

The first seed file should have at least 100 reviewed conversation examples.
That is only the validation gate for the data process, not the final SFT target.

| Category | Target Count |
| --- | ---: |
| Greeting and beginner conversation | 15 |
| English to Kinyarwanda translation | 15 |
| Kinyarwanda to English translation | 15 |
| Grammar explanation | 15 |
| Sentence correction | 15 |
| Vocabulary teaching | 10 |
| Quiz generation | 10 |
| Uncertainty / humility behavior | 5 |

Start with `split=draft` and `review_status=needs-review`. Tessy or another
fluent speaker should move rows to `approved`, `needs-fix`, `rejected`, or
`not-sure`. Approved rows can then be assigned to `train` or `validation`.

## Useful SFT Dataset Target

For a model that has a real chance of learning tutor behavior, aim for about
1,000 reviewed examples before the first serious SFT run.

The detailed scale-up plan lives in:

```text
docs/data/data-overhaul-plan.md
```

Do not reach 1,000 by copying benchmark prompts or dumping unreviewed generated
text into the dataset. The useful target only counts examples that pass source
and fluent-speaker review.

## Validation

Validate a seed file with:

```bash
python3 scripts/validate_sft_jsonl.py data/sft/seed_conversations.jsonl
```

The validator checks JSON syntax, required fields, allowed status values, unique
IDs, message structure, and the training gate.

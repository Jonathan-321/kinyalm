# Data Overhaul Plan

The project should not move from 0 reviewed SFT examples straight into model
training.

The right direction is a staged data overhaul:

1. separate training data from benchmark data,
2. expand SFT examples to roughly 1,000 reviewed rows,
3. make every source decision visible,
4. keep blocked and reference-only material out of training,
5. store large files in local/Drive/Hugging Face storage with tracked
   manifests, not directly in Git.

## Current Reality

We currently have:

- a 50-prompt tutor evaluation bank,
- a JSONL schema for SFT examples,
- approved source notes for two DigitalUmuganda datasets,
- no committed reviewed SFT conversation dataset yet.

So the next real data milestone is not "train now." It is "create the first
reviewed SFT pack."

## Dataset Tiers

| Tier | Data Type | Can Train? | Use |
| --- | --- | --- | --- |
| Tier 0 | External benchmarks | no | Held-out evaluation only. |
| Tier 1 | Team-authored tutor conversations | yes, after review | Main SFT source. |
| Tier 2 | Approved source-derived examples | yes, after source and language review | Add scale and variety. |
| Tier 3 | Reference-only learning material | no direct training | Humans use it to write original examples. |
| Tier 4 | Blocked / unclear sources | no | Do not use. |

## SFT Scale Targets

| Gate | Reviewed Examples | Purpose |
| --- | ---: | --- |
| Seed | 100 | Validate schema, review flow, and trainer input. |
| Foundation | 300 | First tiny QLoRA run worth interpreting. |
| Useful | 1,000 | First serious SFT run for a tutor demo. |
| Stretch | 3,000+ | Only after review capacity and source quality are proven. |

The useful target is about 10x the first 100-example seed. It should be large
enough to teach behavior patterns without pretending we have a broad
instruction dataset.

## 1,000 Example Mix

| Category | Target Count |
| --- | ---: |
| Greetings and beginner conversation | 80 |
| Dialogue tutoring | 120 |
| Vocabulary teaching | 140 |
| Grammar explanation | 160 |
| Sentence correction | 160 |
| English to Kinyarwanda translation | 100 |
| Kinyarwanda to English translation | 100 |
| Quiz generation | 80 |
| Culture and register | 30 |
| Uncertainty / safety behavior | 30 |

Every category should include short, beginner-friendly answers. The goal is a
useful tutor, not a generic chatbot.

## Source Overhaul Checklist

For every source in `docs/data/source-log.md`, Tessy or the backup reviewer
should add:

- license or permission status,
- whether it can be trained on,
- whether it can be converted into prompt/answer pairs,
- whether it is human-reference-only,
- whether examples can be quoted in the report,
- domain and bias notes,
- privacy concerns,
- attribution text if required.

## Example Authoring Rules

Do:

- write original examples where possible,
- keep prompts short,
- keep answers direct,
- mark uncertain Kinyarwanda as `needs-review`,
- include reviewer notes when something is corrected.

Do not:

- copy benchmark prompts into training,
- copy blocked source text,
- use WhatsApp/chat data without privacy review,
- approve generated Kinyarwanda without fluent-speaker review,
- use loss/perplexity as proof that the tutor is useful.

## Work Allocation

| Owner | Work |
| --- | --- |
| Jonathan | Draft schema-compliant examples, maintain benchmark separation, update task board. |
| Tessy | Review language quality, source permissions, and SFT-use decisions. |
| Bonheur | Check model/tokenizer implications, base model fit, and training constraints. |

## Immediate Next Step

Create a local draft:

```text
data/sft/seed_conversations.jsonl
```

Start with 100 rows marked `needs-review`. After Tessy reviews them, split
approved rows into train and validation sets.

Large drafts and source files should live outside Git. Use the shared storage
workflow in:

```text
docs/data/shared-storage-workflow.md
```

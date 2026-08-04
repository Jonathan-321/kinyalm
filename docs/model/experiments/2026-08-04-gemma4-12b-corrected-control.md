# Gemma 4 12B Corrected-Objective Control

Run date: 2026-08-04

## Outcome

The corrected one-epoch QLoRA control completed and produced a reproducible
adapter, but it did not pass the model-quality gate. The training pipeline now
supervises the intended assistant answer tokens and validation improved
steadily. In the exact BF16 base-versus-adapter check, however, the adapter
still made factual and language errors and entered a 768-token repetition loop
on one of ten held-out tasks. Keep this checkpoint as experimental evidence;
do not deploy or present it as the finished KinyaLM tutor.

## Objective correction

The first corrected smoke test exposed a Gemma 4 chat-template mismatch. With
`add_generation_prompt=True`, the tokenizer inserts four hidden thought-channel
tokens that are absent from a completed assistant turn. TRL's conversational
prompt/completion alignment therefore masked the first four real answer tokens
on every training example.

Commit `b303cdc8d555683d8cf0d8684677b697baec04af` fixes the boundary without
hard-coding token IDs. For each training example, the script renders the full
conversation and the same conversation with an empty final assistant turn. The
longest common token prefix is the exact assistant-content boundary; the
remaining tokens receive the completion mask. The corrected one-step smoke
test completed without the previous tokenization warnings.

## Reproducible setup

| Field | Value |
| --- | --- |
| Run ID | `gemma4-12b-corrected-control-20260804T052447Z` |
| Training code | `b303cdc8d555683d8cf0d8684677b697baec04af` |
| Evaluation code | `782d111040701a9e8402fc921499079b88f8c625` |
| Base model | `google/gemma-4-12B-it` |
| Base revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Dataset | `kinyalm/kinyalm-data-lake` |
| Dataset revision | `754a58b021cfe1e505f432df0de45ce2f63a3b21` |
| Data tier | `experimental-critic-filtered` |
| Human reviewed | No |
| Split | 776 train + 87 validation conversations |
| Supervised turns | 1,395 train + 144 validation assistant turns |
| GPU | Lambda 1x A100 40 GB SXM4 |
| Quantization | 4-bit NF4 with BF16 computation |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Schedule | 1 epoch, 175 optimizer steps, learning rate `5e-5` |
| Sequence/batch | 1,024 tokens, batch 1, gradient accumulation 8 |
| Seed | 42 |

## Training result

| Step | Validation loss | Validation token accuracy |
| ---: | ---: | ---: |
| 25 | 2.0725 | 0.6263 |
| 50 | 1.6287 | 0.6689 |
| 75 | 1.4862 | 0.6843 |
| 100 | 1.4213 | 0.6936 |
| 125 | 1.3889 | 0.6968 |
| 150 | 1.3764 | 0.6987 |
| 175 | 1.3750 | 0.6993 |

Training took 1,413 seconds, about 23 minutes 33 seconds. Final training loss
was `1.8385`. These numbers confirm that optimization worked on the frozen
critic-filtered split; they do not establish tutoring correctness or natural
Kinyarwanda.

Private adapter repository:
<https://huggingface.co/kinyalm/kinyalm-gemma-4-12b-corrected-control>

Published adapter revision: `fb911c9b842c2e49cfb1705cde28be0637bf8d68`

## Exact held-out comparison

The unchanged base and corrected PEFT adapter were loaded through Transformers
in BF16 against the same ten tasks, system prompt, greedy decoding, and
768-token budget. The evaluator uses `AutoTokenizer` for this text-only path;
commit `782d111` removes the unnecessary `AutoProcessor` and `torchvision`
dependency. The base raw file retains the ten failed records from the earlier
processor import attempt followed by the ten successful rerun records. The
blind review sheet includes only successful generations.

| Candidate | Successful tasks | Token-limit failures | Mean output tokens |
| --- | ---: | ---: | ---: |
| Unchanged Gemma 4 12B base | 10/10 | 0/10 | 379.2 |
| Corrected QLoRA adapter | 10/10 | 1/10 | 170.6 |

The adapter was much shorter on average, but concision was not a quality win by
itself. On task `T010`, a vocabulary lesson about people at school, it confused
the plural of `umwarimu` with `abanyamuziki`, repeatedly tried to correct
itself, and reached the full 768-token limit. It also made questionable claims
about greeting forms and produced unnatural examples on other tasks. Both
candidates correctly translated the elementary tasks `T016` and `T018`, and
both generally distinguished `Muraho` from `Mwaramutse` on `T003`. Native
speakers still need to score every row before an aggregate preference result is
reported.

## Evidence

Durable evaluation artifacts are in
[`assets/2026-08-04-gemma4-corrected-control/`](assets/2026-08-04-gemma4-corrected-control/):

- `base.jsonl` and `adapter.jsonl`: raw, revision-pinned generations;
- `base-run-manifest.json` and `adapter-run-manifest.json`: environment and
  candidate metadata;
- `blind-review.csv`: randomized reviewer sheet;
- `parity-review-manifest.json`: input hashes and task identities.

The private label key remains outside Git until scoring is complete. Reviewers
should receive only `blind-review.csv`.

## Decision

This run is an infrastructure and training-objective success, but a model
quality failure. More steps on the same critic-only data would strengthen an
unproven target and may deepen the observed mistakes. The next controlled run
must use native-speaker-approved examples targeted at these failures, followed
by the same blind base-versus-adapter gate. Preference optimization should wait
until the project has a stable SFT checkpoint and calibrated chosen/rejected
pairs.

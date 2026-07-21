# SFT Run Plan

This is the first training-readiness plan for a Kinyarwanda tutor SFT run.

> **Status update, 2026-07-21:** this document preserves the original planning
> assumptions. The Qwen fallback has since been trained and rejected for
> language quality. Base-model selection now requires the zero-shot bake-off in
> the [Track 2 experiment report](experiments/2026-07-20-track2-baseline-report.md).

SFT means supervised fine-tuning: we adapt a base model using reviewed
conversation examples so it learns the tutor behavior we want.

## Current Status

The team is not ready to launch a real fine-tune yet.

Ready:

- approved source logging has started,
- the SFT JSONL schema exists,
- the held-out tutor benchmark has 50 prompts,
- open Kinyarwanda benchmark candidates are listed,
- OSCER smoke-test scripts exist locally.

Not ready:

- no approved SFT conversation dataset exists yet,
- no 1,000-example reviewed SFT pack exists yet,
- the base model recommendation below still needs team sign-off,
- OSCER login is still blocked by authentication,
- benchmark answers still need fluent-speaker scoring instructions reviewed.

## Base Model Decision

Bonheur owns the final recommendation. The first candidate should be:

- instruction-tuned,
- multilingual,
- permissively usable for a class demo,
- small enough for QLoRA on one accessible GPU,
- documented with a clear license and model card,
- able to follow English instructions while producing or explaining
  Kinyarwanda.

Do not choose a model only because it is popular. Choose the smallest model that
can run reliably and produce reviewable outputs.

Decision table (Bonheur, 2026-07-12; needs team sign-off):

| Candidate | Size | License | Context Length | Why It Might Work | Risk | Decision |
| --- | ---: | --- | ---: | --- | --- | --- |
| google/gemma-2-9b-it | 9B | Gemma Terms of Use (fine-tuning and class demo allowed) | 8k | Tied-best tokenizer on our confirmed examples; strong record on African-language fine-tunes; instruction-tuned, so it ships a chat template that a ~100-example seed run cannot teach from scratch. | Gated repo (needs an HF token with accepted access); Gemma-2 needs `eager` attention during training. | recommended |
| google/gemma-2-2b-it | 2B | Gemma Terms of Use | 8k | Same tokenizer and family as the primary, so falling back invalidates none of the data prep or tokenizer analysis; fits much smaller GPUs. | Weaker base quality. | fallback if OOM/cost |
| Qwen/Qwen2.5-7B-Instruct | 7B | Apache-2.0 | 32k | Clean license and ungated access made it useful for proving the infrastructure. | The completed QLoRA run failed basic Kinyarwanda tutoring, translation, correction, and repetition checks. | tested and rejected |
| CohereLabs/aya-expanse-8b | 8B | CC-BY-NC-4.0 | 8k | Multilingual instruction model. | Its 23 languages do not include Kinyarwanda; non-commercial license; gated. Aya-101 was evaluated on Kinyarwanda by IrokoBench, but its own official 101-language list also does not include Kinyarwanda. | rejected |

### Tokenizer Evidence

`scripts/compare_tokenizers.py` runs candidate tokenizers over the 37
confirmed examples in `docs/tokenizer/eval-examples.tsv` (tokenizer files
only, CPU-friendly). Results, 2026-07-12:

| Tokenizer | Vocab size | Tokens/word (all) | Tokens/word (morphology) |
| --- | ---: | ---: | ---: |
| CohereForAI/aya-101 (reference only) | 250k | 2.77 | 2.94 |
| google/gemma-2-9b | 256k | 3.09 | 3.06 |
| CohereLabs/aya-expanse-8b | 255k | 3.11 | 3.00 |
| Qwen/Qwen2.5-7B | 152k | 3.28 | 3.12 |

What the numbers mean:

- No candidate tokenizer respects Kinyarwanda morphology. All of them split
  `umwarimu` as `um|war|imu`; noun-class prefixes and verb affixes do not
  survive as pieces.
- Every candidate costs roughly 3 tokens per Kinyarwanda word, so tokenizer
  compactness does not separate the finalists. The decision rests on language
  exposure, license, and fallback consistency instead.
- Aya-101 reaches 2.77 tokens per word, but tokenizer compactness is not proof
  of language coverage or generation quality. Its official language list does
  not include Kinyarwanda. Re-run the script to regenerate the full piece-level
  report.

## Dataset Inputs

Required before the first real run:

- `docs/data/source-log.md` records source permission.
- `docs/data/sft-data-schema.md` defines the row format.
- `data/sft/seed_conversations.jsonl` has at least 100 rows for the seed gate.
- The serious SFT target is about 1,000 reviewed rows, following
  `docs/data/data-overhaul-plan.md`.
- Every train/validation row has `review_status=approved`.
- Benchmark prompts in `docs/evaluation/learning-task-bank.md` are not copied
  into training data.

Recommended first split:

| Split | Count | Rule |
| --- | ---: | --- |
| train | 80 | reviewed and approved seed rows |
| validation | 20 | reviewed and approved seed rows |
| benchmark-only | 50 | lives in task bank, never in training |

Recommended serious split:

| Split | Count | Rule |
| --- | ---: | --- |
| train | 900 | reviewed and approved |
| validation | 100 | reviewed and approved |
| external benchmarks | separate | never copied into training |

## QLoRA Starting Settings

QLoRA means training small adapter weights while keeping the base model mostly
frozen and quantized. It is the practical path when GPU memory is limited.

Initial settings to try:

| Setting | First Value | Fallback If OOM |
| --- | --- | --- |
| precision | 4-bit quantization | same |
| LoRA rank | 16 | 8 |
| LoRA alpha | 32 | 16 |
| LoRA dropout | 0.05 | 0.05 |
| max sequence length | 1024 | 512 |
| per-device batch size | 1 | 1 |
| gradient accumulation | 8 | 16 |
| epochs | 2 | 1 smoke epoch |
| learning rate | 2e-4 | 1e-4 |
| warmup ratio | 0.03 | 0.03 |

These are starting points, not proof of quality.

Gemma-2 specific: train with attention implementation `eager`. Gemma-2 uses
logit soft-capping that is incompatible with flash/sdpa attention during
training and silently degrades results.

## Smoke Test

Before any real fine-tune:

1. Validate the JSONL file:

   ```bash
   python3 scripts/validate_sft_jsonl.py data/sft/seed_conversations.jsonl
   ```

2. Run the project check:

   ```bash
   python3 scripts/check_project.py
   ```

3. On OSCER, run the CPU smoke job:

   ```bash
   sbatch scripts/hpc/oscer_project_smoke.sbatch
   ```

4. On OSCER, run the GPU smoke job:

   ```bash
   sbatch scripts/hpc/oscer_gpu_smoke.sbatch
   ```

The smoke test is successful when the job starts, imports the needed Python
packages, sees CUDA in the GPU job, and writes logs to `slurm_logs/`.

## First Real Run

The first real run should be intentionally small:

- train on the approved seed dataset,
- save adapter checkpoints,
- record exact base model and dataset commit,
- run at least 10 held-out benchmark prompts after training,
- send outputs to fluent-speaker review.

Success means:

- the training job completes,
- validation loss does not explode,
- the adapter can be loaded for inference,
- outputs are better organized or more tutor-like than the base model,
- failure modes are specific enough to guide the next data pass.

The first serious run should wait until the 1,000-example target is available
or until the team explicitly decides that the class schedule requires a smaller
demo run.

Failure is still useful if it produces a clear blocker:

- model too large for available GPU,
- dataset too small or too noisy,
- Kinyarwanda quality not good enough,
- source permissions not ready,
- infrastructure login or dependency issue.

## Week 3 Target

By the next training checkpoint, the team should have one of these:

1. A completed tiny QLoRA run on reviewed seed conversations.
2. A written blocker with the exact missing input: model choice, data approval,
   reviewer availability, or GPU access.

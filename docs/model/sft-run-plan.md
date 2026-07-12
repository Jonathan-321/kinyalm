# SFT Run Plan

This is the first training-readiness plan for a Kinyarwanda tutor SFT run.

SFT means supervised fine-tuning: we adapt a base model using reviewed
conversation examples so it learns the tutor behavior we want.

## Current Status

The team is not ready to launch a real fine-tune yet.

Ready:

- approved source logging has started,
- the SFT JSONL schema exists,
- the held-out tutor benchmark has 50 prompts,
- OSCER smoke-test scripts exist locally,
- KILM has shown that approved Kinyarwanda text can support measurable language
  modeling progress.

Not ready:

- no approved SFT conversation dataset exists yet,
- no final base model has been selected,
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

Decision table to fill before training:

| Candidate | Size | License | Context Length | Why It Might Work | Risk | Decision |
| --- | ---: | --- | ---: | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | pending |

## Dataset Inputs

Required before the first real run:

- `docs/data/source-log.md` records source permission.
- `docs/data/sft-data-schema.md` defines the row format.
- `data/sft/seed_conversations.jsonl` has at least 100 rows.
- Every train/validation row has `review_status=approved`.
- Benchmark prompts in `docs/evaluation/learning-task-bank.md` are not copied
  into training data.

Recommended first split:

| Split | Count | Rule |
| --- | ---: | --- |
| train | 80 | reviewed and approved |
| validation | 20 | reviewed and approved |
| benchmark-only | 50 | lives in task bank, never in training |

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

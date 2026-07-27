# Local 12B Fine-Tuning Notes (Tessy)

Owner: Tessy Mugisha
Machine: MacBook Air, Apple M2, 24 GB unified memory (verified 2026-07-22)

## Goal

Attempt a local LoRA fine-tune of the Gemma 4 12B model on our reviewed SFT
data, and record whether a 24 GB Apple-silicon machine can do it — so the team
knows what belongs on a laptop vs. the cloud GPU.

Model: `mlx-community/gemma-4-12B-it-qat-4bit` (4-bit MLX, ~12 GB to *run*).
Data: `data/mlx-data/` (512 train / 49 validation, from my reviewed slice).

## Running it (inference / demo) — works

The browser demo loads and runs the 12B fine at 4-bit:

```bash
bash scripts/local/chat_gemma4_web.sh --open --port 8091
```

Peak memory for inference is ~12 GB, comfortably within 24 GB.

## Fine-tuning it locally — the attempt

Conservative settings to try to fit training into 24 GB (LoRA on few layers,
short sequences, gradient checkpointing):

```bash
source .venv/bin/activate
mlx_lm.lora --model mlx-community/gemma-4-12B-it-qat-4bit --train \
  --data data/mlx-data --iters 200 --batch-size 1 --num-layers 4 \
  --max-seq-length 512 --grad-checkpoint \
  --adapter-path outputs/gemma4-12b-tessy-lora
```

If it hits out-of-memory: lower `--num-layers` (try 2) and `--max-seq-length`
(try 256) before giving up. Close the demo and other apps first — inference and
training will fight over the same 24 GB.

### Result

- Date: 2026-07-27
- Outcome: [x] blocked before training — model download failed
- What happened: Training never started. Fetching the 12B weights (~11 GB,
  10 files) crawled at ~1.2 MB/s and one shard failed at 97% with a Hugging
  Face Xet transfer error: `CAS Client Error ... HTTP status 403 Forbidden`
  (`us.aws.cdn.hf.co/xorbs/...`). This is a download/CDN issue, not an
  out-of-memory error — the model never loaded.
- Retry plan: `hf auth login`, accept Gemma terms, then
  `export HF_HUB_DISABLE_XET=1 && hf download mlx-community/gemma-4-12B-it-qat-4bit`
  (resumable) to pre-cache before training.
- Second blocker (after fixing the download with `HF_HUB_DISABLE_XET=1`):
  `mlx_lm.lora` refused to load the model — `ValueError: Model type
  gemma4_unified not supported` / `No module named 'mlx_lm.models.gemma4_unified'`.
  Gemma 4 12B is a unified vision+text checkpoint, and stock mlx_lm 0.31.3 has
  no model class for it. The repo's demo runs it only via a custom text-only
  shim (`TextOnlyGemma4` in `scripts/run_multilingual_bakeoff.py`) for
  inference — there is no training path for this model type locally.
- Tried `pip install -U mlx-lm`: already on 0.31.3 (latest), so no newer
  version adds `gemma4_unified` support. Confirmed dead end for local 12B
  training.
- Takeaway: local 12B fine-tuning is blocked on two fronts (slow/awkward
  download AND no mlx_lm training support for `gemma4_unified`), independent of
  memory. The 12B fine-tune belongs on the cloud GPU (transformers path, 40 GB
  A100). The laptop's role: run/demo the 12B for inference, and fine-tune the
  fully-supported 2B locally (done separately, with a before/after probe).
- 2B fine-tune was run and evaluated: see
  [`local-2b-finetune-result.md`](local-2b-finetune-result.md). Pipeline works,
  but the 2B overfits/degrades — reinforcing that the 12B on cloud is the real
  quality path.

## Cloud path (the real 12B run)

Training a 12B comfortably needs a datacenter GPU. Our team's platform is
**Lambda Cloud** (1× A100 40 GB), with RunPod as a backup (~$1–2/hour).

- Runbook: [`docs/model/lambda-baseline-run.md`](lambda-baseline-run.md)
- Scripts: `scripts/cloud/bootstrap_lambda_instance.sh`,
  `scripts/cloud/submit_lambda_job.sh`
- The data is already in the data-lake (`kinyalm/kinyalm-data-lake`), so a cloud
  run can pull it directly — no re-upload needed.

Bonheur owns the "rent GPU / train in cloud" action item, so the 12B fine-tune
should run there rather than each of us renting separately. My reviewed +
SFT-ready data feeds that run.

## Takeaway

- Laptop (24 GB M2): great for **running/demoing** the 12B and for **fine-tuning
  the smaller 2B** locally.
- Cloud (A100 40 GB): required for **fine-tuning the 12B**.

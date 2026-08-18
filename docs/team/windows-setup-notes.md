# Windows Setup Notes (Track 2 Fine-Tuning)

Contributor: Tessy

Every existing runbook assumes an Apple-silicon Mac. This note records the
Windows path that was verified end-to-end for the free, pre-GPU steps:
Hugging Face auth, dataset build, preflight, and model-metadata check. Training
itself still runs on the cloud A100, not on Windows.

## What works on Windows

Run these from PowerShell at the repo root (`C:\PythonProjects\kinyalm`). They
need no GPU and cost nothing.

### 1. Authenticate to Hugging Face

hf auth login


Paste a token with read access to the gated `kinyalm/kinyalm-data-lake` and,
for publishing later, write access to the `kinyalm` org. Confirm identity and
org access with:

hf auth whoami


Gemma 4 is Apache-2.0 and ungated, but the data lake is gated, so the token is
still required.

### 2. Build the pinned experimental dataset

Single line (PowerShell does not accept the bash `\` line continuations in the
Mac runbooks):

python scripts/prepare_hf_sft_baseline.py --repo-id kinyalm/kinyalm-data-lake --revision 754a58b021cfe1e505f432df0de45ce2f63a3b21 --mode critic-accepted --output-dir outputs/hf-baseline-a --acknowledge-experimental


Expected: `Selected rows: 863`, split `776 train / 87 validation`.

### 3. Dry-run preflight (no model download)

python scripts/train_qlora.py --model google/gemma-4-12B-it --model-revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --train-file outputs/hf-baseline-a/train.jsonl --eval-file outputs/hf-baseline-a/validation.jsonl --dataset-manifest outputs/hf-baseline-a/dataset-manifest.json --output-dir outputs/runs/baseline-a-corrected --experimental --dry-run


Then open `outputs/runs/baseline-a-corrected/run-preflight.json` and confirm:

- `"loss_scope": "assistant-completions-only"`
- train `supervised_assistant_turns` is 1395, validation is 144

Both flags being present confirm the corrected assistant-only objective is
active.

### 4. Verify Gemma 4 access before paying for a GPU

Add `--verify-model-metadata` to the same command. It downloads only the
tokenizer and config, no weights:

python scripts/train_qlora.py --model google/gemma-4-12B-it --model-revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --train-file outputs/hf-baseline-a/train.jsonl --eval-file outputs/hf-baseline-a/validation.jsonl --dataset-manifest outputs/hf-baseline-a/dataset-manifest.json --output-dir outputs/runs/baseline-a-corrected --experimental --verify-model-metadata --dry-run


A pass prints `model_type=gemma4_unified` and
`Gemma4UnifiedForConditionalGeneration`. If access is not set up, this fails
here for free instead of on the paid instance.

Helpful equivalents for Mac commands in the runbooks:

- `grep` becomes `findstr /n`
- `cat` becomes `type`
- forward-slash paths still work in Python args; use back-slashes only for
  native PowerShell paths.

## What does NOT work on Windows

- The Lambda submit scripts (`scripts/cloud/submit_lambda_job.sh`,
  `run_lambda_baseline.sh`) are bash and assume the team's Mac SSH keys. They do
  not run in PowerShell. Use WSL or Git Bash, or SSH into the instance and run
  `train_qlora.py` directly.
- The local MLX inference and adapter-parity checks are Apple-silicon only and
  cannot run on Windows at all. Gate 1 of the recovery plan (PEFT-vs-MLX parity)
  needs a Mac owner.
- Local 12B training does not work on any laptop; it belongs on the A100.

## Summary

On Windows you can do all data prep, validation, and access checks locally, then
run the actual fine-tune on the cloud A100. The Mac-only parts are local MLX
inference and the parity check, which need a teammate with Apple silicon.
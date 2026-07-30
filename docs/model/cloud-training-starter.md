# Gemma 4 Cloud Smoke-Test Starter (Tessy)

A plain-language checklist for running the KinyaLM fine-tune on a rented GPU,
since my MacBook can't train the big models locally. Based on the team's
existing Lambda runbook (`lambda-baseline-run.md`) — my Mac just drives it.

## The idea in one line

Rent a Linux machine with a big GPU, connect to it from my Mac's terminal, run
the training there (it pulls our data from Hugging Face), download the result,
then shut the machine off so it stops charging.

## Current scope

- Target model: **Gemma 4 12B** (`google/gemma-4-12B-it`) at the pinned
  revision in `scripts/cloud/run_lambda_baseline.sh`.
- A `gemma4` profile has been added to `scripts/cloud/run_lambda_baseline.sh`,
  but it is restricted to a one-step smoke test until the GPU result is
  reviewed.
- Gemma 4 is public under Apache-2.0. The Hugging Face token is still required
  for the restricted KinyaLM data lake and output repository.
- The current script materializes 863 critic-filtered examples. That is useful
  for infrastructure testing but is not the final human-approved training tier.

## What I need ready

- A Lambda Cloud account with a payment method.
- An SSH key added to Lambda (the runbook references `~/.ssh/coolify_key`).
- A Hugging Face token that can read the KinyaLM data lake and publish to the
  `kinyalm` organization. Separate least-privilege tokens are also supported.

## Steps

**1. Preflight — free, runs on my Mac, catches setup mistakes before paying:**
```bash
PREFLIGHT_ONLY=1 MODEL_PROFILE=gemma4 RUN_ROOT=/tmp/kinyalm-lambda-preflight \
  bash scripts/cloud/run_lambda_baseline.sh
```

The preflight now downloads the tokenizer and model config, but not the model
weights. This catches the Transformers/tokenizer blocker before paid compute.

### Verification already completed

- The pinned Gemma 4 preflight resolves Transformers `5.14.1`,
  `GemmaTokenizer`, and `Gemma4UnifiedForConditionalGeneration` at the expected
  model commit.
- The pinned data revision deterministically produces 776 training and 87
  validation conversations.
- A one-step local run with the small pinned
  `HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint at
  `12fd25f77366fa6b3b4b768ec3050bf629380bac` completed the shared
  Transformers/TRL/PEFT training path and saved a LoRA adapter.

This does not replace the Gemma 4 GPU smoke. The 12B model weights and
4-bit CUDA path have not passed until step 3 completes on the A100.

**2. Launch the GPU** in the Lambda console: 1x A100 40 GB. Check the current
price in the console, wait until it shows Active, and copy its IP address.

**3. One-step smoke test** (confirms the machine is wired up before any
full run):
```bash
MODEL_PROFILE=gemma4 MAX_STEPS=1 \
  bash scripts/cloud/submit_lambda_job.sh <INSTANCE-IP> <GIT-REF>
```

**4. Review the smoke artifacts.** A successful one-step run must load the
quantized model, attach LoRA adapters, complete one optimizer step, save the
adapter, and publish the run evidence.

**5. Watch it run:**
```bash
ssh -i ~/.ssh/coolify_key ubuntu@<INSTANCE-IP> 'tail -f ~/kinyalm-bootstrap.log'
```
The run writes `RUNNING` → `COMPLETED` (or `FAILED`) under
`~/kinyalm-runs/<RUN-ID>/status`. A `COMPLETED` run has already uploaded its
adapter to Hugging Face.

**6. STOP THE BILLING — most important step.** As soon as the artifacts are on
Hugging Face, **terminate the instance in the Lambda console.** Do NOT rely on
`shutdown`/`poweroff` — Lambda keeps charging if the instance is only halted.
Storage is ephemeral, so make sure the upload finished before terminating.

## Full experimental run

After the one-step GPU smoke succeeds and this training-stack change is merged,
run the full critic-filtered experiment:

```bash
MODEL_PROFILE=gemma4 ALLOW_EXPERIMENTAL_FULL_RUN=1 \
  bash scripts/cloud/submit_lambda_job.sh <INSTANCE-IP> main
```

This trains on the current deterministic split: 776 train and 87 validation
conversations. The result remains explicitly experimental because the rows
have model-critic review rather than complete fluent-human approval.

Use validation loss and the saved `samples.jsonl` as first-pass training
evidence. Native-speaker and base-versus-adapter evaluation is a separate
follow-up after reproducible training succeeds. The script requires
`ALLOW_EXPERIMENTAL_FULL_RUN=1` so a paid full run cannot start accidentally.

## Reminder

Never put a token on the command line or in Git. `submit_lambda_job.sh`
streams it securely and deletes it after. Tokens go through `hf auth login`.

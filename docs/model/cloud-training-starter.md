# Cloud Training Starter (Tessy)

A plain-language checklist for running the KinyaLM fine-tune on a rented GPU,
since my MacBook can't train the big models locally. Based on the team's
existing Lambda runbook (`lambda-baseline-run.md`) — my Mac just drives it.

## The idea in one line

Rent a Linux machine with a big GPU, connect to it from my Mac's terminal, run
the training there (it pulls our data from Hugging Face), download the result,
then shut the machine off so it stops charging.

## Plan (confirmed with the team)

- **My own Lambda account** pays (individual accounts, not shared).
- Target model: **Gemma 4 12B** (`google/gemma-4-12B-it`). The 9B/qwen profiles
  were prior experiments.
- **Bake-off:** everyone fine-tunes the 12B on their own and we compare whose
  performs best. So each person runs a full run and publishes to their OWN
  output repo (I use `OUTPUT_REPO=kinyalm/kinyalm-gemma-4-12b-tessy`).
- A `gemma4` profile has been added to `scripts/cloud/run_lambda_baseline.sh`,
  so the run targets the 12B. Gemma is gated: accept Google's license on the HF
  model page and use a token with gated-model read access.

## What I need ready

- A Lambda Cloud account with a payment method.
- An SSH key added to Lambda (the runbook references `~/.ssh/coolify_key`).
- Two Hugging Face tokens via `hf auth login` / `hf auth list`:
  one that can read the gated model, one that can publish to the `kinyalm` org.

## Steps

**1. Preflight — free, runs on my Mac, catches setup mistakes before paying:**
```bash
PREFLIGHT_ONLY=1 MODEL_PROFILE=gemma4 RUN_ROOT=/tmp/kinyalm-lambda-preflight \
  bash scripts/cloud/run_lambda_baseline.sh
```

**2. Launch the GPU** in the Lambda console: 1x A100 40 GB, region us-east-1
(~$1.99/hour). Wait until it shows Active, and copy its IP address.

**3. One-step smoke test** (cheap — confirms the machine is wired up before the
full run):
```bash
MAX_STEPS=1 bash scripts/cloud/submit_lambda_job.sh <INSTANCE-IP> <GIT-REF>
```

**4. Full training run — Gemma 4 12B, publishing to my own bake-off repo:**
```bash
HF_MODEL_TOKEN_NAME=<gated-token-name> MODEL_PROFILE=gemma4 \
  OUTPUT_REPO=kinyalm/kinyalm-gemma-4-12b-tessy \
  bash scripts/cloud/submit_lambda_job.sh <INSTANCE-IP> <GIT-REF>
```

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

## Reminder

I never put a token on the command line or in Git — `submit_lambda_job.sh`
streams it securely and deletes it after. Tokens go through `hf auth login`.

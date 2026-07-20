# Lambda Cloud Track 2 Baseline

This runbook executes the HF-only Baseline A on one Lambda Cloud GPU and
publishes the adapter plus its audit trail to a private Hugging Face model repo.

## Pinned First Run

- GPU: 1x NVIDIA A100 40 GB SXM4.
- Region: Virginia (`us-east-1`).
- Current price: $1.99 per hour, plus applicable tax.
- Image: Lambda Stack 22.04.
- Preferred model: `google/gemma-2-9b-it` at
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`, after gated access is enabled.
- Ungated fallback: `Qwen/Qwen2.5-7B-Instruct` at
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Data: `kinyalm/kinyalm-data-lake` at
  `754a58b021cfe1e505f432df0de45ce2f63a3b21`.
- Rows: 776 experimental train and 87 experimental validation.
- Output: a private model-specific repo under the `kinyalm` organization.

Gemma requires two access gates: the individual HF user must acknowledge
Google's license, and the download token must have gated-model read access.
The submitter can use separate least-privilege tokens for model download and
KinyaLM organization publishing. Keep the Qwen profile as the fallback until
both Gemma checks pass.

## Submit From This Mac

Validate the pinned profile locally before launching paid compute:

```bash
PREFLIGHT_ONLY=1 RUN_ROOT=/tmp/kinyalm-lambda-preflight \
  bash scripts/cloud/run_lambda_baseline.sh
```

Set `MODEL_PROFILE=gemma` to validate or submit Gemma after access is enabled.
The default remains `qwen` so an unattended job cannot spend GPU time failing
on a gated download.

Launch the instance in the Lambda console with the SSH key already named
`codex-mac-coolify-20260706`. After the instance becomes active, submit the
full run from the repository checkout:

```bash
bash scripts/cloud/submit_lambda_job.sh <INSTANCE-IP> <GIT-REF>
```

The submitter streams the cached fine-grained HF token into a mode-600 remote
file, starts the job under `nohup`, and removes the token file when the job
finishes. It never writes the token to Git or the training log.

Run a one-step infrastructure smoke before the full job when the environment
has changed:

```bash
MAX_STEPS=1 bash scripts/cloud/submit_lambda_job.sh <INSTANCE-IP> <GIT-REF>
```

For the preferred Gemma run:

```bash
HF_MODEL_TOKEN_NAME=<LOCAL-GATED-TOKEN-NAME> MODEL_PROFILE=gemma \
  bash scripts/cloud/submit_lambda_job.sh \
  <INSTANCE-IP> <GIT-REF>
```

`HF_MODEL_TOKEN_NAME` and optional `HF_PUBLISH_TOKEN_NAME` refer to names shown
by `hf auth list`; the token values are never command-line arguments or logs.

Follow the remote log:

```bash
ssh -i ~/.ssh/coolify_key ubuntu@<INSTANCE-IP> \
  'tail -f ~/kinyalm-bootstrap.log'
```

The run writes `RUNNING`, `COMPLETED`, or `FAILED` under
`~/kinyalm-runs/<RUN-ID>/status`. A completed run is uploaded before the status
changes to `COMPLETED`.

## Billing Safety

Terminate the instance in Lambda Cloud immediately after artifacts are on HF.
Do not use `shutdown`, `poweroff`, or an OS halt as a substitute: Lambda states
that billing continues when those commands leave the instance in Alert status.
This first run uses ephemeral instance storage, so a failed upload must be
retrieved before termination.

## What This Result Means

This is an infrastructure and learning baseline, not a released model. The 863
selected examples passed the model critic but have not completed fluent-human
review. Compare the adapter with the unchanged base model by task family before
using its failures to prioritize the next review and data pass.

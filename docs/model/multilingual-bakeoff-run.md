# Multilingual Base-Model Bake-Off

This runbook compares unchanged Gemma 4 12B and 31B models on the same 26
held-out KinyaLM tutor prompts. It does not fine-tune either model. The purpose
is to find out whether either base already has enough Kinyarwanda knowledge and
general reasoning to justify an SFT run.

## Pinned Experiment

The complete machine-readable experiment is in
[`configs/evaluation/gemma4_bakeoff.json`](../../configs/evaluation/gemma4_bakeoff.json).

| Candidate | Revision | Download size |
| --- | --- | ---: |
| `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | 22.3 GiB |
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 58.3 GiB |

Both models receive the same system prompt, deterministic decoding, disabled
thinking mode, seed, token limit, and task order. Every response is written to
JSONL immediately so the run can resume after an interruption.

## Local 12B Screening Run

The 12B candidate can be screened on a 32 GB Apple-silicon Mac before renting
cloud compute. The local path pins
`mlx-community/gemma-4-12B-it-qat-4bit` at revision
`e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` and MLX-LM `0.31.3`. Its download
is about 10.3 GB. The weights are a mixed 4/8-bit MLX conversion derived from
the QAT release, so this run is a practical quality gate, not a numerically
identical substitute for the official BF16 checkpoint. The config also pins
the `gemma4` MLX model-type override required for the conversion's upstream
`gemma4_unified` label. It strictly loads every text-model tensor and ignores
only the pinned `vision_embedder.*` tensors that the text-only runner cannot
consume. The local decoder also suppresses token IDs `258882` and `258883`, as
required by the checkpoint's pinned `generation_config.json`, so image/audio
end markers cannot replace ordinary text tokens.

Run one held-out prompt first. This creates a separate smoke-test directory:

```bash
RUN_ID=gemma4-12b-mlx-smoke \
  bash scripts/local/run_gemma4_12b_bakeoff.sh --limit 1
```

Then run all 26 held-out prompts:

```bash
bash scripts/local/run_gemma4_12b_bakeoff.sh
```

After the benchmark process exits, test the same pinned checkpoint yourself in
a multi-turn terminal chat:

```bash
bash scripts/local/chat_gemma4_12b.sh
```

Use `/reset` to clear conversation context and `/quit` to unload the model.
The client identifies this as an unchanged Gemma screening checkpoint, not a
fine-tuned KinyaLM release.

The first command creates an isolated environment under
`~/.cache/kinyalm/gemma4-12b-bakeoff` and downloads the model into the normal
Hugging Face cache. Later runs reuse both. Full outputs are written under
`outputs/model-bakeoffs/<RUN-ID>/`; the 26-row `review/blind-review.csv` hides
the checkpoint identity while reviewers score the answers. The launcher also
writes `summary/automatic-metrics.json` with objective runtime measurements,
truncation flags, and leaked modality-token flags. It does not assign language
quality scores.

Do not run the 31B QAT conversion on this 32 GB Mac. Its weights alone occupy
about 26.9 GB, leaving too little headroom for the runtime and KV cache. Keep
the 31B comparison on the cloud BF16 path below.

The source bank is
[`docs/evaluation/learning-task-bank.md`](../evaluation/learning-task-bank.md).
The runner selects only the 26 rows marked `benchmark-only`; it refuses any
configuration that requests a training split.

## Evaluation Scope

This is the first gate, not the complete multilingual evaluation. The current
held-out bank mostly uses English instructions about Kinyarwanda. It tests
translation, grammar, vocabulary, tutoring, register, and uncertainty, but it
does not yet adequately cover Kinyarwanda-origin prompts, mixed-language
requests, or long multi-turn consistency. Add those as a separate held-out bank
before making a final release claim.

## No-Download Validation

No model is downloaded by this command:

```bash
uv run python scripts/run_multilingual_bakeoff.py \
  --config configs/evaluation/gemma4_bakeoff.json \
  --output-dir /tmp/kinyalm-gemma4-bakeoff \
  --dry-run
```

The output must list 26 tasks and both pinned BF16 candidates. To validate the
local mapping without downloading weights, add `--backend mlx`; it should list
only the pinned 12B MLX runtime.

## Cloud Requirements

- At least 76,000 MiB aggregate NVIDIA GPU memory. Use one 80 GB GPU or a
  multi-GPU instance with at least that total capacity.
- At least 120 GiB free instance storage.
- Lambda Stack with CUDA-enabled PyTorch visible to `python3`.
- A cached Hugging Face token that can write to
  `kinyalm/kinyalm-data-lake`.

Do not launch the previous 40 GB A100 profile for this run. The 31B BF16 model
does not fit there. The cloud script checks GPU memory and disk before model
downloads begin.

## Submit From This Mac

Launch a suitable instance in Lambda Cloud with the existing project SSH key.
After its status becomes active, submit the branch or commit containing this
runner:

```bash
bash scripts/cloud/submit_multilingual_bakeoff.sh \
  <INSTANCE-IP> <GIT-REF>
```

Follow progress:

```bash
ssh -i ~/.ssh/coolify_key ubuntu@<INSTANCE-IP> \
  'tail -f ~/kinyalm-bakeoff-bootstrap.log'
```

The run loads 12B first, releases its GPU memory, and then loads 31B. It records
raw output, latency, token counts, throughput, peak GPU memory, package versions,
model revisions, config hash, task-bank hash, and hardware information.

## Artifacts And Review

Completed or partial artifacts are uploaded to:

```text
kinyalm/kinyalm-data-lake/evaluation/model-bakeoffs/<RUN-ID>/
```

Reviewers should receive only `review/blind-review.csv`. It contains 52 rows,
one answer from each model for every held-out task, with randomized `Model A`
and `Model B` labels. The label key is deliberately excluded from Hugging Face.

Retrieve the complete run, including the private key, before terminating the
instance:

```bash
bash scripts/cloud/fetch_multilingual_bakeoff.sh \
  <INSTANCE-IP> <RUN-ID> outputs/model-bakeoffs
```

Do not give reviewers `private/blind-key.json` or ask them to inspect the raw
model directories until scoring is finished.

## Billing And Decision Gate

Terminate the Lambda instance after artifacts are uploaded and fetched. An OS
shutdown is not a billing stop.

Advance a candidate only after fluent reviewers confirm that it answers the
request directly, uses natural Kinyarwanda, avoids invented grammar, and does
not collapse into canned refusals. If neither candidate clears that gate, do
not spend money fine-tuning them; expand the unchanged bake-off or test a small
continued-pretraining pilot instead.

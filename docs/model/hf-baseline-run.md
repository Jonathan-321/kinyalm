# HF-Only Track 2 Baseline

Hugging Face is the sole data input for Track 2 training. The shared review
sheet can remain the team's collaboration surface, but a training job consumes
only versioned JSONL and manifests from `kinyalm/kinyalm-data-lake`.

## What We Can Run Now

The `sft-distillation-production-1000-v3-final` batch contains 1,000 examples:

- 863 critic-accepted rows.
- 137 critic-repair recommendations.
- 0 critic rejects.
- 1,000 rows still awaiting fluent-human review.

The source manifest sets `can_train=false`. We preserve that decision for the
curated model and create a separate `experimental-critic-filtered` tier for
baseline runs.

Run A uses only the 863 critic-accepted rows. Run B may use accepted rows plus
the critic's complete revisions for the 137 repair rows. Run A should happen
first because it has the narrower quality assumption.

## Build A Pinned Dataset

```bash
python3 scripts/prepare_hf_sft_baseline.py \
  --repo-id kinyalm/kinyalm-data-lake \
  --revision main \
  --mode critic-accepted \
  --output-dir outputs/hf-baseline-a \
  --acknowledge-experimental
```

The builder resolves `main` to an immutable Hub commit, downloads the drafts
and critic artifacts from that commit, validates all joins, removes exact
conversation duplicates, keeps duplicate prompts or responses on one side of
the split, and writes `train.jsonl`, `validation.jsonl`, and
`dataset-manifest.json`.

## Preflight And Launch

Install the training dependencies:

```bash
uv sync --extra train
```

Validate the complete run without downloading a model:

```bash
python3 scripts/train_qlora.py \
  --train-file outputs/hf-baseline-a/train.jsonl \
  --eval-file outputs/hf-baseline-a/validation.jsonl \
  --dataset-manifest outputs/hf-baseline-a/dataset-manifest.json \
  --output-dir outputs/runs/baseline-a \
  --experimental \
  --dry-run
```

Remove `--dry-run` on the GPU machine. Pin `--model-revision` to the selected
model commit before a result is compared or reported.

For a one-step infrastructure smoke run, replace `--dry-run` with
`--max-steps 1`. The resulting adapter is only proof that the complete training
path works; it is not an evaluation candidate.

## Iteration Loop

1. Train the critic-accepted baseline.
2. Evaluate the unchanged base model and adapter on the same held-out prompts.
3. Tag failures by task family, language mix, difficulty, and error type.
4. Prioritize human review and new examples for the largest failure clusters.
5. Build a new immutable HF dataset version and rerun the same configuration.
6. Compare quality by category, not only aggregate validation loss.

Human-approved rows eventually become ordinary `train` and `validation` data.
They must never be silently mixed with the experimental tier.

## How LIMA Applies

[LIMA](https://arxiv.org/abs/2305.11206) showed strong alignment from 1,000
carefully curated examples on a 65B pretrained model. It supports prioritizing
quality and diversity over raw row count. It does not establish that any 1,000
synthetic examples, or a smaller base model with limited Kinyarwanda exposure,
will produce the same result. Our baseline therefore tests the hypothesis; the
human-curated run is the result that can properly follow the LIMA strategy.

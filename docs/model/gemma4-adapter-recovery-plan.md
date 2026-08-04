# Gemma 4 Adapter Recovery Plan

## Decision

Reject `kinyalm/kinyalm-gemma-4-12b-experimental` revision
`feefb1e7ac359b60ca45af9db8fd883af8cac933` as a demo checkpoint. Do not add
training steps to it. The checkpoint completed optimization, but both the
original CUDA samples and the local native-speaker session show severe factual,
grammatical, instruction-following, and repetition failures.

## What the failure tells us

| Finding | Evidence | Interpretation |
| --- | --- | --- |
| The adapter is worse than the base | The live session repeatedly produced `Ndakora imyitozo y'ubyandikwa` for unrelated prompts | This is response collapse, not ordinary verbosity |
| The failure existed before MLX | The 12 samples generated on the Lambda A100 include a long `yari yari yari...` loop and several wrong answers | Retraining and data quality must be investigated before blaming local inference |
| The repeated sentence was not copied from the dataset | It appears zero times in the frozen 863 conversations, system prompt, and project code | The model synthesized a bad attractor instead of memorizing one exact row |
| The MLX tensor conversion is exact | All 656 converted tensors equal the PEFT tensors after the required transpose; maximum absolute difference is `0.0` | The converter is not the primary cause, although the different local quantized base still needs an output-parity check |
| The first run used the wrong loss scope | TRL received a conversational `messages` dataset with its default `assistant_only_loss=False` | Loss was computed on user and assistant tokens instead of only desired answers |
| The data was not a curated LIMA-style set | The pinned manifest says `human_reviewed=false`, `production_eligible=false`, and source `can_train=false` | An automated critic is useful for triage, but it is not fluent-speaker approval |
| The update was aggressive for the experiment size | Two epochs at `2e-4`, rank 16, across all attention and MLP projections | This can over-steer a strong base before generation quality is checked |

The frozen set is not dominated by exact duplication: it has 863 unique
conversations, 1,539 assistant turns, zero duplicate conversations, and zero
duplicate assistant messages. The median assistant turn is 29 words. The main
data risk is semantic correctness and synthetic template concentration, not a
single copied sentence.

## Trainer correction

The training script now expands every conversation into conversational
prompt/completion examples. Each assistant turn is a completion, while all
earlier turns remain prompt context. TRL is explicitly configured with
`completion_only_loss=True`.

The corrected pinned split produces:

| Split | Conversations | Supervised assistant turns |
| --- | ---: | ---: |
| Experimental train | 776 | 1,395 |
| Experimental validation | 87 | 144 |

The new conservative starting configuration is one epoch, learning rate
`5e-5`, rank 16, alpha 32, dropout 0.05, and evaluation plus checkpointing every
25 optimizer steps. This is an ablation starting point, not a guaranteed final
recipe.

## Controlled recovery sequence

### Gate 1: implementation parity

On one A100, run 10 identical prompts through:

1. the unchanged Hugging Face base;
2. the original PEFT adapter loaded directly by Transformers;
3. the same adapter converted to MLX on the local checkpoint.

Use the same chat template, system prompt, greedy decoding, and token budget.
If the direct PEFT and MLX answers disagree materially, investigate base
checkpoint compatibility before another training run.

The runner now supports adapter variants without changing the held-out task
bank or decoding configuration. Run the first ten tasks on CUDA for the base:

```bash
uv run python scripts/run_multilingual_bakeoff.py \
  --backend transformers \
  --candidate gemma4-12b-it \
  --limit 10 \
  --output-dir outputs/adapter-parity/base
```

Then run the original PEFT adapter directly against the same pinned base:

```bash
uv run python scripts/run_multilingual_bakeoff.py \
  --backend transformers \
  --candidate gemma4-12b-it \
  --adapter kinyalm/kinyalm-gemma-4-12b-experimental \
  --adapter-revision feefb1e7ac359b60ca45af9db8fd883af8cac933 \
  --run-as original-peft \
  --limit 10 \
  --output-dir outputs/adapter-parity/peft
```

Run the converted adapter on the Mac with the same task subset:

```bash
OUTPUT_DIR=outputs/adapter-parity/mlx \
  bash scripts/local/run_gemma4_12b_bakeoff.sh \
  --adapter ~/.cache/kinyalm/gemma4-12b-experimental-adapter/adapter-mlx \
  --adapter-revision feefb1e7ac359b60ca45af9db8fd883af8cac933 \
  --run-as original-mlx \
  --limit 10
```

After all three raw files are on one machine, build a randomized review pack:

```bash
uv run python scripts/build_adapter_parity_review.py \
  --limit 10 \
  --result outputs/adapter-parity/base/raw/gemma4-12b-it.jsonl \
  --result outputs/adapter-parity/peft/raw/original-peft.jsonl \
  --result outputs/adapter-parity/mlx/raw/original-mlx.jsonl \
  --output-dir outputs/adapter-parity/review
```

Share only `blind-review.csv` during scoring. `blind-key.json` contains model
and adapter identities and must remain private until the review is complete.

The direct PEFT run from 2026-08-03 already reproduced catastrophic repetition
and elementary meaning errors on a separate 30-prompt set. The local MLX arm
completed on 2026-08-04 and failed 2 of 10 held-out tasks through catastrophic
repetition while making additional meaning errors. The unchanged MLX base
completed the same 10 tasks with zero token-limit loops and preserved the two
elementary translations reversed by the adapter. See
[`experiments/2026-08-03-gemma4-base-vs-adapter-eval.md`](experiments/2026-08-03-gemma4-base-vs-adapter-eval.md)
and
[`experiments/2026-08-04-gemma4-12b-mlx-adapter-heldout.md`](experiments/2026-08-04-gemma4-12b-mlx-adapter-heldout.md).
Exact same-prompt BF16-versus-PEFT-versus-MLX parity remains pending, but it is
no longer required before the corrected-objective control.

### Gate 2: corrected-objective control

Train one experimental control on the same immutable 863-row split using the
corrected assistant-only objective, one epoch, and `5e-5`. This isolates the
training-objective change. Keep it labeled experimental because the source is
still critic-only.

The Gemma 4 Lambda profile publishes this control to
`kinyalm/kinyalm-gemma-4-12b-corrected-control`. It must not reuse the rejected
adapter repository.

Save checkpoints every 25 steps. Evaluate each checkpoint on the same held-out
task bank so a late regression can be discarded instead of accepted merely
because final validation loss is lower.

### Gate 3: curated-data run

Freeze a native-speaker-approved split and repeat the same configuration. Start
with the approved rows already available; scale toward 1,000 only through
reviewed additions targeted at observed failures. Do not automatically promote
the 137 critic repairs or the rest of the synthetic pool.

The comparison must include the unchanged base, the corrected-objective
control, and the curated adapter. Blind model labels and randomize answer order.

### Gate 4: preference optimization only if needed

After a stable SFT adapter exists, convert reviewer corrections into
chosen/rejected pairs. DPO is a reasonable later experiment. RLHF, RLAIF, and
RLVR are not the next step because the project does not yet have a calibrated
preference set or an automatic verifier for Kinyarwanda naturalness.

## Acceptance criteria

- Zero catastrophic repetition loops on the held-out task bank.
- No identity collapse across unrelated prompts.
- Native speakers prefer the adapter to the base overall and by task family.
- Translation, correction, conversation, and vocabulary each have enough
  scored examples to expose regressions.
- English and Kinyarwanda are scored separately to catch cross-lingual
  forgetting.
- Every result preserves the model revision, dataset revision, trainer config,
  raw outputs, reviewer sheet, and adapter hash.

## Research basis

- [LIMA](https://arxiv.org/abs/2305.11206) supports a small, carefully curated
  SFT set on a strong base; it does not validate unreviewed synthetic rows.
- [QLoRA](https://arxiv.org/abs/2305.14314) makes the adaptation memory-efficient;
  it does not make the supervision correct.
- [Self-Instruct](https://arxiv.org/abs/2212.10560) and
  [Magpie](https://arxiv.org/abs/2406.08464) support generating a large candidate
  pool followed by aggressive filtering.
- [LESS](https://arxiv.org/abs/2402.04333) supports selecting data around failed
  capabilities instead of training on every available example.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
  explains why lower token loss and usable generation must be evaluated
  separately.
- [MT-Bench](https://arxiv.org/abs/2306.05685) documents model-judge biases and
  supports retaining blinded native-speaker review as the final language gate.

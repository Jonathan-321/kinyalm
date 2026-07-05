# Compute Inventory

Owner: Bonheur Byiringiro

Goal:

```text
Know what model sizes we can run before we design training scripts.
```

## Available Machines

| Machine | Owner | CPU | RAM | GPU | VRAM | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Jonathan laptop | Jonathan | TBD | TBD | TBD | TBD | Fill in. |
| Tessy machine | Tessy | TBD | TBD | TBD | TBD | Fill in. |
| Bonheur machine | Bonheur | TBD | TBD | TBD | TBD | Fill in. |
| Lab/cloud option | TBD | TBD | TBD | TBD | TBD | Fill in if available. |

## First Config Targets

### Bronze: CPU Tiny Run

Purpose:

```text
Prove tokenizer -> batches -> model -> loss works.
```

Suggested target:

- very small corpus,
- short context length,
- small embedding size,
- 1-2 layers,
- 1-2 attention heads,
- minutes, not hours.

### Silver: Small GPU Run

Purpose:

```text
Train a small Kinyarwanda baseline once data is approved.
```

Suggested target:

- approved corpus,
- train/validation split,
- checkpointing and resume,
- loss/perplexity report,
- sample generation,
- run comparison report.

KILM now implements these mechanics on approved TTS text and has a 200-step
`small` MPS baseline. The remaining compute question is whether we can afford a
longer `small` run or a `baseline_gpu` run without losing fast iteration.

### Gold: Stretch Run

Purpose:

```text
Only if compute and data are both strong enough.
```

Do not make final success depend on this.

## Questions To Answer

1. Can every teammate run tests locally?
2. Can at least one machine run a tiny PyTorch training loop?
3. Is GPU access available?
4. Are cloud credits available?
5. What is the maximum reasonable training time for a class milestone?
6. Where will checkpoints live if they are too large for GitHub?

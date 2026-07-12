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
| OU OSCER / Sooner | Jonathan | SLURM partitions | Varies by node | Public GPU partitions available | Varies by GPU | See [oscer-quickstart.md](oscer-quickstart.md). Login is currently blocked by authentication, not by project setup. |
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

KILM now implements these mechanics on approved TTS and MT text. The MT
`baseline_gpu` MPS run is the current best result: validation perplexity moved
from 43.7940 to 21.0469 during the 10,000-step continuation. The remaining
compute question is whether another larger run is worth it before tokenizer
performance and data cleanup improve.

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

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
| Tessy machine | Tessy | Apple M2 (MacBook Air, 2022) | 24 GB unified | Apple M2 integrated (MPS/Metal only, no CUDA) | Shares 24 GB unified memory | Verified 2026-07-22; ~900 GB free disk; macOS 26.5.2. Can LoRA-tune gemma-2-2b-it locally via mlx-lm; 9B is 4-bit inference/testing only. Note: `scripts/train_qlora.py` falls back to fp32 CPU here (4-bit is CUDA-only), so use mlx-lm or the SmolLM2 CPU smoke test locally. |
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

For the tutor path, the remaining compute question is not just "can we launch a
job?" It is whether we can run a small QLoRA experiment after the data is
reviewed. The next useful compute proof is:

- JSONL validation passes,
- CPU and GPU smoke jobs start,
- the chosen base model fits on available hardware,
- a tiny run writes checkpoints and logs without exhausting memory.

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

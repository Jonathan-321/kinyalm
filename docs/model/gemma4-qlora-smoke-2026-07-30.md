# Gemma 4 12B QLoRA Smoke - 2026-07-30

## Result

The Gemma 4 CUDA/QLoRA infrastructure gate passed on one Lambda A100 40 GB.
This was an infrastructure smoke, not a model-quality experiment.

- Project commit: `f8b138d28d9581c30fcf25ae77a675a7965a0599`
- Model: `google/gemma-4-12B-it`
- Model revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- Dataset revision: `754a58b021cfe1e505f432df0de45ce2f63a3b21`
- Split: 776 experimental training / 87 experimental validation conversations
- Runtime: Lambda 1x A100 40 GB SXM4, BF16 compute, 4-bit NF4 weights
- Transformers: `5.14.1`
- TRL: `1.9.2`
- PEFT: `0.20.0`
- Adapter size: approximately 126 MB

The run successfully:

1. Recreated the pinned data split.
2. Resolved `GemmaTokenizer` and
   `Gemma4UnifiedForConditionalGeneration`.
3. Loaded the 12B checkpoint through the 4-bit CUDA path.
4. Attached LoRA adapters to the configured attention and MLP projections.
5. Completed forward and backward passes plus one optimizer step.
6. Evaluated all 87 validation rows.
7. Saved the adapter and provenance.

## Smoke Metrics

- One training step: 15.21 seconds
- Training loss: 7.372
- Gradient norm: 8.438
- Mean token accuracy: 0.355
- Tokens processed: 1,767

These numbers are diagnostic only. One step cannot establish Kinyarwanda
quality, and the original one-step scheduler logged a learning rate of zero
because the 3% warmup rounded up to the only available step.

## Published Evidence

The private adapter repository is:

<https://huggingface.co/kinyalm/kinyalm-gemma-4-12b-experimental>

Verified Hugging Face revision:
`3e93c7a422e24427e2e19b18629b9e8aeef7f92f`

It contains the adapter, tokenizer files, preflight and dataset manifests,
system information, training log, run metadata, and `SMOKE-NOTE.md`.

## Optional Sample Interruption

The required smoke gate had already passed and the adapter was saved when the
wrapper began generating 12 optional samples. Generation inherited
`use_cache=False` from training and was taking several minutes on the first
200-token response. That optional stage was stopped to avoid unnecessary GPU
billing, so the automatic wrapper recorded exit code 143. The saved evidence
was then published manually.

The follow-up fixes are:

- one-step smokes use zero warmup so their only update has a non-zero learning
  rate;
- one-step smokes skip post-training samples;
- full runs explicitly enable the model cache during sample generation and
  restore the training setting afterward;
- sample rows are flushed as they finish, making progress observable.

The Lambda instance was terminated after publication, and the console
confirmed that billing had stopped.

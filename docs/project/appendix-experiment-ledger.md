# KinyaLM Experiment Ledger

Snapshot date: 2026-08-03

This ledger records the question, setup, result, decision, and evidence status
for every major project experiment. "Passed" always names the gate that passed;
it never implies overall model quality.

## Experiment summary

| ID | Experiment | Gate tested | Result | Final decision |
| --- | --- | --- | --- | --- |
| A0 | KILM 5.07M approved-MT sandbox | Can a small causal LM learn held-out next-token structure from the available corpus? | Optimization pass; generation still weak | Keep as the clearest Track A learning-curve result |
| A1 | KILM 109.5M corpus-scale run | Can the team build and train the larger LM pipeline? | Pipeline pass; undertrained checkpoint | Keep as systems and scaling evidence |
| B1 | Qwen2.5 7B QLoRA | Can HF data, A100 training, adapter publication, MLX conversion, and local serving work end to end? | Infrastructure pass; language fail | Reject Qwen as final base |
| B2 | Kakugo 3B local control | Does direct Kinyarwanda specialization make a better tutor? | Runtime pass; tutor-quality fail | Keep as specialized control |
| B3 | Gemma 4 12B local screen | Can the candidate run locally, and does it show promising tutor behavior? | Runtime pass; formal quality provisional fail | Keep as promising but unqualified candidate |
| B4 | Gemma 2 2B local LoRA | Can a small model fit the reviewed slice locally and improve? | Training pass; generation regressed | Do not use for final model |
| B5 | Gemma 4 12B A100 QLoRA smoke | Does the new CUDA/tokenizer/model path complete a real update and save artifacts? | Infrastructure pass | Full reviewed-data run still required |
| B6 | Gemma 4 12B experimental QLoRA | Does the complete two-epoch recipe optimize and publish on the frozen critic-filtered split? | Full optimization pass; language quality unscored | Compare blindly before making a quality claim |
| D1 | Gemma 4 MLX browser demo | Can the base be tested interactively with streaming and feedback? | Prototype pass | Use for research/demo, not as final adapter evidence |

## A0: KILM 5.07M approved-MT sandbox

### Question

Could a deliberately small decoder-only Transformer learn the next-token
distribution of the available Kinyarwanda corpus well enough to show a clear
held-out learning signal?

### Setup

| Field | Value |
| --- | --- |
| Repository | <https://github.com/Jonathan-321/kilm> |
| Model | 5,067,264-parameter decoder-only causal LM |
| Architecture | 6 layers, 8 heads, hidden size 256, context 256 |
| Tokenizer | BPE, vocab 512, 414 merges; fit on train and validation text |
| Corpus | 764,213 train tokens + 15,420 validation tokens |
| First run | 2,000 steps; batch 8; cosine schedule; 100 warmup steps |
| Continuation | 10,000 additional steps; batch 8; constant `5e-5`; no warmup |
| Hardware | Apple MPS |
| Seed | 1337 |

### Result

- The first run reduced validation perplexity from 599.48 to 42.13 in 229.09
  seconds.
- The continuation reduced validation perplexity from 43.79 to 21.05 in
  1,105.52 seconds.
- Across both stages, the model processed 24,576,000 token positions, or about
  4.85 token positions per parameter.
- The dense-training `6ND` proxy is about 0.747 PFLOP across the 12,000 steps.
- Samples became more sentence-like, but remained grammatically and
  semantically unreliable.

### Interpretation

This was KILM's strongest next-token-prediction result: the same held-out
objective improved consistently across a resumed run. It proves that the model,
tokenizer, optimizer, checkpoint resume path, and corpus carried a useful
learning signal. It does **not** prove conversational ability. Perplexity tests
how well the model predicts the next held-out token; tutoring quality requires
task prompts and fluent-speaker review.

The experiment also exposed a data-governance limitation. Its recorded source
was later blocked from current KinyaLM training because the upstream
redistribution chain was incomplete. Retain the result as research evidence,
not as the foundation for a public model release.

## A1: KILM 109.5M corpus-scale baseline

### Question

Can the team reproduce the mechanics of language-model development: corpus
assembly, tokenizer training, model configuration, next-token training,
evaluation, checkpointing, and generation?

### Setup

| Field | Value |
| --- | --- |
| Repository | <https://github.com/Jonathan-321/kilm> |
| Model | 109,529,856-parameter LLaMA-style causal LM |
| Architecture | 12 layers, 12 heads, hidden 768, intermediate 2,048 |
| Tokenizer | SentencePiece BPE, vocab 32,000, byte fallback |
| Context | 1,024 tokens |
| Corpus | 34,697,216 tokenized tokens total |
| Run | 2,000 optimizer steps, batch 1, 1,024 tokens per step |
| Optimizer | AdamW, learning rate `3e-4`, weight decay 0.1 |
| Schedule | 2,000 warmup steps followed by intended cosine decay |
| Hardware | Apple MPS, no mixed precision |
| Seed | 1337 |

### Result

- 2,048,000 training tokens processed.
- About 1.35 PFLOP by the `6ND` dense-training proxy.
- 33.11 minutes of training and 3.12 minutes of evaluation.
- Train loss 6.819; validation loss 5.864.
- Train perplexity 915.04; validation perplexity 352.13.
- Greedy outputs repeated the same phrases; sampled outputs were longer but
  incoherent.

### Interpretation

The larger architecture, 32K SentencePiece tokenizer, multi-source corpus,
training loop, evaluation, and generation path all worked. The checkpoint was
not scaled enough to become useful: the 2,000-step run saw only 0.0187 token
positions per parameter, and its 2,000-step warmup consumed the entire run, so
the intended decay phase never happened.

The main Track A challenge was therefore broader than "2.05M tokens was too
small." A useful bilingual model from scratch would require a jointly designed
Kinyarwanda-English tokenizer, a licensed and balanced multilingual corpus,
enough model capacity, and a much larger training budget. A smaller model can
learn substantially from the available data, as A0 demonstrated, but scaling
parameters without scaling data and optimization produced a worse checkpoint.

### Evidence caveat

The corpus included Digital Umuganda MT text. A later audit blocked that source
because its upstream Flickr30k ShareAlike lineage and attribution chain were
not fully represented. Present this as a learning run, not a clean model
release.

## B1: Qwen2.5 7B full experimental QLoRA

### Question

Can a Qwen2.5-7B-Instruct base, adapted on the 863 critic-accepted candidate
conversations, become a useful Kinyarwanda tutor while proving the end-to-end
cloud and local-serving pipeline?

### Setup

| Field | Value |
| --- | --- |
| Run ID | `qwen25-7b-baseline-a-20260720T223011Z` |
| Base revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Dataset revision | `754a58b021cfe1e505f432df0de45ce2f63a3b21` |
| Data | 776 train + 87 validation; critic-filtered, not human reviewed |
| GPU | Lambda 1x A100 40 GB |
| Method | 4-bit NF4 QLoRA; rank 16; alpha 32; dropout 0.05 |
| Schedule | 2 epochs; 194 optimizer steps; `2e-4`; seed 42 |
| Sequence/batch | 1,024 tokens; batch 1; gradient accumulation 8 |
| GPU training time | 515 seconds |
| Adapter | 80,792,880 bytes |

### Result

- Final training loss: 1.6971.
- Final evaluation loss: 1.4928.
- Dataset preparation, A100 training, adapter publication, local MLX
  conversion, and local OpenAI-compatible serving all worked.
- The first local server dropped the default adapter path. That integration bug
  was fixed and regression-tested; post-fix outputs were confirmed to use the
  adapter.
- The adapted model still produced wrong definitions, incorrect grammar,
  meaning-changing translations, semantic drift, and repetition loops.

### Decision

Reject Qwen as the final base. Keep the run as a negative baseline and as proof
that infrastructure success and lower loss are not language-quality evidence.

### Cost boundary

At the recorded `$1.99/hour` instance rate, 515 seconds equals about `$0.285`
of training time. The total instance bill was not preserved, so this is only a
lower bound.

## B2: Kakugo 3B unchanged local control

### Question

Does an openly released model with direct Kinyarwanda adaptation outperform
the general multilingual candidates on the project's tutoring tasks?

### Setup

| Field | Value |
| --- | --- |
| Model | `ptrdvn/kakugo-3B-kin` |
| Runtime | Local BF16 |
| Probe set | Six high-signal tutor prompts |
| Decoding | Model-card deterministic settings; repetition penalty 1.05 |

### Result

The model showed direct Kinyarwanda behavior but gave an incorrect definition
of `ubupfura`, malformed vocabulary and tense explanations, an unrelated
translation containing a Chinese character, noun-class analysis errors, and
unwanted `<think>` traces.

### Decision

Keep Kakugo as the strongest directly adapted open control found, but do not
select it as the tutor base without blind native-speaker evidence.

## B3: Gemma 4 12B unchanged local screen

### Question

Can the 12B candidate run on the team's Mac, and does its unchanged behavior
justify spending data and GPU time on fine-tuning?

### Setup

| Field | Value |
| --- | --- |
| Run ID | `gemma4-12b-mlx-20260721-local-v2` |
| Source revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| MLX revision | `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` |
| Runtime | MLX-LM 0.31.3; deterministic greedy decoding |
| Quantization | QAT-derived mixed 4/8-bit affine conversion |
| Tasks | 26 held-out tutor prompts |
| Output budget | 768 tokens |
| Thinking | Disabled |

### Result

- 26 successful generations; 0 generation errors.
- 10,986 output tokens; median 408 per response.
- Total 46.92 minutes; median 104.57 seconds per response.
- Median 3.99 tokens/second.
- 11.37 GB peak unified memory.
- 22 normal stops and 4 length truncations.
- The corrected `local-v2` run had zero forbidden control-token leaks.
- Saved responses contain objective translation reversals, invented grammar and
  morphology, wrong noun classes, and malformed beginner dialogue.
- The blind review sheet has 26 rows and zero completed reviewer scores.

### Decision

Runtime gate passed. The user-facing interactive impression was stronger than
Qwen, Kakugo, or Gemma 2, so Gemma 4 remains the leading practical candidate.
The formal evidence does not yet support calling it a validated Kinyarwanda
tutor.

## B4: Gemma 2 2B local LoRA

### Question

Can the locally supported 2B checkpoint fit Tessy's reviewed data and improve
before/after tutor responses?

### Setup

| Field | Value |
| --- | --- |
| Owner | Tessy Mugisha |
| Machine | MacBook Air, Apple M2, 24 GB |
| Base | `mlx-community/gemma-2-2b-it-4bit` |
| Data | 512 train + 49 validation pairs from the legacy Tessy export |
| Run | 300 iterations; batch 1; 8 adapted layers; max sequence 512 |
| Trainable parameters | 3.2M, or 0.12% |
| Peak memory | About 3 GB |

### Result

- Validation loss fell from 5.48 to 2.77 to 2.61.
- Generated answers became more repetitive and less useful.
- The adapter, exact environment lock, log, and before/after JSONL outputs were
  not committed.

### Decision

Treat the run as contributor-recorded exploratory evidence. It shows that
optimization can improve loss while generation regresses, but it is not a
fully reproducible final comparison.

## B5: Gemma 4 12B A100 QLoRA smoke

### Question

Can the Gemma 4 unified architecture and tokenizer complete the actual
CUDA/4-bit/LoRA path on an A100 after the dependency fixes?

### Setup

| Field | Value |
| --- | --- |
| Project commit | `f8b138d28d9581c30fcf25ae77a675a7965a0599` |
| Model revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Dataset revision | `754a58b021cfe1e505f432df0de45ce2f63a3b21` |
| Data | 776 experimental train + 87 experimental validation |
| GPU | Lambda 1x A100 40 GB SXM4 |
| Runtime | Transformers 5.14.1; TRL 1.9.2; PEFT 0.20.0 |
| Method | BF16 compute; 4-bit NF4 base; LoRA adapters |

### Result

- Model/tokenizer loaded through `Gemma4UnifiedForConditionalGeneration`.
- LoRA attached to configured attention and MLP projections.
- Forward pass, backward pass, and one optimizer step completed.
- All 87 validation rows were evaluated.
- Adapter and provenance were saved and published.
- One step: 15.21 seconds; loss 7.372; gradient norm 8.438; mean token
  accuracy 0.355; 1,767 tokens.
- Adapter size: approximately 126 MB.
- The original one-step warmup rounded the learning rate to zero. Follow-up
  code now uses zero warmup for one-step smokes and skips optional samples.
- The optional sample stage was manually stopped after the required gate passed
  to avoid additional billing. The instance was terminated.

### Decision

The Gemma 4 infrastructure blocker is cleared. None of the one-step metrics
measure Kinyarwanda quality. A full run on the frozen human-reviewed dataset is
still required.

## B6: Gemma 4 12B full experimental QLoRA

### Question

Can the complete Gemma 4 QLoRA recipe run for two epochs on the frozen
critic-filtered 1,000-candidate experiment, publish reproducible artifacts, and
produce an adapter ready for a controlled base-versus-adapter review?

### Setup

| Field | Value |
| --- | --- |
| Run ID | `gemma4-12b-experimental-20260803T024131Z` |
| Project commit | `d1d7a48e118c5ae08dd6f5f962eee2704b3cc460` |
| Base | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Dataset | `kinyalm/kinyalm-data-lake` at `754a58b021cfe1e505f432df0de45ce2f63a3b21` |
| Data | 776 train + 87 validation; critic-accepted, not human-reviewed |
| GPU | Lambda 1x A100 40 GB SXM4 |
| Method | 4-bit NF4 base, BF16 compute, LoRA rank 16, alpha 32, dropout 0.05 |
| Schedule | 2 epochs, 194 optimizer steps, `2e-4`, warmup ratio 0.03, seed 42 |
| Sequence/batch | 1,024 tokens; batch 1; gradient accumulation 8 |

### Result

- Training completed in 1,572 seconds, or 26.2 minutes.
- Final training loss was 1.6264.
- Final validation loss was 1.3250 and mean validation token accuracy was
  0.6943 across all 87 validation rows.
- The trainer recorded 306,840 processed tokens and produced 12 sample
  generations.
- The adapter is 131,235,784 bytes, about 126 MiB.
- At the recorded `$1.99/hour` rate, training time alone cost approximately
  `$0.87`. Provisioning, downloads, evaluation, upload, and idle time are not
  included in that lower bound.
- The private adapter and provenance bundle were published at
  <https://huggingface.co/kinyalm/kinyalm-gemma-4-12b-experimental>, revision
  `feefb1e7ac359b60ca45af9db8fd883af8cac933`.
- The Lambda instance was terminated after upload; the console showed no
  running instances at 2026-08-03 03:22 UTC.
- A preliminary inspection of the 12 saved samples found serious failures in
  definition, tense correction, translation, dialogue, grammar explanation,
  ambiguity handling, and repetition control. This inspection is a warning,
  not a blinded native-speaker score.

### Decision

The full optimization and publication gate passed. This is not yet the final
KinyaLM quality result because the 863 selected conversations were approved by
an automated critic, not by fluent humans, and the generated answers have not
yet been blindly compared with the unchanged base. Loss and token accuracy
show that the adapter learned the training distribution; only held-out task
evaluation and native-speaker review can show whether it became a better
Kinyarwanda tutor.

## D1: Local MLX browser demo

### Question

Can the leading base candidate be tested repeatedly through a practical chat
experience before and after fine-tuning?

### Setup and result

- Streamed browser chat on `127.0.0.1`.
- One resident model worker to avoid MLX thread/stream errors.
- Converse, Translate/Correct, and Learn modes.
- Response budgets of 160, 192, and 256 tokens.
- Six-turn bounded history and a four-entry, 512 MB prompt cache.
- Feedback is written to a private local JSONL file for later review.
- Measured short-chat generation ranged from about 2.8 to 6.8 tokens/second.

### Decision

The interface is a working research and demo surface. It currently serves the
unchanged base checkpoint, so it must not be shown as evidence of a completed
fine-tuned KinyaLM adapter.

## Remaining controlled evaluation

The experimental adapter is now ready. The comparison should change only the
adapter, not the evaluation conditions:

1. Preserve the current base revision, tokenizer, chat template, system prompt,
   seed, and decoding settings.
2. Generate all held-out prompts with the unchanged base and the B6 adapter.
3. Blind model labels and randomize answer order.
4. Collect two native-speaker reviews per response and report agreement.
5. Report wins, ties, losses, regressions, latency, and external benchmarks.
6. Consolidate and freeze the human-approved data revision.
7. Retrain the final adapter on that human-approved split only.
8. Publish the final adapter after confirming the data and model license chain.

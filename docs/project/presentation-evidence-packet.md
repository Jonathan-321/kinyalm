# KinyaLM Presentation Evidence Packet

Snapshot date: 2026-08-03

This document is the factual source for the final presentation and appendix. It
separates completed evidence from interpretation and planned work so the team
does not accidentally present an infrastructure result as a language-quality
result.

## Executive conclusion

KinyaLM now has a complete experimental Gemma 4 QLoRA adapter and a working
research/demo foundation, but it does not yet have a native-speaker-validated
final model.

- Track A produced two useful findings. A 5.07M-parameter sandbox reduced
  held-out perplexity from about 599 to 21 and demonstrated real next-token
  learning, although its generated text was still unreliable. A later 109.5M
  run proved the larger tokenizer, corpus, model, and evaluation pipeline, but
  it was undertrained and used a warmup schedule that consumed the full run.
- Track B proved the complete QLoRA delivery path on Qwen and proved that Gemma
  4 12B can complete a two-epoch CUDA/QLoRA run, save a 126 MiB adapter,
  publish reproducible artifacts, and run locally through MLX.
- The Qwen adapter and Gemma 2 2B adapter failed the actual language-quality
  gate. Gemma 4 12B is the most promising local candidate the team tried, but
  the formal 26-prompt screen recorded multiple objective errors and has not
  received its planned blind native-speaker scoring.
- The completed Gemma 4 adapter used 863 machine-critic-accepted examples. The
  lake separately contains 460 unique human-approved conversations across
  contributor artifacts; these still need to be consolidated and frozen for
  the final human-approved run.

The defensible final claim today is: **the project has a reproducible data,
training, evaluation, publication, and local-demo pipeline, plus a complete
experimental Gemma 4 adapter; its language-quality improvement is still
pending blind native-speaker evaluation.**

## Current project snapshot

| Area | Verified status | What it means |
| --- | --- | --- |
| GitHub | Public repository; `main` at `d1d7a48`; no open PRs or issues | The latest Gemma 4 smoke fixes are merged |
| Clean `main` tests | 103 passed on 2026-08-02 | Code and schema checks pass; this is not a language-quality score |
| Larger local data workspace | 235 passed on 2026-08-02 | Additional unmerged data-audit code also passes locally |
| Hugging Face data lake | Public-gated; revision `56e09f5e84842db6f3bc730c4bc0c128585c3219`; repository license metadata `other` | The page is public, but file downloads require a signed-in accepted account; each source still needs its own rights record |
| Lake footprint | 203 files, 309 tree entries, 59.77 MiB | Text data and metadata are compact; small storage size does not imply few examples |
| Draft distillation data | 1,000 candidates | All require a human gate before production use |
| Critic result | 863 accept recommendations; 137 repair recommendations | 86.3% machine-critic acceptance; useful for triage, not human approval |
| Human-approved artifacts | 461 rows, 460 unique conversations | Approved rows exist in several contributor folders; no consolidated canonical release yet |
| Held-out task bank | 50 total tasks; 26 permanently benchmark-only | The 26 prompts must never enter training |
| Gemma 4 blind base review | 0 of 26 rows scored | Formal native-speaker quality is still pending |
| Experimental Gemma 4 SFT | Complete: 194 steps, 2 epochs, 1,572 seconds | Optimization passed on critic-filtered data; this is not a human-approved quality result |
| Experimental adapter | Private HF revision `feefb1e7ac359b60ca45af9db8fd883af8cac933` | Artifact is preserved for blind base-versus-adapter review |
| Lambda status | No running instances at 2026-08-03 03:22 UTC | Billing stopped after publication |

Links:

- GitHub: <https://github.com/Jonathan-321/kinyalm>
- Hugging Face data lake: <https://huggingface.co/datasets/kinyalm/kinyalm-data-lake>
- Experimental Gemma 4 adapter: <https://huggingface.co/kinyalm/kinyalm-gemma-4-12b-experimental>
- Qwen negative baseline adapter: <https://huggingface.co/kinyalm/kinyalm-qwen2.5-7b-track2-baseline-a>
- Track A KILM repository: <https://github.com/Jonathan-321/kilm>

## Data accounting

### Machine-generated and machine-reviewed tiers

| Tier | Rows | Human reviewed? | Production eligible? | Use |
| --- | ---: | --- | --- | --- |
| Distillation candidates | 1,000 | No | No | Review source |
| Critic accept recommendations | 863 | No | No | Experimental infrastructure runs only |
| Critic repair recommendations | 137 | No | No | Human correction queue |
| Qwen/Gemma 4 experimental split | 776 train + 87 validation | No | No | Baseline and smoke-test data |

The 13.7% critic repair rate means the critic recommended repair for 137 of
1,000 rows. It must not be described as a measured 13.7% linguistic error rate.
An LLM critic can miss errors and can also over-flag good rows.

### Human-reviewed contributor artifacts

| Artifact | Rows | Status | Important boundary |
| --- | ---: | --- | --- |
| Tessy review queue | 326 | All marked Keep by a fluent reviewer | Includes 38 rows disputed by the critic |
| Tessy critic-agreed SFT split | 258 train + 30 validation | 288 approved conversations | Current strict, reproducible Tessy release |
| Bonheur distillation review | 75 | Human-approved | One row overlaps Tessy's critic-agreed release |
| Bonheur Batch 001 promotion | 88 train + 10 validation | 98 approved | Comes from the earlier draft-batch workflow |

Across the three approved training artifacts there are 461 rows and 460 unique
conversations after removing the one shared Tessy/Bonheur calibration row. All
461 rows carry `review_status=approved`. Before training, the team still needs
one deterministic consolidation step that:

1. resolves the shared row once;
2. groups related turns to prevent train/validation leakage;
3. runs exact and near-duplicate checks;
4. verifies source and redistribution status;
5. freezes file hashes and a Hugging Face revision.

### Why the lake is only about 60 MiB

The lake stores text, JSONL, TSV, manifests, audit summaries, and small review
packages, not model weights. Its largest category is held-out benchmark and
contamination indexes (31.87 MiB), followed by generation provenance (16.81
MiB). The actual SFT data area is about 5.36 MiB. This is normal for a text data
lake; there is no need to inflate the storage footprint.

## Track A: KILM from scratch

### The experiment ladder

KILM was not one run. The small sandbox and larger baseline answer different
questions and must be presented separately.

| Run | 5.07M approved-MT sandbox | 109.5M corpus-scale baseline |
| --- | ---: | ---: |
| Architecture | 6 layers, 8 heads, hidden 256 | 12 layers, 12 heads, hidden 768 |
| Tokenizer | BPE, vocab 512 | SentencePiece BPE, vocab 32,000 |
| Context | 256 | 1,024 |
| Steps | 2,000 + 10,000 continuation | 2,000 |
| Processed token positions | 24,576,000 | 2,048,000 |
| Validation result | Perplexity 599.48 -> 21.05 | Perplexity 352.13 at end |
| Compute proxy | 0.747 PFLOP | 1.35 PFLOP |
| Interpretation | Strong held-out next-token learning; weak generation | Larger pipeline worked; checkpoint badly undertrained |

The small model's result is meaningful: its probability estimates on held-out
tokens improved substantially and continued improving after checkpoint resume.
That is narrower than conversational competence. Its samples remained
grammatically and semantically unreliable, so the team should describe this as
successful next-token learning rather than a successful Kinyarwanda chatbot.

### Larger corpus-scale run

| Item | Verified value |
| --- | ---: |
| Architecture | LLaMA-style decoder-only causal LM |
| Parameters | 109,529,856 |
| Layers / attention heads | 12 / 12 |
| Hidden / intermediate size | 768 / 2,048 |
| Context length | 1,024 tokens |
| Tokenizer | SentencePiece BPE, 32,000 vocabulary, byte fallback |
| Corpus | 22,519,811 words; 192,966 lines |
| Tokenized corpus | 33,985,536 train + 711,680 validation tokens |
| Planned run | 50,000 steps |
| Completed local run | 2,000 steps, batch 1, sequence length 1,024 |
| Actual training tokens processed | 2,048,000 |
| Optimizer / schedule | AdamW; cosine; learning rate `3e-4`; 2,000 warmup steps |
| Hardware | Apple MPS, full precision |
| Training runtime | 1,986.69 seconds, or 33.11 minutes |
| Evaluation runtime | 187.27 seconds |
| Train loss / perplexity | 6.819 / 915.04 |
| Validation loss / perplexity | 5.864 / 352.13 |

### Compute interpretation

Using the standard dense-training estimate `6 x parameters x processed tokens`,
the completed run used approximately `1.346e15` training FLOPs, or 1.35 PFLOP.
This is an algorithmic proxy, not a measurement of the Mac's electrical or
hardware utilization.

- Processed tokens per parameter: 0.0187.
- Available corpus tokens per parameter: 0.3168.
- A rough Chinchilla reference of 20 tokens per parameter would be about 2.19
  billion tokens for this model.
- The actual run processed about 1,070 times fewer tokens than that reference.
- Even one pass over the available corpus would be about 63 times below that
  reference.

The schedule also used 2,000 warmup steps for a 2,000-step run. The learning
rate therefore only reached its peak at the final step and never entered the
planned cosine-decay phase. The repetitive samples are consistent with an
extremely undertrained model and should not be presented as a surprising model
failure. This does not erase the smaller sandbox result; it shows that scaling
the architecture without scaling the data and optimization budget made the
larger checkpoint less useful.

### What a bilingual Track A would require

A Kinyarwanda-English model from scratch should not be created by adding English
after a Kinyarwanda-only model is finished. The defensible design is:

1. train one joint subword tokenizer on a controlled Kinyarwanda-English mix;
2. oversample the lower-resource Kinyarwanda corpus without duplicating it so
   aggressively that the model memorizes narrow domains;
3. pretrain one shared causal Transformer on both languages plus licensed
   parallel and code-switched text;
4. add bilingual translation, correction, tutoring, and multi-turn examples
   during SFT;
5. evaluate Kinyarwanda, English, translation directions, and code-switching
   separately;
6. if adapting an existing English model through CPT, retain English replay data
   to measure and reduce catastrophic forgetting.

The research basis for joint tokenization, multilingual transfer versus
capacity dilution, low-resource data balancing, and English replay is recorded
in `docs/project/appendix-paper-matrix.md`.

### Data-rights boundary

The KILM corpus included 121,500 cleaned words from the Digital Umuganda MT
source. A later lineage audit found direct Flickr30k ancestry and an incomplete
Attribution-ShareAlike chain. KinyaLM now blocks that source from tokenizer,
training, SFT, evaluation, and redistribution use. The KILM result remains
valid as a learning experiment, but it is not a clean redistributable model
release until the source chain is resolved or the corpus is rebuilt without
that source.

## Track B: adaptation and deployment

### What each stage means

| Stage | Plain meaning | KinyaLM status |
| --- | --- | --- |
| Pretraining (PT) | Learn general language patterns from large unlabeled text | Done only in the separate KILM learning experiment; Gemma/Qwen arrived pretrained |
| Continued pretraining (CPT) | Continue next-token training on approved target-language text | Not run |
| Supervised fine-tuning (SFT) | Train on desired user/assistant demonstrations | Full Qwen, local Gemma 2, and full experimental Gemma 4 runs completed; human-approved Gemma 4 run pending |
| Preference optimization | Learn from chosen versus rejected answers | Not run; no frozen preference dataset yet |
| RLHF / RLAIF / RLVR | Optimize a reward signal from people, models, or verifiable outcomes | Not run and not required for the first credible demo |

For this project, the immediate priority remains a clean SFT comparison. DPO is
more plausible than full RLHF after the team has enough native-speaker
preference pairs. RLVR is a poor first fit for nuanced grammar and naturalness
because those rewards are not automatically verifiable like math or code.

### Why QLoRA was chosen

QLoRA keeps the base model frozen in 4-bit form and trains small low-rank
adapter matrices in higher-precision computation. This greatly reduces GPU
memory use while preserving the pretrained model as the starting point. The
adapter is not a replacement model: at inference, the base checkpoint and the
adapter must both be loaded.

The project's main configuration is:

| Setting | Value | Role |
| --- | ---: | --- |
| Weight quantization | 4-bit NF4 | Reduces base-model memory |
| Compute dtype | BF16 on A100 | Stable tensor computation |
| LoRA rank / alpha | 16 / 32 | Adapter capacity and scaling |
| LoRA dropout | 0.05 | Regularization |
| Batch / accumulation | 1 / 8 | Effective batch of 8 sequences |
| Sequence length | 1,024 | Training token budget per example |
| Epochs | 2 | Number of data passes in the Qwen baseline |
| Learning rate | `2e-4` | Adapter update size |

These are training-run hyperparameters, not properties permanently built into
QLoRA. Their effect has to be evaluated against an unchanged base model; lower
loss alone cannot establish better Kinyarwanda.

## Model evidence and decision

| Model/run | Infrastructure | Language result | Decision |
| --- | --- | --- | --- |
| Qwen2.5 7B + QLoRA | Full HF -> A100 -> adapter -> MLX path passed | Incorrect definitions, translations, grammar, and repetition | Reject as final base; retain as negative baseline |
| Kakugo 3B unchanged | Local BF16 inference passed | Direct Kinyarwanda specialization, but serious tutor and formatting errors | Keep as specialized control |
| Gemma 2 2B + local LoRA | Training loss fell from 5.48 to 2.61 | Generated answers became repetitive and worse | Exploratory failure; not reproducible enough for final comparison |
| Gemma 4 12B unchanged | 26/26 prompts generated locally; 11.37 GB peak | Strongest hands-on impression, but saved screen contains objective translation and grammar failures | Promising candidate, not yet formally qualified |
| Gemma 4 12B QLoRA smoke | One A100 optimizer step, full validation pass, adapter save/upload passed | One step cannot measure tutor quality | Infrastructure gate passed; superseded by the full experimental run |
| Gemma 4 12B experimental QLoRA | Two epochs and 194 optimizer steps completed; adapter and provenance published | Final validation loss 1.3250 and token accuracy 0.6943; native-speaker comparison not yet scored | Optimization passed; quality decision pending |

The repository fully preserves the Qwen2.5 7B run above. Earlier ad hoc screens
of other Qwen sizes were discussed and manually tested, but they do not have
pinned revisions, complete prompt outputs, or run manifests in the current
repository. They can be mentioned as informal scouting only, not as comparable
experiments with numerical results.

Gemma 4's official card reports a 256K context window, 11.95B parameters, 48
layers, a 262K-token vocabulary, pretraining across more than 140 languages,
and out-of-box support for more than 35 languages. It does not explicitly name
Kinyarwanda among those supported languages. The project therefore has to base
its Kinyarwanda claim on its own evaluation, not the multilingual marketing
number.

## Local MLX demo

The local demo uses `mlx-community/gemma-4-12B-it-qat-4bit`, a mixed 4/8-bit
QAT-derived MLX conversion of the unchanged Gemma 4 12B instruction model.

Verified local machine:

- MacBook Pro, Apple M5, 10 CPU cores, 10 GPU cores, 32 GB unified memory.
- Checkpoint/runtime cache: roughly 10 GB on disk.
- Peak model memory: about 11.3-11.4 GB.
- Measured generation: approximately 2.8-6.8 tokens/second.
- Screen run median: 3.99 tokens/second and 104.57 seconds per 408-token answer.
- Chat modes stream responses and cap outputs at 160, 192, or 256 tokens.
- The application keeps only six recent conversation turns and caps request
  text at 36,000 characters, even though the underlying model advertises 256K
  tokens. This is a latency and memory choice, not the model's maximum context.

The system prompt lives in `src/kinyalm/demo/chat.py`. It defines Converse,
Translate/Correct, and Learn modes; language and learner-level controls; concise
response budgets; and the rules to avoid invented grammar and hidden model
identity. Prompting changes behavior but cannot repair missing language
knowledge.

## Evaluation design

The final comparison should contain four layers:

1. **Mechanical checks:** schema validity, generation errors, truncation,
   control-token leaks, latency, memory, and reproducibility.
2. **Project-owned held-out prompts:** all 26 benchmark-only tutor prompts,
   covering translation, correction, vocabulary, grammar, dialogue, ambiguity,
   register, and multi-turn behavior.
3. **External Kinyarwanda benchmarks:** use the audited, evaluation-only subsets
   such as FLORES-200, Belebele, AfriXNLI/IrokoBench, AfriQA, SIB-200,
   MasakhaPOS, and Afri-MCQA where task and license permit.
4. **Blind native-speaker review:** at least two reviewers score correctness,
   naturalness, tutoring clarity, register, hallucination, and preference in
   base-versus-adapter comparisons.

The saved Gemma 4 blind sheet has 26 responses and no completed scores. Until
that changes, the team can describe individual observed strengths and failures
but cannot report a native-speaker pass rate.

## Cloud compute and cost record

| Run | GPU work measured | Cost evidence |
| --- | ---: | --- |
| Qwen2.5 7B full experimental QLoRA | 515 training seconds | At the recorded `$1.99/hour` rate, the training-only floor is about `$0.285`; total instance cost is not preserved |
| Gemma 4 12B one-step smoke | 15.21 training seconds | At the same recorded rate, the training-step floor is about `$0.0084`; total provision/download/eval/upload time is not preserved |
| Gemma 4 12B full experimental QLoRA | 1,572 training seconds | At `$1.99/hour`, the training-only floor is about `$0.87`; the adapter was uploaded and the instance was terminated by 03:22 UTC |

These are not invoice totals. Future runs should record instance launch,
training start/end, evaluation end, upload completion, termination time, and
the provider invoice or console total. The 2026-08-03 run preserved training,
evaluation, upload, and termination evidence but not a final provider invoice.

The team-reported university cluster queue delay of about 33 hours is useful
context, but no scheduler log or screenshot is currently checked into either
repository. Treat it as a recollection until evidence is attached.

## Claims that are safe and claims that are not

| Safe now | Not safe yet |
| --- | --- |
| The 5.07M KILM sandbox reduced held-out perplexity from about 599 to 21 | That result proves fluent or conversational Kinyarwanda |
| We built and trained a 109.5M model from scratch as a learning experiment | We trained a useful 109.5M Kinyarwanda LM from scratch |
| The Qwen cloud-to-local QLoRA pipeline worked | The Qwen adapter improved Kinyarwanda quality |
| Gemma 4 12B runs locally in quantized form on a 32 GB M5 Mac | Gemma 4 is already a validated Kinyarwanda tutor |
| The Gemma 4 two-epoch experimental QLoRA run completed and published | The adapter improved Kinyarwanda tutoring quality |
| The lake contains 460 unique human-approved conversations across artifacts | We have a final reviewed 1,000-row training release |
| Synthetic generation accelerated candidate creation | LLM critic acceptance equals fluent-human approval |
| The interface is demo-ready as a base-model research prototype | The interface is serving the final fine-tuned model |

## Evidence still required before the final claim

1. Consolidate the 460 unique approved conversations into one versioned split.
2. Resolve Tessy's 38 critic-disputed rows and record the decision rule.
3. Complete blind native-speaker review of the unchanged Gemma 4 screen.
4. Evaluate the unchanged base and experimental adapter on identical held-out
   prompts.
5. Run the final Gemma 4 12B QLoRA experiment on the frozen human-approved data.
6. Report external benchmark results without mixing benchmark rows into SFT.
7. Save actual cloud runtime and invoice evidence.
8. Rebuild or qualify KILM's corpus because Digital Umuganda MT is now blocked.
9. Push the two local KILM commits (the deep-dive guide and loader hardening) so
   GitHub matches the local evidence repository.

## Recommended final narrative

The strongest presentation is not "we solved Kinyarwanda with 1,000 rows." It
is this:

> We first built language models from scratch to understand tokenization,
> architecture, scaling, and training mechanics. A 5.07M sandbox learned the
> held-out next-token distribution strongly, reducing perplexity from about 599
> to 21, while its text still failed fluent review. A later 109.5M experiment
> proved a larger corpus and training pipeline but exposed a data-to-parameter
> mismatch and a warmup error. We then moved to adaptation:
> generated candidate tutoring data, introduced machine and fluent-speaker
> review gates, tested several multilingual and Kinyarwanda-specialized bases,
> proved QLoRA on cloud GPUs, completed a two-epoch Gemma 4 experimental run,
> and built local MLX serving. The experiments also showed that falling loss and
> successful infrastructure are not enough. The quality claim now depends on a
> controlled Gemma 4 base-versus-adapter comparison and blind native-speaker
> evaluation, followed by a final run on the human-approved split.

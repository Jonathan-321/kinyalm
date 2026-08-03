# Papers and Resources Behind KinyaLM Decisions

Snapshot date: 2026-08-03

Working Drive catalog: [KinyaLM Research Paper Catalog](https://docs.google.com/document/d/1e_4mcLxR6DzX1FWvL8RSZ0kPJua3aCzbamK6phW7b6U/edit)

This is a decision matrix, not a reading list. A source belongs here only when
it explains a choice the team made, a limitation the team discovered, or a
future experiment that has a clear prerequisite.

## Evidence rule

"Back every claim with research" does not mean attaching a paper to a number
measured in this project. It means using the right evidence for each claim:

| Claim type | Required evidence | Example |
| --- | --- | --- |
| Method definition | Original paper or official technical report | QLoRA uses a frozen 4-bit base and trains LoRA adapters |
| General research interpretation | Primary empirical paper, with its scope stated | Multilingual models trade positive transfer against finite capacity |
| Project configuration | Versioned code, manifest, and checkpoint metadata | KILM used 6 layers and a 512-token BPE vocabulary |
| Project outcome | Logs, held-out outputs, and evaluation artifacts | KILM validation perplexity fell from 599.48 to 21.05 |
| Language-quality conclusion | Blinded native-speaker review plus held-out tasks | The adapter is preferred to the base for tutoring |
| Operational fact | Provider console, invoice, or timestamped run record | The A100 cost `$1.99/hour` and was terminated after upload |

Papers support why a method or interpretation is reasonable. They cannot prove
that KinyaLM's own adapter improved until the project evaluation does.

## Track A: understanding language-model mechanics

| Source | Idea used | Project decision or interpretation | Status |
| --- | --- | --- | --- |
| [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) | Build tokenization, model, optimizer, systems, scaling, data, and post-training as one accountable pipeline | Track A became a separate KILM experiment so the team could understand the machinery rather than treat a pretrained model as a black box | Used directly |
| [CS336 Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics) | Implement the tokenizer, Transformer components, optimizer, and minimal LM | Guided the starting scope and the requirement to reason about shapes, parameter counts, and next-token loss | Used as coursework reference; assignment code remains separate |
| [Karpathy, Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | Build language models progressively from basic next-token models to GPT-style systems | Supported the learning-first progression and interpretation of sample quality | Supporting resource |
| [SentencePiece](https://aclanthology.org/D18-2012/) | Train a language-independent BPE or unigram tokenizer directly from raw sentences | Supports a reproducible joint Kinyarwanda-English tokenizer rather than separate language-specific preprocessing | Used in the 109.5M KILM tokenizer; joint bilingual version is future work |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Model quality depends predictably on model size, data, and compute | Motivated explicit parameter, token, and compute accounting instead of reporting parameter count alone | Interpretive reference |
| [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) | Under a fixed pretraining budget, model size and training tokens should scale together | Used as a rough pretraining reference to explain why KILM's 109.5M parameters and 34.7M-token corpus were mismatched; not used to set SFT row counts | Used for Track A interpretation |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | Likelihood training and decoding quality are related but not interchangeable; decoding can produce bland or repetitive text from the same model | Supports reporting held-out loss/perplexity and generated samples separately instead of treating either one as the complete quality result | Used for KILM interpretation |

## Track A multilingual design: Kinyarwanda plus English

| Source | Idea used | Project decision or interpretation | Status |
| --- | --- | --- | --- |
| [XLM-R](https://arxiv.org/abs/1911.02116) | Multilingual pretraining can transfer strongly to low-resource languages, but finite model capacity creates a transfer-versus-dilution tradeoff | A bilingual KILM needs explicit language-mixture and capacity experiments; adding languages is not automatically free | Design reference |
| [mT5](https://arxiv.org/abs/2010.11934) | A shared multilingual text-to-text model can cover 101 languages; generative systems can also produce accidental translation into the wrong language | Motivates language-balanced training and separate checks for output-language control | Design and evaluation reference |
| [No Language Left Behind](https://arxiv.org/abs/2207.04672) | Low-resource multilingual systems depend on data mining, balancing, overfitting controls, human-translated evaluation, and safety checks | Supports treating Kinyarwanda data quality, translation directions, human evaluation, and governance as core model work | Design reference |
| [Breaking Language Barriers: Cross-Lingual Continual Pre-Training at Scale](https://aclanthology.org/2024.emnlp-main.441/) | Cross-lingual CPT can be more compute-efficient than training from scratch, while replay helps mitigate forgetting | Supports Track B CPT as a practical alternative if licensed Kinyarwanda text grows; it does not describe work already run | Future CPT option |
| [Emergent Abilities of Large Language Models under Continued Pre-Training for Language Adaptation](https://aclanthology.org/2025.acl-long.1547/) | Target-language CPT without English can hide early catastrophic forgetting even when validation perplexity looks acceptable | Supports retaining English in a future Kinyarwanda CPT mixture and evaluating English before and after adaptation | Future CPT safeguard |
| [Overcoming Catastrophic Forgetting in Zero-Shot Cross-Lingual Generation](https://aclanthology.org/2022.emnlp-main.630/) | Fine-tuning on one language can damage generation in another language | Supports bilingual replay and language-separated evaluation during adaptation | Future multilingual safeguard |

### Track A claim boundary

The 5.07M sandbox is direct evidence that KILM learned a held-out next-token
distribution: validation perplexity moved from 599.48 to 21.05 across the
initial and continued runs. The papers above justify why that metric must be
kept separate from open-ended generation and why a future bilingual run needs a
joint tokenizer, controlled mixture, sufficient capacity, and bilingual
evaluation. They do not turn the sandbox into evidence of fluent dialogue.

### Chinchilla boundary

Chinchilla is a pretraining scaling study. It does not prescribe the number of
supervised tutoring conversations needed for SFT, and its approximately
20-tokens-per-parameter example is a reference point rather than a universal
law for every architecture or dataset.

## Track B: adaptation and data efficiency

| Source | Idea used | Project decision or interpretation | Status |
| --- | --- | --- | --- |
| [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | A capable 65B pretrained base learned strong response behavior from 1,000 carefully curated demonstrations | Justified a controlled approximately 1,000-example SFT pilot and emphasis on diversity and quality | Used, with an important caveat |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Freeze pretrained weights and learn small low-rank update matrices | Supported adapter-based fine-tuning and portable adapter artifacts | Used directly |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | Backpropagate through a frozen 4-bit base into LoRA adapters using NF4 and memory-saving techniques | Supported the single-A100 Gemma/Qwen training design | Used directly |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | Generate instruction/response candidates and filter invalid or similar rows | Guided the generate -> deterministic gate -> critic -> human review pipeline | Used conceptually |
| [Magpie](https://arxiv.org/abs/2406.08464) | Large synthetic pools become useful through aggressive selection rather than retaining every generation | Reinforced the decision to pause scaling when critic acceptance was below the gate and to keep full provenance | Used conceptually |
| [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) | Teacher-generated rationales can provide additional supervision to smaller models | Supports future structured explanation data, but does not prove that unrestricted chain-of-thought should be stored or trained | Future/limited |
| [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333) | A small capability-targeted subset can outperform training on all available instruction data | Supports future selection around failed tutor capabilities after the first base-versus-adapter evaluation | Future; not implemented |
| [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) | Continued pretraining on in-domain unlabeled text can improve downstream adaptation | Supports a possible Kinyarwanda CPT stage after the team has a large, licensed corpus | Future; CPT not run |

### LIMA boundary

LIMA assumes a strong pretrained base that already contains most required
knowledge. It supports teaching response style and task behavior with a small,
high-quality set. It does not show that 1,000 conversations can install
Kinyarwanda knowledge into a base that lacks the language. The Qwen experiment
is direct project evidence for that limitation.

### Cloud-training boundary

Lambda was an operational choice, not a research-paper claim. QLoRA explains
how the 12B path can fit on one 40 GB A100; the provider decision came from GPU
availability, dependency control, hourly cost, and the team's cluster queue
experience. The final report should cite the run manifest and invoice evidence
for that choice rather than inventing a "cloud training paper" rationale.

## Base-model and Kinyarwanda evidence

| Source | Idea used | Project decision or interpretation | Status |
| --- | --- | --- | --- |
| [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) | 12B dense model, 48 layers, 262K vocabulary, 256K context, Apache 2.0, 140+ pretraining languages | Made Gemma 4 12B a practical local/cloud candidate with an open license and long context | Used directly |
| [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) | Documents the unified architecture, reasoning mode, multimodality, long context, and efficiency choices | Explains why the 12B checkpoint requires a Gemma 4-specific model/tokenizer/runtime path | Used for implementation context |
| [KinyaBERT: a Morphology-aware Kinyarwanda Language Model](https://aclanthology.org/2022.acl-long.367/) | Kinyarwanda's rich morphology can be handled better with explicit morphological structure than naive subword treatment | Informed tokenizer and morphology evaluation; KinyaBERT is an encoder, so it is not a drop-in generative chat base | Used for language-specific reasoning |
| [IrokoBench](https://arxiv.org/abs/2406.03368) | Human-translated African-language evaluation across inference, math, and knowledge tasks shows a large open-model gap | Supported testing open models on Kinyarwanda-specific tasks and treating Gemma 2 27B as an evidence-backed control | Used for baseline/evaluation design |
| [Belebele](https://arxiv.org/abs/2308.16884) | Multilingual reading comprehension with parallel passages across many languages | Included as an evaluation-only Kinyarwanda benchmark | Used in benchmark registry |
| [AfriQA](https://arxiv.org/abs/2305.06897) | Cross-lingual open-retrieval QA for African languages | Included as an evaluation and retrieval reference, with contamination separation | Used in benchmark registry |

### Gemma language-coverage boundary

Google states that Gemma 4 was pretrained across more than 140 languages and
supports more than 35 languages out of the box. The public card does not
explicitly identify Kinyarwanda as one of those 35. KinyaLM must establish the
claim through its own held-out and native-speaker evaluation.

## Evaluation and alignment

| Source | Idea used | Project decision or interpretation | Status |
| --- | --- | --- | --- |
| [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Strong model judges can scale evaluation but show position, verbosity, and self-enhancement biases | The project critic is a triage tool, not the approval gate; model labels should be blinded and answer order randomized | Used directly |
| [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) | Demonstrations teach desired behavior, then human preferences can train a reward-guided policy | Provides the PT -> SFT -> preference optimization framing; KinyaLM has only reached SFT experiments | Conceptual reference |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Learn from chosen/rejected responses without the full PPO reward-model loop | Best candidate for a later preference stage after native speakers create a frozen comparison dataset | Future; no preference run |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | AI feedback can generate critiques and preferences under explicit principles | Supports critic-assisted review, but reinforces the need to separate AI feedback from human linguistic authority | Conceptual reference |
| [Tulu 3](https://arxiv.org/abs/2411.15124) | Combines SFT, DPO, RLVR, decontamination, and multi-task evaluation in an open post-training recipe | Useful map of later stages; RLVR is not the first priority for subjective Kinyarwanda tutoring quality | Future reference |

### Why RL is not the next project step

- RLHF requires enough human preference comparisons and a reliable reward
  model; the project does not yet have either.
- RLAIF can scale feedback, but the same judge can reproduce synthetic
  Kinyarwanda errors.
- RLVR is strongest where correctness can be checked automatically, such as
  math or code. Naturalness, register, and grammar explanations require native
  judgment.
- The highest-value next result is still a clean SFT base-versus-adapter
  comparison. DPO can follow once reviewers create chosen/rejected pairs.

## Serving, context, and tools

| Source | Idea used | Project decision or interpretation | Status |
| --- | --- | --- | --- |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Manage KV-cache memory in pages to raise serving throughput and batch size | Points toward vLLM or a similar CUDA serving engine for multi-user deployment; local MLX remains appropriate for the laptop demo | Future deployment |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | A large advertised context window does not guarantee robust use of information in the middle | The 256K Gemma window should not be marketed as 256K reliable tutoring memory without positional retrieval tests | Used for caution |
| [ReAct](https://arxiv.org/abs/2210.03629) | Interleave reasoning and actions to consult external sources and environments | Supports future search, fact-checking, dictionary, and weather tools instead of asking the model to guess current facts | Future |
| [Toolformer](https://arxiv.org/abs/2302.04761) | Train models to decide when and how to call external APIs | Supports future tool-use data after the core language model is evaluated | Future |

## Decision hierarchy for the final report

1. **Directly used:** CS336, LoRA, QLoRA, LIMA, Self-Instruct-style generation,
   LLM-as-judge limitations, Gemma 4 documentation, KinyaBERT, IrokoBench.
2. **Used to interpret results:** Chinchilla, scaling laws, Magpie.
3. **Future only:** CPT, LESS selection, DPO, RLHF/RLAIF/RLVR, vLLM,
   long-context evaluation, ReAct, and Toolformer.

The presentation should visually separate these three groups. A future paper
must not be described as a method the team already implemented. Run metrics
should cite the experiment ledger or artifact revision, not a research paper.

# KinyaLM Multilingual Adaptation Strategy

## Decision

KinyaLM should be **bilingual-first and multilingual-preserving**.

- Kinyarwanda is the primary product language.
- English is the bridge language for questions, explanations, and translation.
- French and Swahili are secondary evaluation tracks because they are useful in
  the regional context, but they are not release promises yet.
- Other capabilities inherited from the base model should be preserved where
  possible, but the team should not claim support without testing them.

This scope is realistic. Trying to make a small project equally strong in 20 or
100 languages would dilute the data and make quality claims impossible to
defend.

## The Training Decision Tree

```text
unchanged multilingual bases
          |
          v
blind Kinyarwanda + English bake-off
          |
          +-- strong Kinyarwanda --> reviewed SFT --> blind evaluation
          |
          +-- coherent but weak --> continued-pretraining pilot
                                      |
                                      v
                                repeat bake-off
                                      |
                                      +-- passes --> reviewed SFT
                                      +-- fails  --> reject the base
```

SFT means supervised fine-tuning on prompt-and-response examples. It teaches
the model how to act as a tutor. Continued pretraining means next-token training
on a large clean text corpus. It improves the model's underlying command of the
language.

The first Qwen experiment showed why these stages must not be confused: 1,000
short conversations can shape style, but they cannot reliably install missing
vocabulary, morphology, grammar, and world knowledge.

## Stage 0: Select the Base Before Training

Start with the unchanged candidates listed in the
[Track 2 experiment report](experiments/2026-07-20-track2-baseline-report.md):

1. `google/gemma-4-12B-it`
2. `google/gemma-4-31B-it`
3. `google/gemma-2-27b-it`
4. `ptrdvn/kakugo-3B-kin`
5. `CohereForAI/aya-101` as a legacy control

The first paid round compares the two Gemma 4 candidates. Kakugo and Qwen are
already recorded controls; add Gemma 2 or Aya only when the first round does
not produce a clear finalist. The pinned settings and launch procedure are in
the [multilingual bake-off runbook](multilingual-bakeoff-run.md).

Run all candidates on identical hidden prompts. Do not fine-tune every model
and compare them afterward; that spends compute before answering the cheaper
question of whether the base understands Kinyarwanda at all.

Advance a base only if fluent reviewers find that it can already:

- maintain Kinyarwanda without drifting into unrelated language;
- translate in both directions without changing the meaning;
- distinguish common words and expressions;
- explain simple grammar without inventing rules;
- follow corrections over multiple turns;
- stop normally without loops or copied filler.

## Four Separate Data Lanes

Keep these data types in separate paths and manifests. They solve different
problems and must not leak into one another.

| Lane | Purpose | Example material | Training use |
| --- | --- | --- | --- |
| Language corpus | Learn Kinyarwanda itself | licensed books, news, public documents, native web text, transcripts | continued pretraining |
| Tutor instructions | Learn desired behavior | reviewed lessons, corrections, translation, dialogue, explanations | SFT |
| Preferences | Learn which answer is better | reviewer-ranked answer pairs and corrected failures | DPO or another preference stage |
| Evaluation | Measure generalization | task bank, Belebele, FLORES, IrokoBench | never train |

Raw books and news are not SFT conversations. Synthetic conversations are not
a substitute for native-written language. Benchmark answers are not training
examples.

## Stage 1: Language Adaptation When Needed

Skip this stage when an unchanged base already clears the Kinyarwanda gate.
Otherwise, test a small continued-pretraining pilot before committing to a
large run.

### Pilot target

- 50–100 million **deduplicated, high-quality** Kinyarwanda tokens.
- One pass or less over repeated sources.
- Low learning rate and frequent held-out checks.
- A replay mixture to protect the base model's English and multilingual skills.

Starting token mixture:

| Language group | Share | Reason |
| --- | ---: | --- |
| Native Kinyarwanda | 70% | Make the target language impossible for larger corpora to drown out |
| English replay | 20% | Preserve instruction following, reasoning, and the bridge language |
| French replay | 5% | Preserve a useful regional language without making a release claim |
| Swahili replay | 5% | Preserve a useful regional language without making a release claim |

Use temperature-based language sampling rather than sampling directly by
corpus size. Direct sampling would let English dominate every batch.

If the pilot improves the held-out Kinyarwanda tasks without a material English
regression, scale toward roughly 0.5–2 billion genuinely useful Kinyarwanda
tokens as the corpus permits. Do not reach a token target by repeating weak
material. Fifty million clean tokens are more informative than a billion
duplicated or machine-corrupted tokens.

When compute allows, full-parameter continued pretraining is the strongest
language-adaptation test. A cheaper pilot may train adapters plus the token
embeddings and language-model head, but ordinary attention-only QLoRA should
not be treated as equivalent: new language knowledge must also reach token
representations and output probabilities.

## Stage 2: Tutor SFT

The current 1,000-row pack is enough for a controlled pilot after fluent review.
It is not the final tutor dataset. A serious next target is 5,000–20,000
reviewed examples, expanded only when each batch survives native-speaker review.

Recommended response-language mix:

| Response mode | Share |
| --- | ---: |
| Kinyarwanda | 70% |
| Bilingual or code-switched | 20% |
| English explanation | 10% |

Coverage requirements:

- at least 25% multi-turn conversations;
- both English-to-Kinyarwanda and Kinyarwanda-to-English translation;
- correction, grammar, vocabulary, reading, dialogue, register, and ambiguity;
- beginner, intermediate, and advanced levels;
- short answers as well as longer guided lessons;
- explicit uncertainty and clarification behavior;
- identity examples that consistently describe the system as KinyaLM;
- at least half of the Kinyarwanda should be native-authored or substantially
  rewritten by fluent speakers, not literal translationese.

Balance task families during sampling. A large easy translation family should
not crowd grammar, multi-turn repair, or culturally sensitive use out of the
training batches.

## Stage 3: Preference Learning

After the first useful SFT model exists, save pairs of:

- original answer versus fluent correction;
- concise answer versus rambling answer;
- natural Kinyarwanda versus translationese;
- honest uncertainty versus confident invention;
- successful multi-turn repair versus repeated failure.

A first preference set of 1,000–3,000 carefully reviewed comparisons can target
the failure modes that remain after SFT. Do not generate preference labels from
the same model being trained without human calibration.

## Three-Layer Quality Control

The project must optimize the real tutoring goal at all three adaptation
layers, not the easiest proxy metric available at each layer.

| Layer | Useful optimization signal | How the signal can mislead us | Required counter-checks |
| --- | --- | --- | --- |
| Language adaptation (CPT) | Held-out next-token loss on approved Kinyarwanda text | A model can lower loss by memorizing duplicated sources or forget English while specializing | Deduplication audit, held-out Kinyarwanda perplexity, English-retention set, contamination check, and generation review |
| Tutor behavior (SFT) | Assistant-only loss on reviewed demonstrations | Lower loss can coexist with repetition, copied phrasing, wrong meaning, or register drift | Base-versus-adapter task results, semantic correctness, register, response-length distribution, repetition rate, and native-speaker review |
| Preference or RL optimization | Human preferences, model judges, and programmatic checks | A single reward can be hacked; the model may become short, casual, or formulaic while scoring well | Multi-objective reward records, hard correctness floors, adversarial prompts, held-out judges, and blinded native-speaker evaluation |

A Castform practitioner experiment shared with the team gives a concrete
warning. Its first rule-based style reward produced very short answers that
dropped source content. Adding semantic-preservation and tone-preservation
judges fixed specific failures, but optimizing a collection of open detectors
still did not generalize to unseen closed detectors; increasing the model size
did not solve the objective problem. This is engineering evidence, not a
Kinyarwanda result or a peer-reviewed paper, but it matches the reward-hacking
and proxy-overoptimization risks documented in the research literature.

For KinyaLM, a later preference or RL environment must record separate reward
components rather than hiding them in one unexplained score:

- Kinyarwanda correctness and naturalness;
- task completion and preservation of the user's meaning;
- requested language, formality, and cultural register;
- tutoring clarity and useful correction;
- concision without missing required information;
- repetition and control-token failures;
- calibrated uncertainty instead of invented rules or facts.

Correctness, meaning preservation, and catastrophic-repetition checks are hard
gates: a fluent-looking answer cannot compensate for failing them. LLM judges
may scale triage, but reward prompts and weights must be varied during stress
tests, tested on held-out task families, and calibrated against native-speaker
decisions. The immediate project step remains reviewed SFT and blind
base-versus-adapter evaluation; this section defines the conditions for a later
DPO or RL experiment.

Research basis:

- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) identifies
  reward hacking as a failure caused by optimizing an imperfect objective.
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
  establishes preference feedback as a way to communicate complex goals.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
  shows that stronger optimization of a proxy reward can reduce performance on
  the underlying target.
- [Castform RL fine-tuning workflow](https://castform.com/docs/getting-started/how-it-works/)
  and the team's shared August 3 practitioner note motivate the concrete
  multi-reward monitoring checklist above.

## Evaluation Matrix

Every candidate and training stage must cover all four directions:

| Prompt | Required response | Product behavior |
| --- | --- | --- |
| Kinyarwanda | Kinyarwanda | normal tutoring and conversation |
| English | Kinyarwanda | translation and immersion practice |
| Kinyarwanda | English | definitions and explanations for learners |
| Mixed | requested language | natural code-switching and clarification |

Score results by task family, learner level, and language direction. An average
score must not hide a model that fails all grammar or all multi-turn examples.

Starting release gates:

- average Kinyarwanda correctness at least 4/5 from fluent reviewers;
- at least 80% of tutor prompts pass overall;
- no task family below 70% pass;
- no repeated refusal to use Kinyarwanda;
- repetition-loop rate below 1%;
- no benchmark leakage;
- English retention within five percentage points of the unchanged base on the
  team's fixed English control set;
- two-reviewer agreement measured and disagreements adjudicated.

These thresholds are initial engineering gates, not claims that a model is
frontier quality. The team can tighten them after reviewer calibration.

## Retrieval Instead of Memorizing Everything

Use retrieval for facts that should be sourced rather than guessed:

- dictionary entries;
- grammar references;
- approved cultural notes;
- curriculum material;
- current public information.

KinyaBERT or a multilingual embedding model can help retrieve Kinyarwanda text,
but an encoder does not replace the generative tutor. The selected LLM should
write the answer from retrieved, licensed sources and cite the source when the
product requires factual reliability.

## Immediate Execution Order

1. Freeze the Qwen adapter as a negative baseline.
2. Run the unchanged base-model bake-off and blind review.
3. Choose the best two candidates by Kinyarwanda quality, not parameter count.
4. Decide separately for each finalist whether it needs continued pretraining.
5. Promote only fluent-approved rows into the 1,000-example SFT pilot.
6. Compare each tuned model with its own unchanged base.
7. Expand data from observed failures, then repeat the same evaluation.

This produces a defensible multilingual system: strong where the team has data
and reviewers, preserved elsewhere, and honest about every language it has not
yet earned the right to claim.

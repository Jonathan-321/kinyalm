# Project Charter

## Title

Kinyarwanda Language Learning LM

## One-Sentence Goal

Build a CS336-inspired language-modeling project that teaches the full LM
pipeline from scratch while producing a small Kinyarwanda learning tutor demo.

## Contributors

- Jonathan Muhire ([@Jonathan-321](https://github.com/Jonathan-321))
- Tessy Mugisha ([@TessyMugisha](https://github.com/TessyMugisha))
- Bonheur Byiringiro ([@BonheurByiringiro](https://github.com/BonheurByiringiro))

## Tracks

### Track A: From-Scratch LM

Build the educational pipeline:

- BPE tokenizer,
- decoder-only Transformer,
- cross-entropy loss,
- AdamW optimizer,
- training loop,
- checkpointing,
- sampling,
- validation loss/perplexity,
- tokenizer analysis.

### Track B: Practical Tutor MVP

Build the useful prototype:

- approved grammar and vocabulary notes,
- retrieval-first answer flow,
- curated examples,
- sentence correction examples,
- quizzes and practice prompts,
- human review.

## Non-Goals

- Do not claim to build a production-grade Kinyarwanda LLM.
- Do not use copyrighted textbooks or lessons without permission.
- Do not present tutor outputs as authoritative without human review.
- Do not depend on expensive compute for final success.
- Do not let a flashy demo replace reproducibility and documentation.

## Success Criteria

- The repo runs from data preparation to tokenizer training, small LM training,
  sampling, and evaluation.
- The tokenizer analysis says something specific about Kinyarwanda morphology.
- The small LM shows decreasing validation loss on a documented corpus.
- The tutor MVP handles 20-30 curated learning tasks with acceptable human
  review.
- The final report is honest about limitations, data, compute, and risks.

## First Milestone

By the end of Week 1:

- owners are assigned,
- data sources are logged,
- professor feedback is requested,
- constraints are documented,
- tokenizer work can begin.

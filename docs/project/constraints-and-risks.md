# Constraints And Risks

This project is feasible only if the constraints stay visible. The main risk is
not that the model code is impossible. The main risk is building the wrong scope
for the available data, time, compute, and review capacity.

## Hard Constraints

### Time

The project window is short: June 30 to August 14, 2026. That means the team
needs weekly demos, not a single final integration week.

### Data

Clean, licensed Kinyarwanda text is the central bottleneck. A weak corpus will
limit both tokenizer quality and LM quality.

Required behavior:

- log every source,
- record license or permission status,
- keep raw and cleaned data separate,
- deduplicate,
- filter non-Kinyarwanda text,
- document known bias.

### Copyright And Permissions

Useful language-learning PDFs and course pages may not be freely reusable. Do
not scrape, redistribute, or train on copyrighted materials unless permission is
clear.

Safe fallback:

- use public/open datasets,
- use teacher-authored notes,
- use short manually written examples,
- summarize sources only when allowed,
- ask the professor before using ambiguous material.

### Compute

The from-scratch LM should be designed to run small first. The final grade or
demo should not depend on getting a large GPU.

Paths:

- Bronze: tokenizer analysis + tiny LM + RAG tutor.
- Silver: one GPU for longer LM training or small LoRA fine-tuning.
- Gold: stronger cloud GPU only if available.

### Evaluation

Automatic loss/perplexity is not enough for a tutor. A grammar explanation can
sound confident and still be wrong.

Required behavior:

- create human-scored tasks,
- include uncertainty behavior,
- record failure cases,
- avoid claiming authority without review.

## Main Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Not enough clean Kinyarwanda text | From-scratch LM quality will be weak | Treat tokenizer analysis and data card as core deliverables; use RAG for useful demo |
| Licensing uncertainty | Data or model cannot be shared | Keep source log; use open/permissioned sources; ask professor early |
| Compute limits | Training may be slow or unstable | Start tiny; keep CPU-friendly configs; scale only after sanity checks |
| Hallucinated grammar explanations | Tutor could mislead learners | Retrieval-first design; approved notes; human evaluation |
| Scope creep | Final demo unfinished | Freeze features by Aug 3; weekly demos; cut fine-tuning if RAG is enough |
| Team coordination issues | Duplicated work and missed deadlines | Assign owners; use weekly written updates; keep issue tracker current |
| Data contamination | Evaluation results become misleading | Separate train/eval examples; document source overlap |
| Overfitting tiny corpus | Loss improves but model memorizes | Use held-out validation, sample checks, and memorization checks |

## Early Decision Questions

Ask the professor:

1. Which data sources are acceptable for class use?
2. Is the Peace Corps language lesson PDF acceptable as a reference, training
   source, or neither?
3. How much from-scratch implementation is expected versus prototype usefulness?
4. Is cloud GPU access available?
5. What evaluation evidence would make the final project strong?

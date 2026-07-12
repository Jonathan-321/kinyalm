# Week 1 Update

## What Changed

- GitHub repo is public: https://github.com/Jonathan-321/kinyalm
- Tessy Mugisha and Bonheur Byiringiro were invited as collaborators with write access.
- Project docs now list contributors, starter owners, Week 1 tasks, data risks,
  and the first implementation target.
- Hugging Face candidate Kinyarwanda datasets were logged for review.
- Digital Umuganda TTS and MT sources are now approved for tokenizer and
  small-LM experiments, with attribution/domain notes still needed for final
  cards.
- Early feasibility work showed that training loss can improve on approved text
  while generated samples still fail learner-quality review.
- The MT Kinyarwanda-side import is cleaner after fixing cp1252 decoding, but
  useful tutor behavior still depends on data cleanup, benchmark separation,
  and fluent-speaker review.

## What We Learned

- The first bottleneck is data quality and permissions, not model code.
- The safest first implementation milestone is tokenizer analysis on a small,
  documented corpus.
- Lower validation loss is not enough evidence for a learner-facing tutor.
- Tutor usefulness should start with retrieval over approved notes, not
  unsupported generation.

## Current Blockers

- Tessy and Bonheur need to accept GitHub collaborator invitations.
- Professor feedback is needed on scope, data permissions, compute expectations,
  and evaluation standards.
- Sample quality needs real linguistic review; lower validation loss alone is
  still not enough.

## Decisions Needed

- Confirm whether the starter role split works for everyone.
- Decide whether any additional Hugging Face datasets are approved,
  reference-only, or blocked.
- Decide the first Friday demo format.
- Confirm available compute.

## Next Week

- Tessy adds attribution/domain notes for the approved sources.
- Bonheur reviews the morphology tokenizer examples and CS336 tokenizer
  requirements.
- Jonathan sends the professor email, keeps GitHub issues current, and drafts
  tutor MVP tasks.

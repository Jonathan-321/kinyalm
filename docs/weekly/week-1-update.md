# Week 1 Update

## What Changed

- GitHub repo is public: https://github.com/Jonathan-321/kinyalm
- Tessy Mugisha and Bonheur Byiringiro were invited as collaborators with write access.
- Project docs now list contributors, starter owners, Week 1 tasks, data risks,
  and the first implementation target.
- Hugging Face candidate Kinyarwanda datasets were logged for review.
- Digital Umuganda TTS and MT sources are now approved for KILM baseline work,
  with attribution/domain notes still needed for final cards.
- KILM ran a `small` MPS baseline on the full approved TTS split; validation
  perplexity moved from 605.7486 to 137.0228 over 200 steps.
- A 10,000-step continuation moved validation perplexity to 59.5324, but the
  generated sample still failed the smoke check.
- The MT Kinyarwanda-side run is stronger: after fixing cp1252 decoding, the
  `baseline_gpu` 10,000-step continuation moved validation perplexity to
  21.0469 and produced a sample marked `needs-linguistic-review`.

## What We Learned

- The first bottleneck is data quality and permissions, not model code.
- The safest first implementation milestone is tokenizer analysis on a small,
  documented corpus.
- Tutor usefulness should start with retrieval over approved notes, not
  unsupported generation.

## Current Blockers

- Tessy and Bonheur need to accept GitHub collaborator invitations.
- Professor feedback is needed on scope, data permissions, compute expectations,
  and evaluation standards.
- MT sample quality now needs real linguistic review; lower validation loss
  alone is still not enough.

## Decisions Needed

- Confirm whether the starter role split works for everyone.
- Decide whether any additional Hugging Face datasets are approved,
  reference-only, or blocked.
- Decide the first Friday demo format.
- Confirm available compute.

## Next Week

- Tessy adds attribution/domain notes for the approved sources.
- Bonheur reviews the KILM morphology tokenizer examples and CS336 tokenizer
  requirements.
- Jonathan sends the professor email, keeps GitHub issues current, and drafts
  tutor MVP tasks.

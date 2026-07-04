# Learning Task Bank

These tasks can evaluate either:

- Track A model outputs, if the from-scratch LM becomes strong enough,
- or a retrieval-first tutor fallback, if Track A is not useful enough alone.

Every answer should be reviewed by a fluent speaker or instructor before being
used as evidence.

## Starter Tasks

| ID | Category | Learner Prompt | Review Focus |
| --- | --- | --- | --- |
| T001 | Greeting | Explain when to use `Muraho`. | correctness, register, beginner clarity |
| T002 | Greeting | What does `Amakuru?` mean, and how could someone answer? | correctness, natural examples |
| T003 | Greeting | Explain the difference between `Muraho` and `Mwaramutse`. | nuance, register, time-of-day usage |
| T004 | Vocabulary | Teach five classroom words in Kinyarwanda. | vocabulary correctness |
| T005 | Vocabulary | Give five family-related words in Kinyarwanda with English meanings. | vocabulary correctness |
| T006 | Sentence correction | Correct: `Ndi kwiga Ikinyarwanda.` | grammar, humility if uncertain |
| T007 | Sentence correction | Correct: `Jyewe yitwa Jonathan.` | pronoun/name construction |
| T008 | Translation | Translate: `I am a student.` | grammar and naturalness |
| T009 | Translation | Translate: `Please repeat slowly.` | politeness and classroom phrasing |
| T010 | Explanation | Explain `Nitwa Jonathan.` word by word. | clarity, morphology |
| T011 | Explanation | Explain `Ndiga Ikinyarwanda.` word by word. | verb form, object/language name |
| T012 | Quiz generation | Create a 5-question beginner vocabulary quiz. | answer key correctness |
| T013 | Fill in blank | Make 5 fill-in-the-blank questions for greetings. | level appropriateness |
| T014 | Dialogue | Write a short beginner dialogue between a student and teacher. | naturalness, level |
| T015 | Culture/register | Explain when a phrase might sound too casual or too formal. | cultural caution |
| T016 | Uncertainty | If unsure about a grammar rule, how should the tutor respond? | humility behavior |
| T017 | Morphology | Give examples of words beginning with `umu-` and explain the pattern carefully. | morphology accuracy |
| T018 | Orthography | Explain how punctuation changes `Muraho! Amakuru?`. | punctuation and tone |
| T019 | Error analysis | Identify what might be wrong in a learner sentence and ask a clarification question. | correction style |
| T020 | Lesson plan | Create a 10-minute beginner lesson on greetings. | structure, correctness |

## Scoring

Use the rubric in:

```text
docs/evaluation/evaluation-plan.md
```

Suggested minimum fields:

- correctness,
- clarity,
- cultural appropriateness,
- helpfulness,
- uncertainty behavior,
- notes from reviewer.

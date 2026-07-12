# Learning Task Bank

These prompts are for evaluating a Kinyarwanda learning tutor. They are not a
training dataset.

Rows marked `benchmark-only` must not be copied into SFT training data. They
exist so the team can test whether a model generalizes beyond the examples it
was shown during training.

## Task Bank

| ID | Category | Split | Learner Prompt | Review Focus |
| --- | --- | --- | --- | --- |
| T001 | Greeting | benchmark-only | Explain when to use `Muraho`. | correctness, register, beginner clarity |
| T002 | Greeting | train-template | What does `Amakuru?` mean, and how could someone answer? | correctness, natural examples |
| T003 | Greeting | benchmark-only | Explain the difference between `Muraho` and `Mwaramutse`. | nuance, register, time-of-day usage |
| T004 | Greeting | train-template | Give three polite ways to greet a teacher in Kinyarwanda. | politeness, natural classroom use |
| T005 | Greeting | benchmark-only | Write a short greeting exchange between a student and a teacher. | naturalness, level, turn-taking |
| T006 | Vocabulary | train-template | Teach five classroom words in Kinyarwanda. | vocabulary correctness |
| T007 | Vocabulary | train-template | Give five family-related words in Kinyarwanda with English meanings. | vocabulary correctness |
| T008 | Vocabulary | benchmark-only | Teach the word `ishuri` and give two simple example sentences. | word meaning, example quality |
| T009 | Vocabulary | train-template | Explain the words `umwarimu` and `umunyeshuri`. | noun meaning, beginner clarity |
| T010 | Vocabulary | benchmark-only | Give a mini vocabulary lesson about people at school. | accuracy, useful grouping |
| T011 | Sentence correction | benchmark-only | Correct: `Ndi kwiga Ikinyarwanda.` | grammar, humility if uncertain |
| T012 | Sentence correction | train-template | Correct: `Jyewe yitwa Jonathan.` | pronoun/name construction |
| T013 | Sentence correction | benchmark-only | Correct: `Ndi umwarimu?` if the learner meant "Am I a teacher?" | question formation, uncertainty |
| T014 | Sentence correction | train-template | Explain what is wrong in a learner sentence before giving the correction. | correction style, clarity |
| T015 | Sentence correction | benchmark-only | Correct a learner who used a literal English word order in Kinyarwanda. | non-literal translation handling |
| T016 | Translation EN-RW | benchmark-only | Translate: `I am a student.` | grammar and naturalness |
| T017 | Translation EN-RW | train-template | Translate: `Please repeat slowly.` | politeness and classroom phrasing |
| T018 | Translation EN-RW | benchmark-only | Translate: `My name is Aline.` | name construction |
| T019 | Translation EN-RW | train-template | Translate: `I am learning Kinyarwanda.` | verb form and language name |
| T020 | Translation EN-RW | benchmark-only | Translate: `The teacher helps the student.` | subject/object clarity |
| T021 | Translation RW-EN | train-template | Translate: `Nitwa Jonathan.` | meaning, word-by-word option |
| T022 | Translation RW-EN | benchmark-only | Translate: `Ndiga Ikinyarwanda.` | tense and natural English |
| T023 | Translation RW-EN | train-template | Translate: `Murakoze.` | register and politeness |
| T024 | Translation RW-EN | benchmark-only | Translate: `Subiramo buhoro.` | classroom command meaning |
| T025 | Translation RW-EN | train-template | Translate: `Mfasha gusobanukirwa iri jambo.` | phrase meaning, directness |
| T026 | Grammar explanation | benchmark-only | Explain `Nitwa Jonathan.` word by word. | clarity, morphology |
| T027 | Grammar explanation | train-template | Explain `Ndiga Ikinyarwanda.` word by word. | verb form, language name |
| T028 | Grammar explanation | benchmark-only | Explain why `umwarimu` and `abarimu` are related. | noun class/plural relation |
| T029 | Grammar explanation | train-template | Explain the pattern in `umuntu` and `abantu`. | noun class basics |
| T030 | Grammar explanation | benchmark-only | Explain the apostrophe in `w'Ikinyarwanda`. | orthography, elision |
| T031 | Morphology | train-template | Give examples of words beginning with `umu-` and explain the pattern carefully. | morphology accuracy |
| T032 | Morphology | benchmark-only | Compare `igitabo` and `ibitabo`. | singular/plural relation |
| T033 | Morphology | train-template | Explain `gukora`, `Ndakora`, and `Barakora`. | verb stem and subject prefixes |
| T034 | Morphology | benchmark-only | Explain why `Sinzabikora` is hard for tokenizers. | segmentation, uncertainty |
| T035 | Morphology | train-template | Give a beginner explanation of noun classes without overloading the learner. | simplicity, correctness |
| T036 | Quiz generation | train-template | Create a 5-question beginner vocabulary quiz. | answer key correctness |
| T037 | Quiz generation | benchmark-only | Make 5 fill-in-the-blank questions for greetings. | level appropriateness |
| T038 | Quiz generation | train-template | Create a short quiz using `Muraho`, `Amakuru`, and `Murakoze`. | prompt clarity, answers |
| T039 | Quiz generation | benchmark-only | Create a correction quiz with three learner mistakes and answers. | correction accuracy |
| T040 | Quiz generation | train-template | Make a quiz that separates translation questions from grammar questions. | structure, grading clarity |
| T041 | Dialogue | benchmark-only | Write a short beginner dialogue between a student and teacher. | naturalness, level |
| T042 | Dialogue | train-template | Write a two-turn conversation where a learner asks for repetition. | classroom usefulness |
| T043 | Dialogue | benchmark-only | Write a dialogue where the tutor explains a word and checks understanding. | tutor behavior |
| T044 | Dialogue | train-template | Write a beginner conversation that uses a greeting, a name, and thanks. | natural flow |
| T045 | Culture/register | benchmark-only | Explain when a phrase might sound too casual or too formal. | cultural caution |
| T046 | Culture/register | train-template | Explain why a tutor should avoid overconfident cultural claims. | safety, humility |
| T047 | Uncertainty | benchmark-only | If unsure about a grammar rule, how should the tutor respond? | humility behavior |
| T048 | Uncertainty | benchmark-only | The learner asks for a proverb translation the tutor is not sure about. What should it do? | uncertainty, safe response |
| T049 | Error analysis | train-template | Identify what might be wrong in a learner sentence and ask a clarification question. | correction style |
| T050 | Lesson plan | benchmark-only | Create a 10-minute beginner lesson on greetings. | structure, correctness |

## Split Rules

- `benchmark-only`: held-out evaluation prompt; never copy into training data.
- `train-template`: safe to use as a pattern for writing separate training
  examples, but do not copy the row itself directly into the SFT file.

This bank currently has 50 prompts, including 26 `benchmark-only` prompts.

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

# Data Source Log

Every source goes here before data enters the project. If the license or
permission is unclear, mark it as blocked.

| Source | URL / Location | Type | Intended Use | License / Permission | Owner | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DigitalUmuganda Kinyarwanda TTS dataset | https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-tts-dataset | Kinyarwanda sentence text | tokenizer analysis, small LM experiments, possible seed examples after review | dataset LICENSE fetched as `cc0-1.0` | Tessy Mugisha | approved | Local prepared text contains 3,922 unique sentence lines. Provenance risks remain; do not quote sensitive rows. |
| DigitalUmuganda Kinyarwanda-English MT dataset | https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset | Parallel text | Kinyarwanda-side tokenizer/LM corpus, bilingual examples, possible evaluation | visible `cc-by-4.0` | Tessy Mugisha | approved | Kinyarwanda side can be imported after cp1252 decoding; current prepared corpus has 44,527 clean lines and zero replacement characters. Preserve attribution and document translation/domain bias in data cards. |
| mbazaNLP Kinyarwanda-English parallel dataset | https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset | Parallel text | bilingual examples, possible evaluation | visible `cc-by-4.0` | Tessy Mugisha | investigate | Card says religious-source web data and not curated, only cleaned; use carefully and document bias. |
| mbazaNLP Kinyarwanda language model dataset | https://huggingface.co/datasets/mbazaNLP/kinyarwanda-language-model-dataset | Monolingual text | tokenizer and LM pretraining candidate | no visible license | Tessy Mugisha | blocked | Metadata exposes no useful files beyond `.gitattributes`; do not use unless source is clarified. |
| DigitalUmuganda Common Voice Kinyarwanda text dataset | https://huggingface.co/datasets/DigitalUmuganda/common-voice-kinyarwanda-text-dataset | Speech transcript text | tokenizer corpus candidate | no visible license | Tessy Mugisha | blocked | Large candidate, but license was not visible in HF metadata; verify before any use. |
| RogerB Kinyarwanda Wikipedia snapshot | https://huggingface.co/datasets/RogerB/Kinyarwanda_wikipedia20230920 | Wikipedia text | tokenizer and LM pretraining candidate | no visible license | Tessy Mugisha | blocked | Likely needs attribution/share-alike review; dataset card is sparse. |
| Mikecyane Kinyarwanda chat | https://huggingface.co/datasets/Mikecyane/Kinyarwanda_chat | Chat/instruction text | tutor examples or evaluation inspiration | visible `apache-2.0` | Tessy Mugisha | investigate | License is visible, but the WhatsApp source raises privacy/consent and quality questions. |
| saillab alpaca Kinyarwanda cleaned | https://huggingface.co/datasets/saillab/alpaca-kinyarwanda-cleaned | Instruction data | tutor/fine-tuning candidate | no visible license | Tessy Mugisha | blocked | Later-stage instruction candidate; do not use until license/provenance is clarified. |
| ChrisToukmaji Kinyarwanda instruction tuning | https://huggingface.co/datasets/ChrisToukmaji/kinyarwanda_instruction_tuning | Instruction data | tutor/fine-tuning candidate | no visible license | Tessy Mugisha | blocked | Later-stage research candidate; do not use until license/provenance is clarified. |
| Peace Corps Kinyarwanda language lessons PDF | https://files.peacecorps.gov/multimedia/audio/languagelessons/rwanda/RW_Kinyarwanda_Language_Lessons.pdf | Lesson PDF | reference or approved notes if allowed | permission needed | Tessy Mugisha | blocked | Ask professor whether it can be used and how. |
| University of Chicago Kinyarwanda course page | https://linguistics.uchicago.edu/kinyarwanda | Course/resource page | reference only unless permission is clear | pending review | Tessy Mugisha | investigate | Do not scrape without permission. |
| Belebele Kinyarwanda (`kin_Latn`) | https://huggingface.co/datasets/facebook/belebele | Multiple-choice reading comprehension | held-out external benchmark | visible `cc-by-sa-4.0` | Jonathan Muhire | benchmark-only | 900 Kinyarwanda rows. Use for evaluation only; do not train on test questions. |
| FLORES-200 Kinyarwanda (`kin_Latn`) | https://huggingface.co/datasets/facebook/flores | Translation benchmark | held-out English/Kinyarwanda translation evaluation | visible `cc-by-sa-4.0` | Jonathan Muhire | benchmark-only | Use chrF/spBLEU plus human spot checks. Keep separate from SFT examples. |
| IrokoBench AfriXNLI Kinyarwanda (`kin`) | https://huggingface.co/datasets/masakhane/afrixnli | Natural language inference | held-out reasoning benchmark | visible `apache-2.0` | Jonathan Muhire | benchmark-only | 450 validation and 600 test rows for Kinyarwanda. |
| IrokoBench AfriMGSM Kinyarwanda (`kin`) | https://huggingface.co/datasets/masakhane/afrimgsm | Math reasoning | held-out reasoning benchmark | visible `apache-2.0` | Jonathan Muhire | benchmark-only | 8 train and 250 test rows for Kinyarwanda; keep test rows held out. |
| IrokoBench AfriMMLU Kinyarwanda (`kin`) | https://huggingface.co/datasets/masakhane/afrimmlu | Multiple-choice knowledge QA | held-out knowledge benchmark | visible `apache-2.0` | Jonathan Muhire | benchmark-only | 608 Kinyarwanda rows across validation/dev/test. |
| AfriSenti-SemEval Kinyarwanda (`kin`) | https://github.com/afrisenti-semeval/afrisent-semeval-2023 | Sentiment classification | held-out classification benchmark | visible `cc-by-4.0` | Jonathan Muhire | benchmark-only | Useful for classification only; domain is social media, not classroom tutoring. |
| KINNEWS | https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus | News topic classification | held-out classification benchmark | visible `MIT` repo license | Jonathan Muhire | benchmark-only | 21,268 Kinyarwanda news articles in 14 classes; use for external classification evaluation, not tutor SFT. |

## Status Values

- `investigate`: source needs review.
- `approved`: allowed for intended use.
- `benchmark-only`: can be used for held-out evaluation, not training.
- `reference-only`: can inform humans, not training data.
- `blocked`: do not use until resolved.
- `rejected`: not suitable.

## Required Questions

For each source:

1. Who created it?
2. What license or permission applies?
3. Can we redistribute processed data?
4. Can we train on it?
5. Can we quote examples in the final report?
6. What bias or domain does it contain?

## SFT Use Decisions

This section is the training-data view of the source log. A source can be
useful for evaluation or reference while still being wrong for SFT.

| Source | SFT Decision | Reason |
| --- | --- | --- |
| DigitalUmuganda Kinyarwanda TTS dataset | possible source-derived examples after review | Approved license, but domain is mixed scraped sentence text rather than tutor dialogue. Use only short, reviewed, non-sensitive rows. |
| DigitalUmuganda Kinyarwanda-English MT dataset | possible translation examples after review | Approved license and useful bilingual signal. Preserve CC-BY attribution and review translations before adding to SFT. |
| mbazaNLP Kinyarwanda-English parallel dataset | not yet | License is visible, but source/domain notes need review before training. |
| mbazaNLP Kinyarwanda language model dataset | no | No visible license. |
| DigitalUmuganda Common Voice Kinyarwanda text dataset | no | No visible license in first-pass metadata. |
| RogerB Kinyarwanda Wikipedia snapshot | no | Sparse card and likely attribution/share-alike review needed. |
| Mikecyane Kinyarwanda chat | not yet | Visible license is not enough; WhatsApp provenance creates privacy and consent risk. |
| saillab alpaca Kinyarwanda cleaned | no | No visible license. |
| ChrisToukmaji Kinyarwanda instruction tuning | no | No visible license. |
| Peace Corps Kinyarwanda language lessons PDF | human reference only unless permission changes | Useful for humans writing original examples; do not copy into training without permission. |
| University of Chicago Kinyarwanda course page | human reference only unless permission changes | Useful for humans writing original examples; do not scrape or copy without permission. |
| Belebele, FLORES-200, IrokoBench, AfriSenti, KINNEWS | no, benchmark-only | Keep external evaluation separate from training data. |


## Approval Notes

### DigitalUmuganda Kinyarwanda TTS dataset

```text
Source: DigitalUmuganda/kinyarwanda-tts-dataset
Reviewer: Tessy Mugisha
Date: 2026-07-11
Decision: approved
Allowed use: tokenizer analysis, small LM experiments, quoted examples in report
License: CC0 1.0 Universal, confirmed by reading the LICENSE file in the repo
Attribution requirement: none legally required; we credit Digital Umuganda anyway
Redistribution allowed: yes
Training allowed: yes
Report examples allowed: yes
Risks:
  - The dataset card has no license tag and no provenance statement. The CC0
    license exists only as a LICENSE file. Hugging Face shows a missing-metadata
    warning on the card.
  - Contents look scraped, not authored. Rows include news sentences, Bible
    verses, political commentary about named public figures, and genocide
    survivor testimony.
  - CC0 waives Digital Umuganda's rights only. Section 4(c) of CC0 says the
    Affirmer does not clear third-party rights. If these sentences came from
    news outlets, those rights were never cleared by anyone.
  - Domain is news and religious text. It is not learner language. A tutor
    trained on this will not sound like a tutor.
  - Some sentences name private individuals in sensitive contexts.
Next action:
  - Document the provenance gap in the data card and the final report.
  - Do not quote genocide-testimony rows as tokenizer or tutor examples.
  - Decide as a team whether to keep using it. Recommendation: keep it for
    tokenizer analysis, disclose the provenance gap, do not present the LM
    samples as learner content.
```

### DigitalUmuganda Kinyarwanda-English MT dataset

```text
Source: DigitalUmuganda/kinyarwanda-english-machine-translation-dataset
Reviewer: Tessy Mugisha
Date: 2026-07-11
Decision: approved
Allowed use: Kinyarwanda-side tokenizer and LM corpus, bilingual examples, evaluation
License: CC-BY-4.0, tagged on the dataset card
Attribution requirement: yes, required by CC-BY-4.0
Redistribution allowed: yes, with attribution and a note of changes
Training allowed: yes
Report examples allowed: yes, with attribution
Risks:
  - Source files are cp1252-encoded, not UTF-8. Hugging Face's own dataset
    viewer fails to load this dataset for that reason. Our cp1252 decoding
    should be documented as a change we made.
  - The card says sentences were curated and translated. It does not say where
    the original Kinyarwanda came from.
  - Translation-domain bias. Sentences chosen for MT training are not the same
    distribution as natural Kinyarwanda.
Next action:
  - Add the attribution string below to the data card, model card, and report.
  - State clearly that we modified the data: re-decoded from cp1252 to UTF-8,
    dropped the English side, filtered to 44,527 lines.
```

## Attribution

Required by CC-BY-4.0. Must appear in the data card, the model card, and the
final report.

```text
Kinyarwanda-English Machine Translation Dataset by Digital Umuganda.
https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset
Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Modified: re-decoded from cp1252 to UTF-8, Kinyarwanda side extracted,
filtered to 44,527 lines.
```

Not legally required, but we credit it anyway:

```text
Kinyarwanda TTS Dataset by Digital Umuganda.
https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-tts-dataset
Released under CC0 1.0.
```

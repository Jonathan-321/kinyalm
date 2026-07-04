# Data Source Log

Every source goes here before data enters the project. If the license or
permission is unclear, mark it as blocked.

| Source | URL / Location | Type | Intended Use | License / Permission | Owner | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DigitalUmuganda Kinyarwanda-English MT dataset | https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset | Parallel text | bilingual examples, possible evaluation | visible `cc-by-4.0` | Tessy Mugisha | investigate | Good candidate for deeper review; still check provenance, attribution, and source/domain bias. |
| mbazaNLP Kinyarwanda-English parallel dataset | https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset | Parallel text | bilingual examples, possible evaluation | visible `cc-by-4.0` | Tessy Mugisha | investigate | Card says religious-source web data and not curated, only cleaned; use carefully and document bias. |
| mbazaNLP Kinyarwanda language model dataset | https://huggingface.co/datasets/mbazaNLP/kinyarwanda-language-model-dataset | Monolingual text | tokenizer and LM pretraining candidate | no visible license | Tessy Mugisha | blocked | Metadata exposes no useful files beyond `.gitattributes`; do not use unless source is clarified. |
| DigitalUmuganda Common Voice Kinyarwanda text dataset | https://huggingface.co/datasets/DigitalUmuganda/common-voice-kinyarwanda-text-dataset | Speech transcript text | tokenizer corpus candidate | no visible license | Tessy Mugisha | blocked | Large candidate, but license was not visible in HF metadata; verify before any use. |
| RogerB Kinyarwanda Wikipedia snapshot | https://huggingface.co/datasets/RogerB/Kinyarwanda_wikipedia20230920 | Wikipedia text | tokenizer and LM pretraining candidate | no visible license | Tessy Mugisha | blocked | Likely needs attribution/share-alike review; dataset card is sparse. |
| Mikecyane Kinyarwanda chat | https://huggingface.co/datasets/Mikecyane/Kinyarwanda_chat | Chat/instruction text | tutor examples or evaluation inspiration | visible `apache-2.0` | Tessy Mugisha | investigate | License is visible, but the WhatsApp source raises privacy/consent and quality questions. |
| saillab alpaca Kinyarwanda cleaned | https://huggingface.co/datasets/saillab/alpaca-kinyarwanda-cleaned | Instruction data | tutor/fine-tuning candidate | no visible license | Tessy Mugisha | blocked | Later-stage instruction candidate; do not use until license/provenance is clarified. |
| ChrisToukmaji Kinyarwanda instruction tuning | https://huggingface.co/datasets/ChrisToukmaji/kinyarwanda_instruction_tuning | Instruction data | tutor/fine-tuning candidate | no visible license | Tessy Mugisha | blocked | Later-stage research candidate; do not use until license/provenance is clarified. |
| Peace Corps Kinyarwanda language lessons PDF | https://files.peacecorps.gov/multimedia/audio/languagelessons/rwanda/RW_Kinyarwanda_Language_Lessons.pdf | Lesson PDF | reference or approved notes if allowed | permission needed | Tessy Mugisha | blocked until reviewed | Ask professor whether it can be used and how. |
| University of Chicago Kinyarwanda course page | https://linguistics.uchicago.edu/kinyarwanda | Course/resource page | reference only unless permission is clear | pending review | Tessy Mugisha | investigate | Do not scrape without permission. |

## Status Values

- `investigate`: source needs review.
- `approved`: allowed for intended use.
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

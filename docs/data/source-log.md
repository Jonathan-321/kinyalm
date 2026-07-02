# Data Source Log

Every source goes here before data enters the project. If the license or
permission is unclear, mark it as blocked.

| Source | URL / Location | Type | Intended Use | License / Permission | Owner | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DigitalUmuganda Kinyarwanda-English MT dataset | https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset | Parallel text | bilingual examples, possible evaluation | TBD | TBD | investigate | Check dataset card and source bias. |
| mbazaNLP Kinyarwanda-English parallel dataset | https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset | Parallel text | bilingual examples, possible evaluation | TBD | TBD | investigate | Check dataset card and source bias. |
| mbazaNLP Kinyarwanda language model dataset | https://huggingface.co/datasets/mbazaNLP/kinyarwanda-language-model-dataset | Monolingual text | tokenizer and LM pretraining candidate | TBD | TBD | investigate | Strong early candidate; inspect license, size, source mix, and language quality. |
| DigitalUmuganda Common Voice Kinyarwanda text dataset | https://huggingface.co/datasets/DigitalUmuganda/common-voice-kinyarwanda-text-dataset | Speech transcript text | tokenizer corpus candidate | TBD | TBD | investigate | Recent HF result; verify license and whether text style is useful for language learning. |
| RogerB Kinyarwanda Wikipedia snapshot | https://huggingface.co/datasets/RogerB/Kinyarwanda_wikipedia20230920 | Wikipedia text | tokenizer and LM pretraining candidate | TBD | TBD | investigate | Check license, cleaning needs, article quality, and possible boilerplate. |
| Mikecyane Kinyarwanda chat | https://huggingface.co/datasets/Mikecyane/Kinyarwanda_chat | Chat/instruction text | tutor examples or evaluation inspiration | TBD | TBD | investigate | Do not use for training until license and quality are reviewed. |
| saillab alpaca Kinyarwanda cleaned | https://huggingface.co/datasets/saillab/alpaca-kinyarwanda-cleaned | Instruction data | tutor/fine-tuning candidate | TBD | TBD | investigate | Later-stage candidate; check translation quality and license. |
| ChrisToukmaji Kinyarwanda instruction tuning | https://huggingface.co/datasets/ChrisToukmaji/kinyarwanda_instruction_tuning | Instruction data | tutor/fine-tuning candidate | TBD | TBD | investigate | Later-stage candidate; inspect source provenance carefully. |
| Peace Corps Kinyarwanda language lessons PDF | https://files.peacecorps.gov/multimedia/audio/languagelessons/rwanda/RW_Kinyarwanda_Language_Lessons.pdf | Lesson PDF | reference or approved notes if allowed | TBD | TBD | blocked until reviewed | Ask professor whether it can be used and how. |
| University of Chicago Kinyarwanda course page | https://linguistics.uchicago.edu/kinyarwanda | Course/resource page | reference only unless permission is clear | TBD | TBD | investigate | Do not scrape without permission. |

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

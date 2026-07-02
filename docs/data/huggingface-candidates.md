# Hugging Face Candidate Datasets

Initial search page:

https://huggingface.co/datasets?search=kinyarwanda

These are candidates, not approved sources. Each one needs a dataset-card,
license, provenance, and quality review before training use.

## Strong Early Candidates

| Dataset | Possible Use | First Review Question |
| --- | --- | --- |
| https://huggingface.co/datasets/mbazaNLP/kinyarwanda-language-model-dataset | tokenizer and LM corpus | What are the original sources and license? |
| https://huggingface.co/datasets/RogerB/Kinyarwanda_wikipedia20230920 | tokenizer and LM corpus | How much cleaning and deduplication is needed? |
| https://huggingface.co/datasets/DigitalUmuganda/common-voice-kinyarwanda-text-dataset | tokenizer corpus | Are speech transcripts useful for this tutor domain? |
| https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset | bilingual examples/eval | Is the license clear and is domain bias documented? |
| https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset | bilingual examples/eval | What sources created the parallel pairs? |

## Later-Stage Tutor / Instruction Candidates

| Dataset | Possible Use | First Review Question |
| --- | --- | --- |
| https://huggingface.co/datasets/Mikecyane/Kinyarwanda_chat | tutor examples | Is the quality high enough for learning tasks? |
| https://huggingface.co/datasets/saillab/alpaca-kinyarwanda-cleaned | instruction tuning | Is it translated synthetic data, and can we use it? |
| https://huggingface.co/datasets/ChrisToukmaji/kinyarwanda_instruction_tuning | instruction tuning | What is the source provenance? |

## Rule

No candidate becomes training data until it is marked `approved` in
`docs/data/source-log.md`.

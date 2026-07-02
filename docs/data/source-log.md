# Data Source Log

Every source goes here before data enters the project. If the license or
permission is unclear, mark it as blocked.

| Source | URL / Location | Type | Intended Use | License / Permission | Owner | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DigitalUmuganda Kinyarwanda-English MT dataset | https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset | Parallel text | bilingual examples, possible evaluation | TBD | TBD | investigate | Check dataset card and source bias. |
| mbazaNLP Kinyarwanda-English parallel dataset | https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset | Parallel text | bilingual examples, possible evaluation | TBD | TBD | investigate | Check dataset card and source bias. |
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

# Gemma 4 12B Local Screening Report

**Run date:** 2026-07-21

**Runtime decision:** passed; the 12B model runs locally on the 32 GB Mac

**Model-quality decision:** provisional fail; native-speaker blind review pending

## Outcome

The unchanged Gemma 4 12B instruction model completed all 26 held-out KinyaLM
tutor prompts locally. The run did not use Lambda or incur cloud cost. Its
mixed 4/8-bit MLX checkpoint stayed below 11.4 GB peak unified memory and
produced a complete reviewer sheet.

The first-pass language result is not strong enough to justify fine-tuning this
base yet. Several failures are objectively visible without resolving subtle
native-speaker judgments: two direct translations reverse the requested
meaning, multiple grammar explanations invent analyses, and four responses end
at the token limit. Fluent reviewers still need to score every blinded row
before the team reports a formal Kinyarwanda pass rate.

## Reproducible Run Record

| Item | Value |
| --- | --- |
| Run ID | `gemma4-12b-mlx-20260721-local-v2` |
| Source model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Local checkpoint | `mlx-community/gemma-4-12B-it-qat-4bit` at `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` |
| Runtime | MLX-LM `0.31.3`, deterministic greedy decoding |
| Quantization | QAT-derived mixed 4/8-bit affine MLX conversion |
| Tasks | 26 `benchmark-only` rows from the 50-task learning bank |
| Thinking | Disabled; no non-empty thought channels recorded |
| Result rows | 26 successful, 0 generation errors |
| Generated tokens | 10,986 total; 408 median per response |
| Generation time | 46.92 minutes total; 104.57 seconds median per response |
| Throughput | 3.99 median tokens/second |
| Peak memory | 11.37 GB unified memory |
| Stop behavior | 22 normal stops, 4 length truncations |
| Published artifacts | [`kinyalm-data-lake/evaluation/model-bakeoffs/gemma4-12b-mlx-20260721-local-v2`](https://huggingface.co/datasets/kinyalm/kinyalm-data-lake/tree/main/evaluation/model-bakeoffs/gemma4-12b-mlx-20260721-local-v2) |

The checkpoint uses the upstream `gemma4_unified` configuration label. The
text runner pins MLX-LM's `gemma4` implementation, ignores only the 11
`vision_embedder.*` tensors that text inference cannot consume, and then
strictly validates all remaining text-model weights.

The original `local-v1` output is superseded. MLX-LM did not automatically
apply Gemma's generation-level suppression of token IDs `258882` and `258883`,
so that run exposed audio/image end markers that the official generation
configuration forbids. The pinned local runtime now suppresses those two IDs.
The definitive `local-v2` run has zero control-token leaks; its language errors
remain and therefore cannot be attributed to that decoding defect.

This is a quantized local screen, not a numerically identical BF16 run. A
candidate that passes locally should still be confirmed in the formal BF16
comparison; quantization does not explain basic meaning reversals such as the
examples below.

## Preliminary Failure Evidence

| Task | Saved behavior | Preliminary result |
| --- | --- | --- |
| T005 | Greeting dialogue repeats `wowe` and ends with an unnatural classroom response | fail |
| T008 | Uses `abanyeshimi` for students and gives an unreliable noun-class note | fail |
| T010 | Invents school-role vocabulary such as `umubashyikiriza` and `umugabane` | fail |
| T015 | Uses the non-Kinyarwanda form `apfelu` while claiming to teach word order | fail |
| T020 | Misspells `umunyeshuri` in a basic translation | fail |
| T022 | Translates `Ndiga Ikinyarwanda` as "I speak Kinyarwanda" | fail |
| T024 | Translates `Subiramo buhoro` as "Wait a little" instead of "Repeat slowly" | fail |
| T026 | Analyzes `Nitwa` as `Ni-` + `-twa`, an unreliable morphology explanation | fail |
| T032 | Assigns `igitabo`/`ibitabo` to the wrong noun class and prefix analysis | fail |
| T034 | Invents a segmentation of `Sinzabikora` and reaches the token limit | fail |
| T039 | Repeats the same "correction" while trying to create a quiz | fail |
| T041 | Beginner dialogue contains malformed phrases such as `Mwese` and `niteranye` | fail |
| T043 | Teaches `Nukora` as "I work" | fail |
| T050 | Teaches `Murakoze` as goodbye, gives questionable usage, and reaches the limit | fail |

These flags are a screening aid. Reviewers should score the complete responses
in `review/blind-review.csv`, not infer a percentage from this selected table.
The automatically generated `summary/automatic-metrics.json` contains only
mechanical flags: errors, truncations, output length, and control-token leaks.

## Decision And Next Step

Do not fine-tune Gemma 4 12B on the 1,000-row SFT pack yet. First, complete the
blind review and record reviewer agreement. In parallel, run the 31B BF16
candidate on cloud compute or add a directly measured Kinyarwanda baseline such
as Gemma 2 27B. The purpose of that next run is not to rescue Gemma 4 by scale;
it is to find a base that already preserves basic Kinyarwanda meaning before
the team spends data and compute shaping its tutoring behavior.

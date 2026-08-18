"""Validate and summarize blind native-speaker model reviews."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

SCORE_COLUMNS = (
    "kinyarwanda_correctness_1_5",
    "beginner_clarity_1_5",
    "grammar_explanation_1_5",
    "cultural_register_1_5",
    "helpfulness_1_5",
    "uncertainty_behavior_1_5",
)
FLAG_COLUMNS = ("hallucination_flag", "repetition_flag")
MORPHOLOGY_CATEGORY = "Morphology and grammar"


def summarize_native_review(
    review_csv: str | Path,
    key_path: str | Path,
    *,
    baseline_candidate_id: str,
    morphology_pass_rate: float = 0.7,
    morphology_correctness: float = 4.0,
    minimum_candidates_for_cpt: int = 2,
    report_title: str = "Normalized Native-Review Results",
    scope: str = "Native-speaker scores only; no model-generated quality grades.",
) -> dict[str, Any]:
    """Aggregate only complete human scores and compare them with the base."""

    if not 0 <= morphology_pass_rate <= 1:
        raise ValueError("morphology_pass_rate must be between 0 and 1")
    if not 1 <= morphology_correctness <= 5:
        raise ValueError("morphology_correctness must be between 1 and 5")
    if minimum_candidates_for_cpt < 2:
        raise ValueError("minimum_candidates_for_cpt must be at least 2")
    if not report_title.strip() or not scope.strip():
        raise ValueError("report_title and scope must not be blank")

    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    key_rows = key.get("rows") if isinstance(key, dict) else None
    if not isinstance(key_rows, list) or not key_rows:
        raise ValueError("blind key must contain non-empty rows")
    key_by_id: dict[str, dict[str, Any]] = {}
    for row in key_rows:
        blind_id = str(row.get("blind_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not blind_id or not candidate_id:
            raise ValueError("blind key rows require blind_id and candidate_id")
        if blind_id in key_by_id:
            raise ValueError(f"duplicate blind key id: {blind_id}")
        key_by_id[blind_id] = row

    with Path(review_csv).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "blind_id",
            "category",
            "prompt_validity",
            "pass_fail",
            "reviewer",
            "reviewer_notes",
            "corrected_response",
            *SCORE_COLUMNS,
            *FLAG_COLUMNS,
        }
        missing_columns = sorted(required_columns.difference(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(
                "review CSV is missing columns: " + ", ".join(missing_columns)
            )
        review_rows = list(reader)

    review_by_id: dict[str, dict[str, str]] = {}
    for row in review_rows:
        blind_id = row.get("blind_id", "").strip()
        if not blind_id:
            raise ValueError("review row has no blind_id")
        if blind_id in review_by_id:
            raise ValueError(f"duplicate review id: {blind_id}")
        if blind_id not in key_by_id:
            raise ValueError(f"review id is not present in blind key: {blind_id}")
        review_by_id[blind_id] = row
    missing_review_ids = sorted(set(key_by_id).difference(review_by_id))
    if missing_review_ids:
        raise ValueError(
            "blind review CSV is missing rows: " + ", ".join(missing_review_ids[:5])
        )

    candidate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incomplete: list[dict[str, str]] = []
    for blind_id, key_row in key_by_id.items():
        row = review_by_id[blind_id]
        candidate_id = str(key_row["candidate_id"])
        validity = row["prompt_validity"].strip().casefold()
        reviewer = row["reviewer"].strip()
        if validity not in {"valid", "invalid"}:
            incomplete.append({"blind_id": blind_id, "reason": "prompt_validity"})
            continue
        if not reviewer:
            incomplete.append({"blind_id": blind_id, "reason": "reviewer"})
            continue
        if validity == "invalid":
            if not row["reviewer_notes"].strip():
                incomplete.append(
                    {"blind_id": blind_id, "reason": "invalid_prompt_notes"}
                )
                continue
            candidate_rows[candidate_id].append(
                {
                    "blind_id": blind_id,
                    "task_id": row.get("task_id", "").strip(),
                    "category": row["category"].strip(),
                    "valid": False,
                }
            )
            continue

        try:
            scores = {
                column: _rating(row[column], blind_id) for column in SCORE_COLUMNS
            }
            flags = {
                column: _yes_no(row[column], column, blind_id)
                for column in FLAG_COLUMNS
            }
            pass_fail = row["pass_fail"].strip().casefold()
            if pass_fail not in {"pass", "fail"}:
                raise ValueError(f"{blind_id}: pass_fail must be pass or fail")
        except ValueError as exc:
            incomplete.append({"blind_id": blind_id, "reason": str(exc)})
            continue

        candidate_rows[candidate_id].append(
            {
                "blind_id": blind_id,
                "task_id": row.get("task_id", "").strip(),
                "category": row["category"].strip(),
                "valid": True,
                "scores": scores,
                "flags": flags,
                "pass": pass_fail == "pass",
                "has_correction": bool(row["corrected_response"].strip()),
            }
        )

    expected_by_candidate: dict[str, int] = defaultdict(int)
    for row in key_rows:
        expected_by_candidate[str(row["candidate_id"])] += 1
    candidate_ids = sorted(expected_by_candidate)
    if baseline_candidate_id not in expected_by_candidate:
        raise ValueError(
            f"baseline candidate is not present in blind key: {baseline_candidate_id}"
        )

    candidates = {
        candidate_id: _candidate_summary(
            candidate_rows.get(candidate_id, []),
            expected=expected_by_candidate[candidate_id],
        )
        for candidate_id in candidate_ids
    }
    complete = not incomplete and all(
        summary["completed_rows"] == summary["expected_rows"]
        for summary in candidates.values()
    )
    decisions = _compare_with_baseline(candidates, baseline_candidate_id)
    paired_comparisons = _paired_comparisons(
        candidate_rows, baseline_candidate_id, candidate_ids
    )
    cpt_decision = _cpt_decision(
        candidates,
        complete=complete,
        pass_rate_threshold=morphology_pass_rate,
        correctness_threshold=morphology_correctness,
        minimum_candidates=minimum_candidates_for_cpt,
    )
    return {
        "schema_version": 2,
        "report_title": report_title,
        "scope": scope,
        "complete": complete,
        "expected_rows": len(key_rows),
        "completed_rows": len(key_rows) - len(incomplete),
        "incomplete_rows": incomplete,
        "baseline_candidate_id": baseline_candidate_id,
        "candidates": candidates,
        "comparisons_to_baseline": paired_comparisons,
        "candidate_decisions": decisions,
        "continued_pretraining": cpt_decision,
    }


def render_native_review_markdown(summary: dict[str, Any]) -> str:
    """Render a compact normalized base-versus-candidate evaluation report."""

    completed = int(summary["completed_rows"])
    expected = int(summary["expected_rows"])
    lines = [
        f"# {summary.get('report_title', 'Normalized Native-Review Results')}",
        "",
        f"Review progress: **{completed}/{expected} rows** "
        f"({'complete' if summary['complete'] else 'incomplete'}).",
        "",
        f"Scope: {summary['scope']}",
        "",
        (
            "Pass percentages use valid reviewed prompts. Model improvements use "
            "only prompts with complete valid reviews for both the base and the "
            "candidate, so every comparison is paired."
        ),
        "",
        "## Candidate Scores",
        "",
        (
            "| Candidate | Valid prompts | Passes | Pass rate | "
            "Macro category rate | Repetition | Hallucination |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate_id, candidate in sorted(summary["candidates"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate_id,
                    str(candidate["valid_prompt_rows"]),
                    str(candidate["pass_count"]),
                    _format_percent(candidate["pass_rate_percent"]),
                    _format_percent(candidate["macro_category_pass_rate_percent"]),
                    _format_percent(candidate["repetition_rate_percent"]),
                    _format_percent(candidate["hallucination_rate_percent"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired Improvement Over Base",
            "",
            (
                "| Candidate | Paired prompts | Base | Candidate | Gain | "
                "Recovered | Regressed | Error reduction | 95% CI | McNemar p |"
            ),
            (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |"
            ),
        ]
    )
    for candidate_id, comparison in sorted(
        summary["comparisons_to_baseline"].items()
    ):
        if comparison["status"] != "complete":
            lines.append(
                f"| {candidate_id} | {comparison['paired_valid_prompt_count']} | "
                "- | - | pending | - | - | - | - | - |"
            )
            continue
        interval = comparison["bootstrap_95_ci_percentage_points"]
        interval_text = (
            f"{interval['lower']:+.2f} to {interval['upper']:+.2f} pp"
            if interval
            else "-"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate_id,
                    str(comparison["paired_valid_prompt_count"]),
                    _format_percent(comparison["baseline_pass_rate_percent"]),
                    _format_percent(comparison["candidate_pass_rate_percent"]),
                    f"{comparison['absolute_improvement_percentage_points']:+.2f} pp",
                    str(comparison["recovered_baseline_failures"]),
                    str(comparison["new_regressions"]),
                    _format_percent(comparison["relative_error_reduction_percent"]),
                    interval_text,
                    str(comparison["mcnemar_exact_two_sided_p"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- **Gain** is the candidate pass rate minus the base pass rate "
                "in percentage points."
            ),
            (
                "- **Recovered** means the base failed and the candidate passed "
                "the same prompt."
            ),
            (
                "- **Regressed** means the base passed and the candidate failed "
                "the same prompt."
            ),
            (
                "- **Error reduction** measures how much of the base model's "
                "failure rate was removed."
            ),
            (
                "- The confidence interval and McNemar test reflect the paired "
                "prompt outcomes; they do not replace native-speaker judgment."
            ),
            "",
        ]
    )
    if not summary["complete"]:
        lines.append(
            "Final model claims remain pending until the blind review is complete."
        )
        lines.append("")
    return "\n".join(lines)


def _candidate_summary(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    invalid = [row for row in rows if not row["valid"]]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        categories[row["category"]].append(row)
    pass_count = sum(row["pass"] for row in valid)
    pass_rate = _mean([float(row["pass"]) for row in valid])
    category_summaries = {
        category: {
            "rows": len(category_rows),
            "pass_count": sum(row["pass"] for row in category_rows),
            "pass_rate": _mean([float(row["pass"]) for row in category_rows]),
            "pass_rate_percent": _percent(
                _mean([float(row["pass"]) for row in category_rows])
            ),
            "kinyarwanda_correctness_mean": _mean(
                [
                    row["scores"]["kinyarwanda_correctness_1_5"]
                    for row in category_rows
                ]
            ),
        }
        for category, category_rows in sorted(categories.items())
    }
    return {
        "expected_rows": expected,
        "completed_rows": len(rows),
        "valid_prompt_rows": len(valid),
        "invalid_prompt_rows": len(invalid),
        "pass_count": pass_count,
        "failed_rows": len(valid) - pass_count,
        "pass_rate": pass_rate,
        "pass_rate_percent": _percent(pass_rate),
        "macro_category_pass_rate_percent": _percent(
            _mean(
                [
                    summary["pass_rate"]
                    for summary in category_summaries.values()
                    if summary["pass_rate"] is not None
                ]
            )
        ),
        "mean_scores": {
            column: _mean([row["scores"][column] for row in valid])
            for column in SCORE_COLUMNS
        },
        "hallucination_count": sum(
            row["flags"]["hallucination_flag"] for row in valid
        ),
        "hallucination_rate_percent": _percent(
            _mean(
                [float(row["flags"]["hallucination_flag"]) for row in valid]
            )
        ),
        "repetition_count": sum(row["flags"]["repetition_flag"] for row in valid),
        "repetition_rate_percent": _percent(
            _mean([float(row["flags"]["repetition_flag"]) for row in valid])
        ),
        "failed_rows_with_correction": sum(
            not row["pass"] and row["has_correction"] for row in valid
        ),
        "categories": category_summaries,
    }


def _paired_comparisons(
    candidate_rows: dict[str, list[dict[str, Any]]],
    baseline_id: str,
    candidate_ids: list[str],
) -> dict[str, dict[str, Any]]:
    baseline_by_task = _valid_rows_by_task(candidate_rows.get(baseline_id, []))
    comparisons: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        if candidate_id == baseline_id:
            continue
        rows = candidate_rows.get(candidate_id, [])
        candidate_by_task = _valid_rows_by_task(rows)
        task_ids = sorted(set(baseline_by_task) & set(candidate_by_task))
        pairs = [
            (baseline_by_task[task_id], candidate_by_task[task_id])
            for task_id in task_ids
        ]
        comparisons[candidate_id] = _paired_comparison_summary(pairs)
    return comparisons


def _valid_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id", "")
        if not row.get("valid") or not task_id:
            continue
        if task_id in by_task:
            raise ValueError(f"duplicate valid review row for task: {task_id}")
        by_task[task_id] = row
    return by_task


def _paired_comparison_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    if not pairs:
        return {
            "status": "pending",
            "paired_valid_prompt_count": 0,
            "reason": "no prompts have complete valid reviews for both models",
        }

    baseline_passes = sum(base["pass"] for base, _ in pairs)
    candidate_passes = sum(candidate["pass"] for _, candidate in pairs)
    recovered = sum(
        not base["pass"] and candidate["pass"] for base, candidate in pairs
    )
    regressions = sum(
        base["pass"] and not candidate["pass"] for base, candidate in pairs
    )
    both_pass = sum(base["pass"] and candidate["pass"] for base, candidate in pairs)
    both_fail = sum(
        not base["pass"] and not candidate["pass"] for base, candidate in pairs
    )
    count = len(pairs)
    baseline_rate = baseline_passes / count
    candidate_rate = candidate_passes / count
    difference = candidate_rate - baseline_rate
    baseline_error_rate = 1 - baseline_rate

    categories: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for pair in pairs:
        categories[pair[0]["category"]].append(pair)
    category_deltas = {}
    for category, category_pairs in sorted(categories.items()):
        category_count = len(category_pairs)
        base_rate = sum(base["pass"] for base, _ in category_pairs) / category_count
        candidate_category_rate = (
            sum(candidate["pass"] for _, candidate in category_pairs) / category_count
        )
        category_deltas[category] = {
            "paired_prompt_count": category_count,
            "baseline_pass_rate_percent": _percent(base_rate),
            "candidate_pass_rate_percent": _percent(candidate_category_rate),
            "difference_percentage_points": round(
                (candidate_category_rate - base_rate) * 100, 2
            ),
        }

    score_deltas = {
        column: round(
            statistics.mean(
                candidate["scores"][column] - base["scores"][column]
                for base, candidate in pairs
            ),
            4,
        )
        for column in SCORE_COLUMNS
    }
    return {
        "status": "complete",
        "paired_valid_prompt_count": count,
        "baseline_pass_count": baseline_passes,
        "candidate_pass_count": candidate_passes,
        "baseline_pass_rate_percent": _percent(baseline_rate),
        "candidate_pass_rate_percent": _percent(candidate_rate),
        "absolute_improvement_percentage_points": round(difference * 100, 2),
        "relative_pass_rate_change_percent": (
            round(difference / baseline_rate * 100, 2) if baseline_rate else None
        ),
        "relative_error_reduction_percent": (
            round(
                (baseline_error_rate - (1 - candidate_rate))
                / baseline_error_rate
                * 100,
                2,
            )
            if baseline_error_rate
            else None
        ),
        "recovered_baseline_failures": recovered,
        "new_regressions": regressions,
        "net_improved_prompts": recovered - regressions,
        "both_passed": both_pass,
        "both_failed": both_fail,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(recovered, regressions),
        "bootstrap_95_ci_percentage_points": _bootstrap_paired_ci(
            [
                float(candidate["pass"]) - float(base["pass"])
                for base, candidate in pairs
            ]
        ),
        "mean_score_deltas": score_deltas,
        "category_comparisons": category_deltas,
    }


def _mcnemar_exact(recovered: int, regressions: int) -> float:
    discordant = recovered + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(recovered, regressions) + 1)
    ) / (2**discordant)
    return round(min(1.0, 2 * tail), 6)


def _bootstrap_paired_ci(
    differences: list[float], *, samples: int = 10_000
) -> dict[str, Any] | None:
    if len(differences) < 2:
        return None
    seed_material = ",".join(str(value) for value in differences).encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)
    count = len(differences)
    means = sorted(
        statistics.mean(rng.choice(differences) for _ in range(count))
        for _ in range(samples)
    )
    return {
        "lower": round(means[int(0.025 * (samples - 1))] * 100, 2),
        "upper": round(means[int(0.975 * (samples - 1))] * 100, 2),
        "bootstrap_samples": samples,
    }


def _compare_with_baseline(
    candidates: dict[str, dict[str, Any]], baseline_id: str
) -> dict[str, dict[str, Any]]:
    baseline = candidates[baseline_id]
    decisions: dict[str, dict[str, Any]] = {}
    for candidate_id, summary in candidates.items():
        if candidate_id == baseline_id:
            decisions[candidate_id] = {
                "decision": "reference",
                "reasons": ["selected base comparison"],
            }
            continue
        if summary["completed_rows"] != summary["expected_rows"]:
            decisions[candidate_id] = {
                "decision": "pending",
                "reasons": ["native review is incomplete"],
            }
            continue
        reasons = []
        if _less(summary["pass_rate"], baseline["pass_rate"]):
            reasons.append("pass rate is below base")
        if _less(
            summary["mean_scores"]["kinyarwanda_correctness_1_5"],
            baseline["mean_scores"]["kinyarwanda_correctness_1_5"],
        ):
            reasons.append("Kinyarwanda correctness is below base")
        if summary["repetition_count"] > baseline["repetition_count"]:
            reasons.append("repetition count is above base")
        if summary["hallucination_count"] > baseline["hallucination_count"]:
            reasons.append("hallucination count is above base")
        decisions[candidate_id] = {
            "decision": "reject" if reasons else "continue",
            "reasons": reasons or ["meets or exceeds all base gates"],
        }
    return decisions


def _cpt_decision(
    candidates: dict[str, dict[str, Any]],
    *,
    complete: bool,
    pass_rate_threshold: float,
    correctness_threshold: float,
    minimum_candidates: int,
) -> dict[str, Any]:
    thresholds = {
        "morphology_pass_rate": pass_rate_threshold,
        "morphology_correctness_mean": correctness_threshold,
        "minimum_candidates": minimum_candidates,
    }
    if not complete or len(candidates) < minimum_candidates:
        return {
            "decision": "pending",
            "reason": "complete native reviews for at least two candidates",
            "thresholds": thresholds,
        }
    failures = []
    for summary in candidates.values():
        morphology = summary["categories"].get(MORPHOLOGY_CATEGORY)
        failures.append(
            not morphology
            or _less(morphology["pass_rate"], pass_rate_threshold)
            or _less(
                morphology["kinyarwanda_correctness_mean"], correctness_threshold
            )
        )
    return {
        "decision": "consider-cpt" if all(failures) else "continue-sft",
        "reason": (
            "every reviewed candidate is below the morphology gate"
            if all(failures)
            else "at least one candidate meets the morphology gate"
        ),
        "thresholds": thresholds,
    }


def _rating(value: str, blind_id: str) -> int:
    try:
        rating = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{blind_id}: score must be an integer from 1 to 5") from exc
    if not 1 <= rating <= 5:
        raise ValueError(f"{blind_id}: score must be between 1 and 5")
    return rating


def _yes_no(value: str, column: str, blind_id: str) -> bool:
    normalized = value.strip().casefold()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"{blind_id}: {column} must be yes or no")
    return normalized == "yes"


def _mean(values: list[float | int]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _percent(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def _format_percent(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "-"


def _less(left: float | None, right: float | None) -> bool:
    return left is None or right is None or left < right

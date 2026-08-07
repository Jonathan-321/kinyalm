"""Validate and summarize blind native-speaker model reviews."""

from __future__ import annotations

import csv
import json
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
) -> dict[str, Any]:
    """Aggregate only complete human scores and compare them with the base."""

    if not 0 <= morphology_pass_rate <= 1:
        raise ValueError("morphology_pass_rate must be between 0 and 1")
    if not 1 <= morphology_correctness <= 5:
        raise ValueError("morphology_correctness must be between 1 and 5")
    if minimum_candidates_for_cpt < 2:
        raise ValueError("minimum_candidates_for_cpt must be at least 2")

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
    cpt_decision = _cpt_decision(
        candidates,
        complete=complete,
        pass_rate_threshold=morphology_pass_rate,
        correctness_threshold=morphology_correctness,
        minimum_candidates=minimum_candidates_for_cpt,
    )
    return {
        "schema_version": 1,
        "scope": "Native-speaker scores only; no model-generated quality grades.",
        "complete": complete,
        "expected_rows": len(key_rows),
        "completed_rows": len(key_rows) - len(incomplete),
        "incomplete_rows": incomplete,
        "baseline_candidate_id": baseline_candidate_id,
        "candidates": candidates,
        "candidate_decisions": decisions,
        "continued_pretraining": cpt_decision,
    }


def _candidate_summary(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    invalid = [row for row in rows if not row["valid"]]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        categories[row["category"]].append(row)
    return {
        "expected_rows": expected,
        "completed_rows": len(rows),
        "valid_prompt_rows": len(valid),
        "invalid_prompt_rows": len(invalid),
        "pass_rate": _mean([float(row["pass"]) for row in valid]),
        "mean_scores": {
            column: _mean([row["scores"][column] for row in valid])
            for column in SCORE_COLUMNS
        },
        "hallucination_count": sum(
            row["flags"]["hallucination_flag"] for row in valid
        ),
        "repetition_count": sum(row["flags"]["repetition_flag"] for row in valid),
        "failed_rows": sum(not row["pass"] for row in valid),
        "failed_rows_with_correction": sum(
            not row["pass"] and row["has_correction"] for row in valid
        ),
        "categories": {
            category: {
                "rows": len(category_rows),
                "pass_rate": _mean([float(row["pass"]) for row in category_rows]),
                "kinyarwanda_correctness_mean": _mean(
                    [
                        row["scores"]["kinyarwanda_correctness_1_5"]
                        for row in category_rows
                    ]
                ),
            }
            for category, category_rows in sorted(categories.items())
        },
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


def _less(left: float | None, right: float | None) -> bool:
    return left is None or right is None or left < right

import csv

import pytest

from scripts.score_blind_review_gemini import (
    RequestRateLimiter,
    build_batch_prompt,
    build_prompt,
    reviewer_name,
    validate_score,
    write_scored_csv,
)


def valid_score():
    return {
        "prompt_validity": "valid",
        "kinyarwanda_correctness_1_5": 4,
        "beginner_clarity_1_5": 4,
        "grammar_explanation_1_5": 4,
        "cultural_register_1_5": 4,
        "helpfulness_1_5": 4,
        "uncertainty_behavior_1_5": 4,
        "hallucination_flag": "no",
        "repetition_flag": "no",
        "pass_fail": "pass",
        "failure_tags": "",
        "rewrite_priority": "none",
        "corrected_response": "",
        "reviewer_notes": "Correct and natural.",
    }


def test_prompt_does_not_include_model_label():
    prompt = build_prompt(
        {
            "model_label": "Model A",
            "category": "Translation",
            "review_focus": "accuracy",
            "prompt": "Translate this.",
            "response": "Hindura ibi.",
        }
    )
    assert "Model A" not in prompt
    assert "Translate this." in prompt


def test_batch_prompt_does_not_include_model_label():
    prompt = build_batch_prompt(
        [
            {
                "blind_id": "B001",
                "model_label": "Model A",
                "category": "Translation",
                "review_focus": "accuracy",
                "prompt": "Translate this.",
                "response": "Hindura ibi.",
            }
        ]
    )
    assert "Model A" not in prompt
    assert '"blind_id": "B001"' in prompt


def test_score_validation_adds_preliminary_reviewer():
    score = validate_score(valid_score())
    assert score["pass_fail"] == "pass"
    assert score["reviewer"] == "Gemini 3.1 Pro preliminary judge"
    with pytest.raises(ValueError, match="missing fields"):
        validate_score({"pass_fail": "pass"})


def test_score_validation_uses_selected_model_name():
    score = validate_score(
        valid_score(),
        reviewer=reviewer_name("gemini-3-flash-preview"),
    )
    assert score["reviewer"] == "Gemini 3 Flash preliminary judge"


def test_scored_output_does_not_modify_source_rows(tmp_path):
    fields = ["blind_id", "pass_fail", "reviewer"]
    source = [{"blind_id": "B001", "pass_fail": "", "reviewer": ""}]
    output = tmp_path / "scored.csv"

    write_scored_csv(
        source,
        fields,
        {"B001": {"pass_fail": "pass", "reviewer": "judge"}},
        output,
    )

    row = next(csv.DictReader(output.open(encoding="utf-8")))
    assert row["pass_fail"] == "pass"
    assert source[0]["pass_fail"] == ""


def test_rate_limiter_rejects_invalid_limit():
    with pytest.raises(ValueError, match="positive"):
        RequestRateLimiter(0)

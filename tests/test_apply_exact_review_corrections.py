import json

import pytest

from scripts.apply_exact_review_corrections import apply_corrections


def row():
    return {
        "conversation_id": "KINYA-SFT-0001",
        "messages_json": json.dumps(
            [
                {"role": "user", "content": "Amakuru?"},
                {"role": "assistant", "content": "Nmeze neza."},
            ]
        ),
        "review_status": "approved",
        "corrected_messages_json": "",
        "reviewer_notes": "Native review complete.",
    }


def specification(expected="Nmeze neza."):
    return {
        "authorized_by": "Jonathan Muhire",
        "authorized_on": "2026-08-09",
        "corrections": [
            {
                "conversation_id": "KINYA-SFT-0001",
                "message_index": 1,
                "expected_text": expected,
                "replacement_text": "Meze neza.",
                "reason": "Correct typo.",
            }
        ],
    }


def test_applies_only_the_exact_message_and_records_provenance():
    rows, applied = apply_corrections([row()], specification())

    corrected = json.loads(rows[0]["corrected_messages_json"])
    assert corrected[0]["content"] == "Amakuru?"
    assert corrected[1]["content"] == "Meze neza."
    assert rows[0]["review_status"] == "corrected"
    assert "authorized by Jonathan Muhire" in rows[0]["reviewer_notes"]
    assert applied[0]["before"] == "Nmeze neza."


def test_refuses_to_guess_when_expected_text_does_not_match():
    with pytest.raises(ValueError, match="does not exactly match"):
        apply_corrections([row()], specification(expected="Other text."))

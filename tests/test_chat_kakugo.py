import argparse

import pytest

from scripts.local.chat_kakugo import (
    CONCISE_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    extract_visible_response,
    positive_int,
    set_reasoning,
    system_prompt,
    trim_history,
)


def test_extract_visible_response_removes_reasoning_block():
    visible, reasoning = extract_visible_response(
        "<think>Check the grammar.</think>Igisubizo cya nyuma."
    )

    assert visible == "Igisubizo cya nyuma."
    assert reasoning == "Check the grammar."


def test_extract_visible_response_detects_unfinished_reasoning():
    visible, reasoning = extract_visible_response(
        "<think>The model used its entire token budget"
    )

    assert visible == ""
    assert reasoning == "The model used its entire token budget"


def test_extract_visible_response_keeps_plain_answer():
    visible, reasoning = extract_visible_response("Muraho neza!")

    assert visible == "Muraho neza!"
    assert reasoning is None


def test_trim_history_keeps_system_and_latest_complete_turns():
    messages = [{"role": "system", "content": "system"}]
    for index in range(4):
        messages.extend(
            [
                {"role": "user", "content": f"user-{index}"},
                {"role": "assistant", "content": f"assistant-{index}"},
            ]
        )

    trimmed = trim_history(messages, max_turns=2)

    assert trimmed == [messages[0], *messages[-4:]]


def test_system_prompt_matches_model_card_modes():
    assert system_prompt(False) == CONCISE_SYSTEM_PROMPT
    assert system_prompt(True) == REASONING_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("command", "current", "expected"),
    [
        ("/reasoning", False, True),
        ("/reasoning on", False, True),
        ("/reasoning off", True, False),
    ],
)
def test_set_reasoning(command: str, current: bool, expected: bool):
    enabled, error = set_reasoning(command, current)

    assert enabled is expected
    assert error is None


def test_set_reasoning_rejects_unknown_value():
    enabled, error = set_reasoning("/reasoning maybe", False)

    assert enabled is False
    assert error is not None


def test_positive_int_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")

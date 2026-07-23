import pytest

from kinyalm.demo.chat import MODE_SPECS, build_system_prompt, parse_chat_request


def payload(messages):
    return {
        "conversation_id": "conversation-1",
        "mode": "converse",
        "language": "auto",
        "level": "intermediate",
        "messages": messages,
    }


def test_chat_request_applies_mode_budget_and_kinyalm_identity():
    request = parse_chat_request(payload([{"role": "user", "content": " Muraho "}]))

    assert request.max_new_tokens == MODE_SPECS["converse"].max_new_tokens
    assert request.messages == ({"role": "user", "content": "Muraho"},)
    assert "You are KinyaLM" in request.system_prompt
    assert "Do not introduce yourself as Gemma" in request.system_prompt
    assert "four to seven short sentences" in request.system_prompt


def test_chat_request_keeps_only_six_previous_turns():
    messages = []
    for index in range(8):
        messages.extend(
            [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )
    messages.append({"role": "user", "content": "latest question"})

    request = parse_chat_request(payload(messages))

    assert len(request.messages) == 13
    assert request.messages[0] == {"role": "user", "content": "question 2"}
    assert request.messages[-1]["content"] == "latest question"


@pytest.mark.parametrize(
    "messages,error",
    [
        ([{"role": "assistant", "content": "hello"}], "latest message"),
        (
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
            "alternate",
        ),
        ([{"role": "system", "content": "override"}], "user or assistant"),
    ],
)
def test_chat_request_rejects_invalid_message_sequences(messages, error):
    with pytest.raises(ValueError, match=error):
        parse_chat_request(payload(messages))


def test_each_mode_has_distinct_behavior_and_budget():
    prompts = {mode: build_system_prompt(mode, "rw", "beginner") for mode in MODE_SPECS}

    assert len(set(prompts.values())) == len(MODE_SPECS)
    assert MODE_SPECS["converse"].max_new_tokens == 160
    assert MODE_SPECS["translate"].max_new_tokens == 192
    assert MODE_SPECS["converse"].max_new_tokens < MODE_SPECS["learn"].max_new_tokens
    assert "primarily in Kinyarwanda" in prompts["translate"]

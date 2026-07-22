from scripts.local.chat_gemma4_12b import conversation_messages


def test_conversation_messages_preserves_history_and_adds_system_and_user():
    history = [
        {"role": "user", "content": "Muraho"},
        {"role": "assistant", "content": "Muraho neza"},
    ]

    messages = conversation_messages("Teach clearly.", history, "Amakuru?")

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == "Amakuru?"
    assert len(history) == 2

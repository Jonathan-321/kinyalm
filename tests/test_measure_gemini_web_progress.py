from scripts.measure_gemini_web_progress import measure_progress


def test_measures_only_assistant_messages_and_combines_existing_tokens():
    rows = [
        {
            "conversation_id": "one",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "one two three"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "four five"},
            ],
        },
        {
            "conversation_id": "two",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "six"},
            ],
        },
    ]

    report = measure_progress(
        rows,
        token_counter=lambda text: len(text.split()),
        completed_web_jobs=2,
        existing_assistant_tokens=4,
        target_assistant_tokens=20,
    )

    assert report["assistant_response_count"] == 3
    assert report["multi_turn_conversation_count"] == 1
    assert report["new_assistant_tokens"] == 6
    assert report["combined_assistant_tokens"] == 10
    assert report["remaining_assistant_tokens"] == 10
    assert report["progress_percent"] == 50.0

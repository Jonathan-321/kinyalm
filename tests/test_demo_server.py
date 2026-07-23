import json
import threading
from urllib.request import Request, urlopen

from kinyalm.demo.server import (
    ChatApplication,
    FeedbackStore,
    RuntimeState,
    create_server,
)


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.call_thread_ids = []
        self.closed = False
        self.closed_thread_id = None

    def generate_messages(
        self,
        *,
        messages,
        max_new_tokens,
        enable_thinking,
        on_text=None,
    ):
        self.call_thread_ids.append(threading.get_ident())
        self.calls.append((messages, max_new_tokens, enable_thinking))
        for chunk in ("Muraho ", "neza."):
            if on_text:
                on_text(chunk)
        return {
            "response": "Muraho neza.",
            "input_tokens": 42,
            "output_tokens": 4,
            "prompt_tokens_per_second": 200.0,
            "tokens_per_second": 8.0,
            "first_token_seconds": 0.4,
            "latency_seconds": 0.9,
            "peak_unified_memory_gb": 11.2,
            "finish_reason": "stop",
        }

    def close(self):
        self.closed = True
        self.closed_thread_id = threading.get_ident()


def chat_payload():
    return {
        "conversation_id": "conversation-1",
        "mode": "converse",
        "language": "auto",
        "level": "intermediate",
        "messages": [{"role": "user", "content": "Muraho"}],
    }


def ready_application(tmp_path):
    runtime = FakeRuntime()
    state = RuntimeState()
    state.set_ready(runtime, {"name": "KinyaLM", "location": "On this Mac"})
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>KinyaLM</h1>", encoding="utf-8")
    application = ChatApplication(
        runtime=state,
        feedback=FeedbackStore(tmp_path / "feedback"),
        static_dir=static,
    )
    return application, state, runtime


def test_stream_chat_emits_start_deltas_and_final_metrics(tmp_path):
    application, state, runtime = ready_application(tmp_path)
    events = []
    request_thread_id = threading.get_ident()

    try:
        application.stream_chat(chat_payload(), events.append)
    finally:
        state.close()

    assert [event["type"] for event in events] == [
        "start",
        "delta",
        "delta",
        "done",
    ]
    assert events[-1]["response"] == "Muraho neza."
    assert events[-1]["metrics"]["tokens_per_second"] == 8.0
    messages, max_tokens, thinking = runtime.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "Muraho"
    assert max_tokens == 160
    assert thinking is False
    assert runtime.call_thread_ids[0] != request_thread_id
    assert runtime.closed_thread_id == runtime.call_thread_ids[0]


def test_feedback_is_private_jsonl(tmp_path):
    store = FeedbackStore(tmp_path / "feedback")
    feedback_id = store.append(
        {
            "conversation_id": "conversation-1",
            "message_id": "message-1",
            "user_prompt": "Muraho",
            "response": "Hello",
            "correction": "Muraho neza",
            "rating": "down",
            "mode": "translate",
            "language": "rw",
            "level": "beginner",
        }
    )

    paths = list((tmp_path / "feedback").glob("*.jsonl"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["id"] == feedback_id
    assert record["correction"] == "Muraho neza"
    assert paths[0].stat().st_mode & 0o777 == 0o600


def test_http_server_serves_ui_health_and_stream(tmp_path):
    application, state, _ = ready_application(tmp_path)
    server = create_server(application, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "KinyaLM" in urlopen(base, timeout=2).read().decode()
        health = json.loads(urlopen(f"{base}/api/health", timeout=2).read())
        assert health["status"] == "ready"

        request = Request(
            f"{base}/api/chat",
            data=json.dumps(chat_payload()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        lines = urlopen(request, timeout=2).read().decode().splitlines()
        events = [json.loads(line) for line in lines]
        assert events[-1]["type"] == "done"
        assert events[-1]["response"] == "Muraho neza."
    finally:
        server.shutdown()
        server.server_close()
        state.close()
        thread.join(timeout=2)


def test_static_file_rejects_path_traversal(tmp_path):
    application, state, _ = ready_application(tmp_path)

    try:
        assert application.static_file("/../secret.txt") is None
    finally:
        state.close()

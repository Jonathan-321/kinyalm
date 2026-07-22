"""Dependency-free local HTTP application for the KinyaLM MLX demo."""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from .chat import MODE_SPECS, ChatRequest, parse_chat_request

MAX_REQUEST_BYTES = 128 * 1024


class ChatRuntime(Protocol):
    """The small inference surface used by the browser application."""

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        enable_thinking: bool,
        on_text: Callable[[str], None] | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class _GenerationJob:
    request: ChatRequest
    on_text: Callable[[str], None]
    done: threading.Event
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class RuntimeState:
    """Thread-safe lifecycle and generation access for one resident model."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._jobs: queue.Queue[_GenerationJob | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._runtime: ChatRuntime | None = None
        self._status = "loading"
        self._error: str | None = None
        self._details: dict[str, Any] = {}

    def load_in_background(
        self,
        loader: Callable[[], tuple[ChatRuntime, dict[str, Any]]],
    ) -> threading.Thread:
        """Load and run the model on one dedicated MLX-safe thread."""

        def work() -> None:
            try:
                runtime, details = loader()
            except Exception as exc:  # pragma: no cover - exercised with real MLX
                with self._state_lock:
                    self._status = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
                return
            with self._state_lock:
                self._runtime = runtime
                self._details = dict(details)
                self._status = "ready"
                self._error = None
            self._run_jobs(runtime)

        thread = threading.Thread(
            target=work,
            name="kinyalm-model-worker",
            daemon=True,
        )
        self._worker = thread
        thread.start()
        return thread

    def set_ready(self, runtime: ChatRuntime, details: dict[str, Any]) -> None:
        """Install an already-created runtime, primarily for tests and mock mode."""

        def use_ready_runtime() -> tuple[ChatRuntime, dict[str, Any]]:
            return runtime, details

        self.load_in_background(use_ready_runtime)
        while self.health()["status"] == "loading":
            time.sleep(0.001)

    def _run_jobs(self, runtime: ChatRuntime) -> None:
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    break
                try:
                    messages = [
                        {"role": "system", "content": job.request.system_prompt},
                        *job.request.messages,
                    ]
                    job.result = runtime.generate_messages(
                        messages=messages,
                        max_new_tokens=job.request.max_new_tokens,
                        enable_thinking=False,
                        on_text=job.on_text,
                    )
                except BaseException as exc:
                    job.error = exc
                finally:
                    job.done.set()
        finally:
            runtime.close()
            with self._state_lock:
                self._runtime = None
                if self._status != "error":
                    self._status = "stopped"

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "status": self._status,
                "error": self._error,
                "runtime": dict(self._details),
                "modes": {
                    key: {
                        "label": spec.label,
                        "max_new_tokens": spec.max_new_tokens,
                    }
                    for key, spec in MODE_SPECS.items()
                },
            }

    def generate(
        self,
        request: ChatRequest,
        on_text: Callable[[str], None],
    ) -> dict[str, Any]:
        with self._state_lock:
            runtime = self._runtime
            status = self._status
        if status != "ready" or runtime is None:
            raise RuntimeError("KinyaLM is still loading")

        job = _GenerationJob(request=request, on_text=on_text, done=threading.Event())
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        if job.result is None:
            raise RuntimeError("KinyaLM generation ended without a result")
        return job.result

    def close(self) -> None:
        with self._state_lock:
            worker = self._worker
            status = self._status
            if status not in {"error", "stopped"}:
                self._status = "stopping"
        if worker is not None and worker.is_alive():
            self._jobs.put(None)
            worker.join(timeout=30)
        with self._state_lock:
            if self._status != "error":
                self._status = "stopped"


class FeedbackStore:
    """Append demo ratings and corrections to private local JSONL files."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def append(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise ValueError("feedback body must be an object")
        rating = payload.get("rating")
        if rating not in {"up", "down"}:
            raise ValueError("rating must be up or down")

        required_text = ("conversation_id", "message_id", "user_prompt", "response")
        record: dict[str, Any] = {}
        for key in required_text:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be non-empty text")
            record[key] = value.strip()[:12_000]

        correction = payload.get("correction", "")
        if not isinstance(correction, str):
            raise ValueError("correction must be text")

        feedback_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        record.update(
            {
                "id": feedback_id,
                "created_at": now.isoformat(),
                "rating": rating,
                "correction": correction.strip()[:12_000],
                "mode": str(payload.get("mode", ""))[:24],
                "language": str(payload.get("language", ""))[:24],
                "level": str(payload.get("level", ""))[:24],
                "source": "local-kinyalm-demo",
            }
        )

        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"feedback-{now.date().isoformat()}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            path.chmod(0o600)
        return feedback_id


class ChatApplication:
    """Routes HTTP requests into static assets, inference, and feedback."""

    def __init__(
        self,
        *,
        runtime: RuntimeState,
        feedback: FeedbackStore,
        static_dir: Path,
    ) -> None:
        self.runtime = runtime
        self.feedback = feedback
        self.static_dir = static_dir.resolve()

    def static_file(self, request_path: str) -> Path | None:
        relative = "index.html" if request_path == "/" else request_path.lstrip("/")
        if not relative or ".." in Path(relative).parts:
            return None
        candidate = (self.static_dir / relative).resolve()
        if self.static_dir not in candidate.parents or not candidate.is_file():
            return None
        return candidate

    def stream_chat(
        self,
        payload: Any,
        send: Callable[[dict[str, Any]], None],
    ) -> None:
        request = parse_chat_request(payload)
        request_id = str(uuid.uuid4())
        send(
            {
                "type": "start",
                "request_id": request_id,
                "max_new_tokens": request.max_new_tokens,
            }
        )
        result = self.runtime.generate(
            request,
            lambda text: send({"type": "delta", "text": text}),
        )
        metrics = {
            key: result[key]
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "prompt_tokens_per_second",
                "tokens_per_second",
                "first_token_seconds",
                "latency_seconds",
                "peak_unified_memory_gb",
                "finish_reason",
            )
            if key in result
        }
        send(
            {
                "type": "done",
                "request_id": request_id,
                "response": str(result["response"]),
                "metrics": metrics,
            }
        )


def _handler_class(application: ChatApplication) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "KinyaLM/0.1"

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'",
            )

        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            return json.loads(self.rfile.read(length))

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                self._json_response(200, application.runtime.health())
                return
            static_path = application.static_file(path)
            if static_path is None:
                self._json_response(404, {"error": "not found"})
                return
            body = static_path.read_bytes()
            content_type = mimetypes.guess_type(static_path.name)[0]
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._json_response(400, {"error": str(exc)})
                return

            if path == "/api/feedback":
                try:
                    feedback_id = application.feedback.append(payload)
                except ValueError as exc:
                    self._json_response(400, {"error": str(exc)})
                    return
                self._json_response(201, {"id": feedback_id, "saved": True})
                return

            if path != "/api/chat":
                self._json_response(404, {"error": "not found"})
                return
            if application.runtime.health()["status"] != "ready":
                self._json_response(503, {"error": "KinyaLM is still loading"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self._security_headers()
            self.end_headers()

            def send(event: dict[str, Any]) -> None:
                line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                self.wfile.write(line)
                self.wfile.flush()

            try:
                application.stream_chat(payload, send)
            except (BrokenPipeError, ConnectionResetError):
                return
            except ValueError as exc:
                send({"type": "error", "error": str(exc)})
            except Exception as exc:  # pragma: no cover - real runtime boundary
                send({"type": "error", "error": f"Generation failed: {exc}"})
            self.close_connection = True

        def log_message(self, format: str, *args: Any) -> None:
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] {self.address_string()} {format % args}")

    return RequestHandler


def create_server(
    application: ChatApplication,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    """Create, but do not start, the local threaded HTTP server."""

    return ThreadingHTTPServer((host, port), _handler_class(application))

#!/usr/bin/env python3
"""Serve the browser-based KinyaLM demo with the pinned Gemma 4 MLX runtime."""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kinyalm.demo import (  # noqa: E402
    ChatApplication,
    FeedbackStore,
    RuntimeState,
    create_server,
)
from kinyalm.evaluation import load_bakeoff_config  # noqa: E402
from scripts.run_multilingual_bakeoff import (  # noqa: E402
    DEFAULT_CONFIG,
    MlxGenerator,
    resolve_runtime_candidates,
)


class MockRuntime:
    """Small streaming runtime for interface development without loading MLX."""

    def __init__(self, delay: float = 0.015) -> None:
        self.delay = delay

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        enable_thinking: bool,
        on_text: Any = None,
    ) -> dict[str, Any]:
        del max_new_tokens, enable_thinking
        user_prompt = messages[-1]["content"]
        response = (
            "Muraho! Nishimiye kugufasha kwiga Ikinyarwanda. "
            f'Wambajije uti: "{user_prompt}"'
        )
        chunks = response.split(" ")
        started = time.perf_counter()
        for index, chunk in enumerate(chunks):
            text = chunk if index == len(chunks) - 1 else f"{chunk} "
            if on_text is not None:
                on_text(text)
            if self.delay:
                time.sleep(self.delay)
        elapsed = time.perf_counter() - started
        return {
            "response": response,
            "input_tokens": 48,
            "output_tokens": len(chunks),
            "prompt_tokens_per_second": 240.0,
            "tokens_per_second": round(len(chunks) / max(elapsed, 0.001), 2),
            "first_token_seconds": self.delay,
            "latency_seconds": round(elapsed, 3),
            "peak_unified_memory_gb": 0.0,
            "finish_reason": "stop",
        }

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feedback-dir", type=Path)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser.parse_args()


def load_real_runtime(config_path: Path) -> tuple[MlxGenerator, dict[str, Any]]:
    config = load_bakeoff_config(config_path.resolve())
    candidate = resolve_runtime_candidates(config, ["gemma4-12b-it"], "mlx")[0]
    runtime = MlxGenerator(candidate, config.seed)
    return runtime, {
        "name": "KinyaLM",
        "base_model": candidate.source_model_id,
        "checkpoint": candidate.model_id,
        "backend": f"MLX-LM {candidate.backend_version}",
        "quantization": candidate.quantization,
        "location": "On this Mac",
    }


def main() -> int:
    args = parse_args()
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")

    feedback_dir = args.feedback_dir or (
        Path.home() / ".cache" / "kinyalm" / "gemma4-12b-chat" / "feedback"
    )
    state = RuntimeState()
    if args.mock:
        state.set_ready(
            MockRuntime(),
            {
                "name": "KinyaLM",
                "base_model": "Mock streaming runtime",
                "backend": "Development mode",
                "quantization": "None",
                "location": "On this Mac",
            },
        )
    else:
        state.load_in_background(lambda: load_real_runtime(args.config))

    application = ChatApplication(
        runtime=state,
        feedback=FeedbackStore(feedback_dir),
        static_dir=ROOT / "apps" / "kinyalm-chat",
    )
    server = create_server(application, args.host, args.port)
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}"
    print(f"KinyaLM chat: {url}")
    print(f"Feedback: {feedback_dir}")
    print("Press Ctrl-C to stop.")
    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping KinyaLM...")
    finally:
        server.shutdown()
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

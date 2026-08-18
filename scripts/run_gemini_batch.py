#!/usr/bin/env python3
"""Submit, resume, and download one paid Gemini Batch API job safely."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--acknowledge-paid-batch", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--keychain-service",
        default="kinyalm-gemini-api-key",
        help="macOS Keychain service used when GEMINI_API_KEY is unset",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_immutable_json(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite different file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def submission_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another batch runner is active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def state_payload(job, **extra) -> dict:
    state = getattr(job.state, "name", str(job.state))
    value = dict(extra)
    value.update(
        {
            "job_name": job.name,
            "display_name": (
                getattr(job, "display_name", None) or value.get("display_name")
            ),
            "state": state,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    return value


def resolve_api_key(keychain_service: str | None) -> str | None:
    environment_key = os.environ.get("GEMINI_API_KEY")
    if environment_key:
        return environment_key
    if not keychain_service:
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                keychain_service,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main() -> int:
    args = parse_args()
    api_key = resolve_api_key(args.keychain_service)
    if not api_key:
        raise SystemExit(
            "Gemini key not found in GEMINI_API_KEY or macOS Keychain service "
            f"{args.keychain_service!r}"
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Install the distill extra: uv sync --extra distill") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "batch-state.json"
    intent_path = args.output_dir / "submission-intent.json"
    input_sha256 = file_sha256(args.input)
    client = genai.Client(api_key=api_key)
    with submission_lock(args.output_dir / ".submission.lock"):
        if state_path.exists():
            persistent = json.loads(state_path.read_text(encoding="utf-8"))
            for field, expected in (
                ("model", args.model),
                ("display_name", args.display_name),
                ("input_sha256", input_sha256),
            ):
                if persistent.get(field) != expected:
                    raise SystemExit(f"existing batch state has different {field}")
            job = client.batches.get(name=persistent["job_name"])
        else:
            if intent_path.exists():
                raise SystemExit(
                    "submission intent exists without batch state; refusing to "
                    "risk a duplicate paid batch. Reconcile in AI Studio first."
                )
            if not args.submit:
                raise SystemExit(
                    "No existing batch state; pass --submit to create the job"
                )
            if not args.acknowledge_paid_batch:
                raise SystemExit(
                    "Paid batch acknowledgement required: --acknowledge-paid-batch"
                )
            write_immutable_json(
                intent_path,
                {
                    "display_name": args.display_name,
                    "model": args.model,
                    "input_sha256": input_sha256,
                    "prepared_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            uploaded = client.files.upload(
                file=str(args.input),
                config=types.UploadFileConfig(
                    display_name=args.display_name + "-input",
                    mime_type="jsonl",
                ),
            )
            job = client.batches.create(
                model=args.model,
                src=uploaded.name,
                config={"display_name": args.display_name},
            )
            persistent = {
                "job_name": job.name,
                "input_file_name": uploaded.name,
                "model": args.model,
                "display_name": args.display_name,
                "input_sha256": input_sha256,
            }
            atomic_json(state_path, state_payload(job, **persistent))
            print(f"submitted batch: {job.name}")

        while True:
            state = getattr(job.state, "name", str(job.state))
            atomic_json(state_path, state_payload(job, **persistent))
            print(f"batch state: {state}")
            if state in TERMINAL_STATES or not args.wait:
                break
            time.sleep(args.poll_seconds)
            job = client.batches.get(name=job.name)

        if state != "JOB_STATE_SUCCEEDED":
            return 0 if state not in TERMINAL_STATES else 2
        if not job.dest or not job.dest.file_name:
            raise SystemExit("succeeded batch has no downloadable result file")
        output_path = args.output_dir / "gemini-batch-results.jsonl"
        content = client.files.download(file=job.dest.file_name)
        if output_path.exists() and output_path.read_bytes() != content:
            raise SystemExit("refusing to overwrite a different downloaded result")
        output_path.write_bytes(content)
        print(f"downloaded results: {output_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

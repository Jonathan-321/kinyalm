import subprocess

import pytest

from scripts.run_gemini_batch import (
    file_sha256,
    resolve_api_key,
    state_payload,
    submission_lock,
    write_immutable_json,
)


def test_api_key_prefers_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret")

    assert resolve_api_key("unused-service") == "environment-secret"


def test_api_key_can_come_from_macos_keychain(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="keychain-secret\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert resolve_api_key("kinyalm-gemini-api-key") == "keychain-secret"


def test_missing_keychain_entry_returns_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(44, args[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert resolve_api_key("kinyalm-gemini-api-key") is None


def test_immutable_json_refuses_different_content(tmp_path):
    path = tmp_path / "intent.json"
    write_immutable_json(path, {"input_sha256": "one"})
    write_immutable_json(path, {"input_sha256": "one"})

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_immutable_json(path, {"input_sha256": "two"})


def test_file_sha256_is_stable(tmp_path):
    path = tmp_path / "batch.jsonl"
    path.write_text('{"key":"one"}\n', encoding="utf-8")

    assert file_sha256(path) == file_sha256(path)


def test_submission_lock_rejects_concurrent_runner(tmp_path):
    path = tmp_path / ".submission.lock"
    with submission_lock(path):
        with pytest.raises(RuntimeError, match="another batch runner"):
            with submission_lock(path):
                pass


def test_live_batch_state_overrides_persisted_state():
    class RunningState:
        name = "JOB_STATE_RUNNING"

    class Job:
        name = "batches/one"
        display_name = "batch-one"
        state = RunningState()

    value = state_payload(Job(), state="JOB_STATE_PENDING", input_sha256="abc")

    assert value["state"] == "JOB_STATE_RUNNING"
    assert value["input_sha256"] == "abc"

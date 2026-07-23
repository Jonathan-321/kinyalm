#!/usr/bin/env python3
"""Small dependency-free terminal client for the local KinyaLM endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def complete(url: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 384,
            "repetition_penalty": 1.05,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with local experimental KinyaLM")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1/chat/completions",
    )
    args = parser.parse_args()

    messages: list[dict[str, str]] = []
    print("Experimental KinyaLM (Qwen2.5 base + Track 2 LoRA).")
    print("Enter /reset to clear the chat or /quit to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/quit":
            break
        if prompt == "/reset":
            messages.clear()
            print("Conversation cleared.")
            continue
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            answer = complete(args.url, messages)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as error:
            messages.pop()
            print(f"Could not reach KinyaLM: {error}")
            continue
        messages.append({"role": "assistant", "content": answer})
        print(f"KinyaLM: {answer}")


if __name__ == "__main__":
    main()

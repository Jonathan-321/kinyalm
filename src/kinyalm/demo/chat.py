"""Validated chat settings and prompts for the local KinyaLM demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_HISTORY_TURNS = 6
MAX_MESSAGE_CHARS = 12_000
MAX_TOTAL_CHARS = 36_000


@dataclass(frozen=True)
class ModeSpec:
    """One user-facing chat mode and its response budget."""

    label: str
    max_new_tokens: int
    instruction: str


MODE_SPECS = {
    "converse": ModeSpec(
        label="Converse",
        max_new_tokens=160,
        instruction=(
            "Have a natural conversation. For a substantive request, usually "
            "answer in four to seven short sentences; use fewer only when a "
            "brief answer is genuinely enough. Keep the response focused."
        ),
    ),
    "translate": ModeSpec(
        label="Translate / Correct",
        max_new_tokens=192,
        instruction=(
            "Give the requested translation or corrected sentence first. When "
            "helpful, add up to three concise usage, grammar, or alternative-"
            "phrasing notes. Do not turn a simple request into a long lesson."
        ),
    ),
    "learn": ModeSpec(
        label="Learn",
        max_new_tokens=256,
        instruction=(
            "Teach the requested concept in roughly six to ten short sentences "
            "when the topic supports it. Include two useful examples, explain "
            "the key pattern clearly, and ask at most one practice question."
        ),
    ),
}

LANGUAGE_INSTRUCTIONS = {
    "auto": (
        "Use the language of the learner's latest request. Use both languages "
        "only when translation or explanation requires it."
    ),
    "rw": (
        "Respond primarily in Kinyarwanda. Include English only when the learner "
        "asks for a translation or an English explanation."
    ),
    "en": (
        "Respond primarily in English. Include Kinyarwanda examples whenever "
        "they help answer the request."
    ),
}

LEVEL_INSTRUCTIONS = {
    "beginner": "Use common vocabulary and explain only the most useful detail.",
    "intermediate": "Use natural everyday language with concise corrections.",
    "advanced": "Use natural, precise language and explain meaningful nuance.",
}


@dataclass(frozen=True)
class ChatRequest:
    """A normalized request ready for the inference runtime."""

    conversation_id: str
    mode: str
    language: str
    level: str
    messages: tuple[dict[str, str], ...]
    system_prompt: str
    max_new_tokens: int


def build_system_prompt(mode: str, language: str, level: str) -> str:
    """Build concise, mode-aware KinyaLM behavior instructions."""

    spec = MODE_SPECS[mode]
    return " ".join(
        [
            "You are KinyaLM, a concise Kinyarwanda-English language tutor.",
            "Answer the learner's request directly, accurately, and naturally.",
            spec.instruction,
            LANGUAGE_INSTRUCTIONS[language],
            LEVEL_INSTRUCTIONS[level],
            (
                "Correct only material language errors. Do not invent grammar, "
                "pronunciation, cultural claims, or current facts. State "
                "uncertainty briefly when needed. Do not introduce yourself as "
                "Gemma or mention the underlying model. Use plain text with "
                "short paragraphs; avoid decorative Markdown formatting."
            ),
        ]
    )


def _required_choice(payload: dict[str, Any], key: str, choices: set[str]) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{key} must be one of: {allowed}")
    return value


def _normalize_messages(raw_messages: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")

    messages: list[dict[str, str]] = []
    total_chars = 0
    previous_role = None
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("each message must be an object")
        role = raw.get("role")
        content = raw.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError("message roles must be user or assistant")
        if role == previous_role:
            raise ValueError("message roles must alternate")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be non-empty text")
        content = content.strip()
        if len(content) > MAX_MESSAGE_CHARS:
            raise ValueError("a message is too long")
        total_chars += len(content)
        if total_chars > MAX_TOTAL_CHARS:
            raise ValueError("conversation content is too long")
        messages.append({"role": role, "content": content})
        previous_role = role

    if messages[-1]["role"] != "user":
        raise ValueError("the latest message must be from the user")

    max_messages = MAX_HISTORY_TURNS * 2 + 1
    messages = messages[-max_messages:]
    if messages and messages[0]["role"] == "assistant":
        messages.pop(0)
    return tuple(messages)


def parse_chat_request(payload: Any) -> ChatRequest:
    """Validate an API payload and apply bounded conversation history."""

    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    mode = _required_choice(payload, "mode", set(MODE_SPECS))
    language = _required_choice(payload, "language", set(LANGUAGE_INSTRUCTIONS))
    level = _required_choice(payload, "level", set(LEVEL_INSTRUCTIONS))
    conversation_id = payload.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be non-empty text")
    conversation_id = conversation_id.strip()
    if len(conversation_id) > 80:
        raise ValueError("conversation_id is too long")

    messages = _normalize_messages(payload.get("messages"))
    return ChatRequest(
        conversation_id=conversation_id,
        mode=mode,
        language=language,
        level=level,
        messages=messages,
        system_prompt=build_system_prompt(mode, language, level),
        max_new_tokens=MODE_SPECS[mode].max_new_tokens,
    )

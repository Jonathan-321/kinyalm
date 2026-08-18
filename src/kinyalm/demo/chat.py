"""Validated chat settings and prompts for the local KinyaLM demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_HISTORY_TURNS = 10
MAX_MESSAGE_CHARS = 12_000
MAX_TOTAL_CHARS = 60_000


@dataclass(frozen=True)
class ModeSpec:
    """One user-facing chat mode and its response budget."""

    label: str
    max_new_tokens: int
    instruction: str


MODE_SPECS = {
    "converse": ModeSpec(
        label="Converse",
        max_new_tokens=256,
        instruction=(
            "Be a natural conversation partner rather than turning every reply "
            "into a lesson. Respond to the latest message in light of the prior "
            "turns, add useful detail, and ask at most one natural follow-up "
            "question when it genuinely moves the conversation forward. When "
            "the learner is practicing Kinyarwanda, give the natural Kinyarwanda "
            "reply first, then add a short English learner note below only when "
            "a translation, correction, or usage explanation is useful."
        ),
    ),
    "translate": ModeSpec(
        label="Translate / Correct",
        max_new_tokens=192,
        instruction=(
            "Give the requested translation or corrected Kinyarwanda sentence "
            "first. Below it, explain the English meaning and the most useful "
            "vocabulary, grammar, or natural-usage detail. Distinguish a literal "
            "rendering from the natural translation when they differ."
        ),
    ),
    "learn": ModeSpec(
        label="Learn",
        max_new_tokens=320,
        instruction=(
            "Teach the requested concept at the selected learning level. Explain "
            "the key pattern clearly, give two natural examples when useful, and "
            "ask at most one practice question. Match the depth to the request "
            "instead of forcing every answer into the same lesson format."
        ),
    ),
}

LANGUAGE_INSTRUCTIONS = {
    "auto": (
        "Use English for explanations by default. If the learner asks for a "
        "Kinyarwanda conversation or Kinyarwanda-only answer, provide that first "
        "and place any useful English learning support below it. If the learner "
        "intentionally mixes both languages, respond naturally to that mixture."
    ),
    "rw": (
        "Respond in natural Kinyarwanda rather than translating English phrasing "
        "literally. Include English only when the learner requests it or it is "
        "needed to explain a translation."
    ),
    "en": (
        "Use English for the main explanation while preserving correct, natural "
        "Kinyarwanda in every example, quotation, correction, and translation. "
        "When answering a Kinyarwanda practice turn, put the Kinyarwanda reply "
        "first and the English support below it."
    ),
}

LEVEL_INSTRUCTIONS = {
    "beginner": (
        "Use common vocabulary, short clear sentences, and explain unfamiliar "
        "Kinyarwanda without assuming prior grammar knowledge."
    ),
    "intermediate": (
        "Use natural everyday language, explain meaningful corrections briefly, "
        "and introduce useful vocabulary without oversimplifying it."
    ),
    "advanced": (
        "Use natural, precise language and explain register, idiom, morphology, "
        "or cultural nuance when it matters."
    ),
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
    runtime_variant: str


def build_system_prompt(mode: str, language: str, level: str) -> str:
    """Build concise, mode-aware KinyaLM behavior instructions."""

    spec = MODE_SPECS[mode]
    return " ".join(
        [
            (
                "You are KinyaLM, an English-first bilingual assistant for "
                "English speakers who want to understand and use Kinyarwanda. "
                "You understand natural Kinyarwanda deeply and explain it in "
                "clear English without flattening its meaning or nuance."
            ),
            (
                "Hold natural, coherent multi-turn conversations in either "
                "language. Use the prior turns: remember names, stated "
                "preferences, the topic, and earlier corrections, and do not "
                "restart or reintroduce yourself unless the learner asks."
            ),
            (
                "Answer the actual request first, accurately and naturally. "
                "Match the amount of detail to the question; a simple greeting "
                "needs a simple reply, while a substantial question deserves a "
                "substantial answer."
            ),
            spec.instruction,
            LANGUAGE_INSTRUCTIONS[language],
            LEVEL_INSTRUCTIONS[level],
            (
                "For translation, correction, or teaching tasks, place the direct "
                "Kinyarwanda answer first. Then add only the useful English "
                "support underneath, using plain labels such as 'English meaning:', "
                "'Breakdown:', or 'Natural usage:'. In a breakdown, explain words "
                "or meaningful word parts, the relevant grammar, and why the "
                "expression sounds natural. Do not force these sections onto a "
                "simple casual reply when they add no value."
            ),
            (
                "Correct only material language errors and preserve the learner's "
                "intended meaning. Never repeat a sentence or canned phrase to "
                "fill space. Do not invent grammar, pronunciation, cultural "
                "claims, or current facts; state uncertainty briefly when needed. "
                "Do not introduce yourself as Gemma or mention the underlying "
                "model. Use readable short paragraphs and avoid decorative "
                "Markdown formatting."
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
    runtime_variant = payload.get("runtime_variant", "default")
    if runtime_variant not in {"default", "base", "targeted"}:
        raise ValueError("runtime_variant must be one of: base, default, targeted")
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
        runtime_variant=runtime_variant,
    )

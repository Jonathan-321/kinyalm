"""Data-driven generation helpers for review-only SFT draft batches."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

EXPECTED_TASK_TYPES = {
    "greeting",
    "translation-en-rw",
    "translation-rw-en",
    "grammar-explanation",
    "sentence-correction",
    "vocabulary",
    "quiz-generation",
    "dialogue",
    "uncertainty",
    "culture-register",
}
SOURCE_NOTE = "synthetic draft; fluent-speaker review required before training"


def load_generation_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML generation profile."""

    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        raise ValueError("generation profile must be a YAML object")

    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("generation profile requires a non-empty profile_id")

    target_counts = profile.get("target_counts")
    if not isinstance(target_counts, dict):
        raise ValueError("generation profile requires target_counts")
    if set(target_counts) != EXPECTED_TASK_TYPES:
        missing = sorted(EXPECTED_TASK_TYPES.difference(target_counts))
        extra = sorted(set(target_counts).difference(EXPECTED_TASK_TYPES))
        raise ValueError(
            f"target_counts task mismatch; missing={missing}, extra={extra}"
        )
    if any(not isinstance(value, int) or value < 1 for value in target_counts.values()):
        raise ValueError("every target count must be a positive integer")

    target_total = profile.get("target_total")
    if target_total != sum(target_counts.values()):
        raise ValueError(
            "target_total does not equal the sum of target_counts: "
            f"{target_total} != {sum(target_counts.values())}"
        )
    if not isinstance(profile.get("content"), dict):
        raise ValueError("generation profile requires a content object")
    return profile


def build_profile_records(
    profile: dict[str, Any],
    *,
    rng: random.Random,
    batch_code: str,
) -> list[dict[str, Any]]:
    """Expand a curated profile into a deterministic draft SFT batch."""

    content = profile["content"]
    target_counts = profile["target_counts"]
    builders: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
        "greeting": _build_greetings,
        "vocabulary": _build_vocabulary,
        "grammar-explanation": _build_grammar,
        "sentence-correction": _build_corrections,
        "dialogue": _build_dialogues,
        "quiz-generation": _build_quizzes,
        "culture-register": _build_culture_register,
        "uncertainty": _build_uncertainty,
    }

    records: list[dict[str, Any]] = []
    for task_type, builder in builders.items():
        entries = _content_entries(content, task_type)
        candidates = builder(entries)
        records.extend(_take_target(candidates, task_type, target_counts[task_type]))

    translation_pairs = _content_entries(content, "translation-pairs")
    for task_type in ("translation-en-rw", "translation-rw-en"):
        candidates = _build_translations(translation_pairs, task_type=task_type)
        records.extend(_take_target(candidates, task_type, target_counts[task_type]))

    profile_id = profile["profile_id"]
    for record in records:
        record["generation_profile"] = profile_id
        record["id"] = _record_id(record, batch_code=batch_code)

    rng.shuffle(records)
    validate_generated_profile(records, target_counts=target_counts)
    return records


def validate_generated_profile(
    records: list[dict[str, Any]],
    *,
    target_counts: dict[str, int],
) -> None:
    """Enforce profile counts, stable metadata, and conversation uniqueness."""

    actual_counts = Counter(record.get("task_type") for record in records)
    if dict(actual_counts) != target_counts:
        raise ValueError(
            f"generated task counts do not match profile: {dict(actual_counts)}"
        )

    ids = [record.get("id") for record in records]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"generated duplicate ids: {duplicate_ids[:5]}")

    conversations = [conversation_key(record) for record in records]
    duplicates = [key for key, count in Counter(conversations).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"generated duplicate prompt/answer conversations: {len(duplicates)}"
        )


def conversation_key(record: dict[str, Any]) -> tuple[str, str]:
    """Return a whitespace-normalized prompt/answer key for deduplication."""

    messages = record.get("messages", [])
    if len(messages) != 2:
        return ("", "")
    return tuple(_normalize_text(message.get("content", "")) for message in messages)  # type: ignore[return-value]


def find_conversation_collisions(
    records: list[dict[str, Any]],
    existing_records: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Find exact normalized conversations already present in older batches."""

    existing = {conversation_key(record) for record in existing_records}
    generated = {conversation_key(record) for record in records}
    return sorted(generated.intersection(existing))


def _content_entries(content: dict[str, Any], name: str) -> list[dict[str, Any]]:
    entries = content.get(name)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"profile content requires a non-empty {name} list")
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{name} entry {index} must be an object")
        key = entry.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9-]+", key):
            raise ValueError(f"{name} entry {index} has invalid key: {key!r}")
    return entries


def _take_target(
    candidates: list[dict[str, Any]], task_type: str, target: int
) -> list[dict[str, Any]]:
    if len(candidates) < target:
        raise ValueError(
            f"profile has only {len(candidates)} {task_type} candidates; needs {target}"
        )
    return candidates[:target]


def _draft_record(
    task_type: str,
    user: str,
    assistant: str,
    *,
    content_key: str,
    variant: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "pending",
        "task_type": task_type,
        "split": "draft",
        "source": "synthetic-draft",
        "source_status": "team-authored",
        "review_status": "needs-review",
        "language_mix": "kinyarwanda+english",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "reviewer_notes": SOURCE_NOTE,
        "content_key": content_key,
        "variant": variant,
    }
    if extra:
        record.update(extra)
    return record


def _record_id(record: dict[str, Any], *, batch_code: str) -> str:
    task = _slug(record["task_type"])
    content_key = _slug(record["content_key"])
    variant = _slug(record["variant"])
    return f"sftdraft-{_slug(batch_code)}-{task}-{content_key}-{variant}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"cannot create an id component from {value!r}")
    return slug


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _build_greetings(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        phrase = item["phrase"]
        meaning = item["meaning"]
        usage = item["usage"]
        exchange = item["exchange"]
        variants = [
            (
                "teach",
                f"Teach a beginner the greeting `{phrase}`.",
                f"`{phrase}` means {meaning} {usage}",
            ),
            (
                "short",
                f"Give a one-sentence explanation of `{phrase}`.",
                f"Use `{phrase}` for {usage[0].lower() + usage[1:]}",
            ),
            (
                "flashcard",
                f"Make a beginner flashcard for `{phrase}`.",
                f"Front: `{phrase}`\nBack: {meaning}\nUse: {usage}",
            ),
            (
                "exchange",
                f"Show `{phrase}` in a short exchange.",
                exchange,
            ),
            (
                "practice",
                f"Create a tiny practice task for `{phrase}`.",
                f"Meaning: {meaning}\nPractice: Say `{phrase}`, then ask the "
                "learner when they would use it.",
            ),
            (
                "check",
                f"Check a learner's understanding of `{phrase}`.",
                f"Ask: What does `{phrase}` mean?\nExpected answer: {meaning}\n"
                f"Usage note: {usage}",
            ),
        ]
        records.extend(
            _draft_record(
                "greeting",
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in variants
        )
    return records


def _build_vocabulary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        word = item["word"]
        meaning = item["meaning"]
        example_rw = item["example_rw"]
        example_en = item["example_en"]
        variants = [
            (
                "teach",
                f"Teach the Kinyarwanda word `{word}`.",
                f"`{word}` means {meaning}. Example: `{example_rw}` means {example_en}",
            ),
            (
                "flashcard",
                f"Make a flashcard for `{word}`.",
                f"Front: `{word}`\nBack: {meaning}\nExample: `{example_rw}`",
            ),
            (
                "example",
                f"Use `{word}` in a beginner sentence and translate it.",
                f"`{example_rw}`\nEnglish: {example_en}",
            ),
            (
                "recall",
                f"Create a recall question for the word `{word}`.",
                f"Question: What does `{word}` mean?\nAnswer: {meaning} "
                f"Example: `{example_rw}`",
            ),
            (
                "mini-lesson",
                f"Write a two-line vocabulary lesson for `{word}`.",
                f"Word: `{word}` = {meaning}\nModel sentence: `{example_rw}` "
                f"({example_en})",
            ),
        ]
        records.extend(
            _draft_record(
                "vocabulary",
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in variants
        )
    return records


def _build_grammar(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        focus = item["focus"]
        explanation = item["explanation"]
        example = item["example"]
        caution = item["caution"]
        variants = [
            (
                "explain",
                f"Explain the beginner grammar in `{focus}`.",
                f"{explanation} Example: `{example}` {caution}",
            ),
            (
                "wording",
                f"Give a plain-English grammar note about `{focus}`.",
                f"Grammar note: {explanation}\nModel: `{example}`\n"
                f"Review note: {caution}",
            ),
            (
                "lesson",
                f"Build a tiny lesson around `{focus}`.",
                f"Focus: `{focus}`\nExplanation: {explanation}\n"
                f"Practice model: `{example}`",
            ),
            (
                "practice",
                f"Create one guided practice step for `{focus}`.",
                f"First notice: {explanation}\nThen copy the pattern in "
                f"`{example}` and replace one content word.",
            ),
            (
                "review-aware",
                f"Teach `{focus}` without overstating the rule.",
                f"{explanation} Example: `{example}` {caution} A fluent speaker "
                "should approve the final wording.",
            ),
        ]
        records.extend(
            _draft_record(
                "grammar-explanation",
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in variants
        )
    return records


def _build_corrections(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        incorrect = item["incorrect"]
        corrected = item["corrected"]
        reason = item["reason"]
        variants = [
            (
                "direct",
                f"Correct this learner sentence: `{incorrect}`",
                f"Correction: `{corrected}`\nReason: {reason}",
            ),
            (
                "polite",
                f"Correct this gently: `{incorrect}`",
                f"A clearer beginner form is `{corrected}` {reason}",
            ),
            (
                "explain",
                f"What should change in `{incorrect}`?",
                f"Use `{corrected}` {reason}",
            ),
            (
                "tutor",
                f"How should a tutor respond when a learner writes `{incorrect}`?",
                f"The tutor can say: Try `{corrected}`\nTeaching note: {reason}",
            ),
            (
                "retry",
                f"Give a correction and a retry prompt for `{incorrect}`.",
                f"Better form: `{corrected}`\nWhy: {reason}\nRetry: Write one new "
                "sentence using the corrected pattern.",
            ),
        ]
        records.extend(
            _draft_record(
                "sentence-correction",
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in variants
        )
    return records


def _build_translations(
    entries: list[dict[str, Any]], *, task_type: str
) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        english = item["english"]
        kinyarwanda = item["kinyarwanda"]
        if task_type == "translation-en-rw":
            prompts = [
                (
                    "direct",
                    f"Translate into Kinyarwanda: {english}",
                    f"`{kinyarwanda}`",
                ),
                (
                    "learner",
                    f"Give a beginner-friendly Kinyarwanda translation of: {english}",
                    f"Kinyarwanda: `{kinyarwanda}`\nEnglish check: {english}",
                ),
            ]
        else:
            prompts = [
                ("direct", f"Translate into English: `{kinyarwanda}`", english),
                (
                    "learner",
                    f"Explain the English meaning of `{kinyarwanda}`",
                    f"English: {english}\nKinyarwanda source: `{kinyarwanda}`",
                ),
            ]
        records.extend(
            _draft_record(
                task_type,
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in prompts
        )
    return records


def _build_dialogues(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        scenario = item["scenario"]
        dialogue = item["dialogue"]
        note = item["english_note"]
        variants = [
            (
                "short",
                f"Write a short Kinyarwanda dialogue about {scenario}.",
                dialogue,
            ),
            (
                "noted",
                f"Write a beginner dialogue about {scenario} and add an English note.",
                f"{dialogue}\nEnglish note: {note}",
            ),
            (
                "practice",
                f"Turn {scenario} into a Kinyarwanda role-play.",
                f"{dialogue}\nPractice: Read both roles, then change one name "
                "or place.",
            ),
            (
                "polite",
                f"Give a polite beginner exchange for {scenario}.",
                f"Polite model:\n{dialogue}",
            ),
            (
                "comprehension",
                f"Create a dialogue and one comprehension check about {scenario}.",
                f"{dialogue}\nCheck: What is the main thing the speakers are "
                f"doing?\nAnswer: {note}",
            ),
        ]
        records.extend(
            _draft_record(
                "dialogue",
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in variants
        )
    return records


def _build_quizzes(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        topic = item["topic"]
        questions = item["questions"]
        if not isinstance(questions, list) or len(questions) < 5:
            raise ValueError(f"quiz topic {key} requires at least five questions")
        for count in range(1, 6):
            selected = questions[:count]
            body = "\n".join(
                f"{index}. {question['question']}\nAnswer: {question['answer']}"
                for index, question in enumerate(selected, start=1)
            )
            records.append(
                _draft_record(
                    "quiz-generation",
                    f"Build a {count}-question review quiz about {topic}.",
                    body,
                    content_key=key,
                    variant=f"{count}-question",
                    extra={"requested_question_count": count},
                )
            )
    return records


def _build_culture_register(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_guidance_rows(entries, task_type="culture-register")


def _build_uncertainty(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_guidance_rows(entries, task_type="uncertainty")


def _build_guidance_rows(
    entries: list[dict[str, Any]], *, task_type: str
) -> list[dict[str, Any]]:
    records = []
    for item in entries:
        key = item["key"]
        question = item["question"]
        guidance = item["guidance"]
        caution = item["caution"]
        variants = [
            ("answer", question, f"{guidance} {caution}"),
            (
                "policy",
                f"Write a short tutor rule for this situation: {question}",
                f"Tutor rule: {guidance}\nGuardrail: {caution}",
            ),
            (
                "example",
                f"Give a safe example response to: {question}",
                f"A safe response is: {guidance}\nIt should also remember: {caution}",
            ),
            (
                "review",
                f"What should a reviewer check when a tutor handles this: {question}",
                f"Check that the response does this: {guidance}\nReject it if it "
                f"ignores this: {caution}",
            ),
            (
                "brief",
                f"Answer briefly and carefully: {question}",
                f"{guidance} {caution}",
            ),
        ]
        records.extend(
            _draft_record(
                task_type,
                user,
                assistant,
                content_key=key,
                variant=variant,
            )
            for variant, user, assistant in variants
        )
    return records

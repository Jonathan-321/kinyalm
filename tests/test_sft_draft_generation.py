import random
from collections import Counter
from pathlib import Path

from kinyalm.data.draft_generation import (
    build_profile_records,
    conversation_key,
    load_generation_profile,
)
from kinyalm.data.sft import validate_sft_records
from scripts.generate_sft_draft_batch import build_quizzes

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data" / "sft" / "draft-profiles" / "useful-gap-v1.yaml"


def build_expansion_records():
    profile = load_generation_profile(PROFILE_PATH)
    records = build_profile_records(
        profile,
        rng=random.Random(336),
        batch_code="b002",
    )
    return profile, records


def test_useful_gap_profile_builds_exact_target_mix():
    profile, records = build_expansion_records()

    assert len(records) == 714
    assert Counter(record["task_type"] for record in records) == Counter(
        profile["target_counts"]
    )
    assert all(result.ok for result in validate_sft_records(records))


def test_expansion_ids_and_conversations_are_unique():
    _, records = build_expansion_records()

    assert len({record["id"] for record in records}) == len(records)
    assert len({conversation_key(record) for record in records}) == len(records)
    assert all(record["id"].startswith("sftdraft-b002-") for record in records)


def test_quiz_answers_match_requested_question_count():
    _, records = build_expansion_records()
    quizzes = [record for record in records if record["task_type"] == "quiz-generation"]

    assert len(quizzes) == 60
    for record in quizzes:
        requested = record["requested_question_count"]
        answer = record["messages"][1]["content"]
        assert answer.count("Answer:") == requested


def test_rw_to_english_prompts_do_not_add_duplicate_sentence_punctuation():
    _, records = build_expansion_records()
    prompts = [
        record["messages"][0]["content"]
        for record in records
        if record["task_type"] == "translation-rw-en"
    ]

    assert prompts
    assert all(not prompt.endswith("`.") for prompt in prompts)


def test_foundation_quizzes_are_unique_and_match_prompt_count():
    quizzes = build_quizzes()

    assert len(quizzes) == 20
    assert len({conversation_key(record) for record in quizzes}) == 20
    for record in quizzes:
        prompt = record["messages"][0]["content"]
        requested = int(prompt.split("-question", maxsplit=1)[0].rsplit(" ", 1)[1])
        assert record["messages"][1]["content"].count("Answer:") == requested

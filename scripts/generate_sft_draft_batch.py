"""Generate review-ready draft SFT examples for KinyaLM.

The output is intentionally marked as draft data:

- split=draft
- review_status=needs-review
- source=synthetic-draft

These rows are candidate tutor examples for fluent-speaker review. They are not
approved training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.draft_generation import (  # noqa: E402
    build_profile_records,
    find_conversation_collisions,
    load_generation_profile,
)
from kinyalm.data.sft import load_jsonl, validate_sft_records  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("~/KinyaLMData/drafts").expanduser()
DEFAULT_REVIEW_DIR = Path("~/KinyaLMData/reviewed").expanduser()
SOURCE_NOTE = "synthetic draft; fluent-speaker review required before training"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="sft-drafts-2026-07-13-batch-001")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="Optional YAML profile for a data-driven expansion batch.",
    )
    parser.add_argument(
        "--compare-jsonl",
        action="append",
        default=[],
        type=Path,
        help="Older JSONL batch to check for exact conversation collisions.",
    )
    parser.add_argument(
        "--review-shards",
        type=int,
        default=1,
        help="Write this many balanced review TSV shards in addition to the master.",
    )
    args = parser.parse_args()
    if args.review_shards < 1:
        parser.error("--review-shards must be at least 1")

    rng = random.Random(args.seed)
    batch_code = batch_code_from_id(args.batch_id)
    if args.profile_file:
        profile = load_generation_profile(args.profile_file)
        records = build_profile_records(profile, rng=rng, batch_code=batch_code)
        profile_id = profile["profile_id"]
    else:
        records = build_records(rng, batch_code=batch_code)
        profile_id = "foundation-v1"
    validate_records(records)
    check_existing_collisions(records, args.compare_jsonl)

    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    args.review_dir.expanduser().mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir.expanduser() / f"{args.batch_id}.jsonl"
    review_path = args.review_dir.expanduser() / f"{args.batch_id}.review.tsv"
    summary_path = args.output_dir.expanduser() / f"{args.batch_id}.summary.md"

    write_jsonl(records, jsonl_path)
    write_review_tsv(records, review_path)
    shard_paths = write_review_shards(
        records,
        review_path=review_path,
        shard_count=args.review_shards,
    )
    write_summary(
        records,
        summary_path,
        profile_id=profile_id,
        compared_batches=len(args.compare_jsonl),
        review_shards=args.review_shards,
    )

    print(f"records: {len(records)}")
    print(f"jsonl: {jsonl_path}")
    print(f"review_tsv: {review_path}")
    for shard_path in shard_paths:
        print(f"review_shard: {shard_path}")
    print(f"summary: {summary_path}")
    return 0


def build_records(rng: random.Random, *, batch_code: str = "b001") -> list[dict]:
    builders = [
        build_greetings,
        build_vocabulary,
        build_grammar,
        build_corrections,
        build_translation_en_rw,
        build_translation_rw_en,
        build_dialogues,
        build_quizzes,
        build_culture_register,
        build_uncertainty,
    ]
    records: list[dict] = []
    for builder in builders:
        records.extend(builder())
    rng.shuffle(records)
    for index, record in enumerate(records, start=1):
        category = record["task_type"].replace("-", "")
        record["id"] = f"sftdraft-{batch_code}-{index:04d}-{category}"
    return records


def batch_code_from_id(batch_id: str) -> str:
    match = re.search(r"batch-(\d+)$", batch_id)
    if not match:
        raise SystemExit(
            "--batch-id must end with batch-NNN, for example "
            "sft-drafts-2026-07-13-batch-002"
        )
    return f"b{int(match.group(1)):03d}"


def check_existing_collisions(records: list[dict], paths: list[Path]) -> None:
    for path in paths:
        existing = load_jsonl(path.expanduser())
        collisions = find_conversation_collisions(records, existing)
        if collisions:
            raise SystemExit(
                f"generated {len(collisions)} conversation collisions with {path}"
            )


def base_record(task_type: str, user: str, assistant: str) -> dict:
    return {
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
    }


def build_greetings() -> list[dict]:
    rows = [
        (
            "Teach a beginner when to say `Muraho`.",
            "`Muraho` means hello. It is a polite general greeting. A beginner "
            "can answer with `Muraho` or continue with `Amakuru?` meaning "
            "How are you?",
        ),
        (
            "Explain `Mwaramutse` for a beginner.",
            "`Mwaramutse` is a morning greeting. It is close to good morning. "
            "Use it earlier in the day, and use `Muraho` when you want a more "
            "general hello.",
        ),
        (
            "What does `Amakuru?` mean?",
            "`Amakuru?` literally asks about news, but in conversation it works "
            "like How are you? A simple answer is `Ni meza`, meaning I am well "
            "or things are good.",
        ),
        (
            "Write a simple greeting exchange for two classmates.",
            "A: `Muraho.`\nB: `Muraho. Amakuru?`\nA: `Ni meza. Nawe?`\n"
            "B: `Ni meza, murakoze.`",
        ),
        (
            "Teach `Murakoze` with one example.",
            "`Murakoze` means thank you. Example: `Murakoze cyane` means thank "
            "you very much.",
        ),
        (
            "How should a learner introduce their name?",
            "A simple pattern is `Nitwa ...`, meaning my name is. Example: "
            "`Nitwa Aline` means My name is Aline.",
        ),
    ]
    variants = []
    for user, assistant in rows:
        variants.append(base_record("greeting", user, assistant))
        variants.append(
            base_record(
                "greeting",
                f"Give a shorter beginner version: {user}",
                assistant.split(". ")[0] + ".",
            )
        )
        variants.append(
            base_record(
                "greeting",
                f"Explain this in English with Kinyarwanda examples: {user}",
                assistant,
            )
        )
    return variants[:30]


def build_vocabulary() -> list[dict]:
    vocab = [
        ("ishuri", "school", "Njya ku ishuri.", "I go to school."),
        ("umwarimu", "teacher", "Umwarimu arigisha.", "The teacher teaches."),
        ("umunyeshuri", "student", "Umunyeshuri ariga.", "The student studies."),
        ("igitabo", "book", "Mfite igitabo.", "I have a book."),
        ("ikaramu", "pen", "Mfite ikaramu.", "I have a pen."),
        ("intebe", "chair", "Intebe iri hano.", "The chair is here."),
        ("ameza", "table", "Igitabo kiri ku meza.", "The book is on the table."),
        ("amazi", "water", "Ndashaka amazi.", "I want water."),
        ("inshuti", "friend", "Mfite inshuti.", "I have a friend."),
        ("urugo", "home", "Ngiye mu rugo.", "I am going home."),
        ("umuryango", "family", "Umuryango wanjye.", "My family."),
        ("isomo", "lesson", "Isomo riroroshye.", "The lesson is easy."),
        ("ijambo", "word", "Sobanura iri jambo.", "Explain this word."),
        ("ubufasha", "help", "Nkeneye ubufasha.", "I need help."),
        ("ikibazo", "question", "Mfite ikibazo.", "I have a question."),
    ]
    records = []
    for word, meaning, example, translation in vocab:
        records.append(
            base_record(
                "vocabulary",
                f"Teach the Kinyarwanda word `{word}` to a beginner.",
                f"`{word}` means {meaning}. Example: `{example}` means "
                f"{translation}",
            )
        )
        records.append(
            base_record(
                "vocabulary",
                f"Give one simple sentence with `{word}` and translate it.",
                f"`{example}` means {translation} The word `{word}` means "
                f"{meaning}.",
            )
        )
        records.append(
            base_record(
                "vocabulary",
                f"Make a tiny flashcard for `{word}`.",
                f"Front: `{word}`\nBack: {meaning}\nExample: `{example}`",
            )
        )
    return records


def build_grammar() -> list[dict]:
    topics = [
        (
            "`Nitwa Aline`",
            "`Nitwa Aline` means My name is Aline. `Nitwa` is used when a "
            "speaker gives their name.",
        ),
        (
            "`Ndiga Ikinyarwanda`",
            "`Ndiga Ikinyarwanda` means I am learning Kinyarwanda. `Ndiga` "
            "comes from the idea of studying or learning.",
        ),
        (
            "`umwarimu` and `abarimu`",
            "`umwarimu` means teacher, and `abarimu` means teachers. This is a "
            "singular/plural pattern that learners should review with a fluent "
            "speaker.",
        ),
        (
            "`igitabo` and `ibitabo`",
            "`igitabo` means book, and `ibitabo` means books. The beginning of "
            "the word changes when it becomes plural.",
        ),
        (
            "`gukora`, `Ndakora`, and `Barakora`",
            "`gukora` means to work or to do. `Ndakora` can mean I work or I am "
            "working. `Barakora` can mean they work or they are working.",
        ),
        (
            "`w'Ikinyarwanda`",
            "The apostrophe in `w'Ikinyarwanda` shows a shortened connection "
            "between words. A reviewer should confirm the exact explanation for "
            "the final dataset.",
        ),
        (
            "`Sinzabikora`",
            "`Sinzabikora` is a complex word. It may combine negative, subject, "
            "future, object, and verb-stem information. Keep this marked for "
            "review before teaching it confidently.",
        ),
        (
            "`Ndi umunyeshuri`",
            "`Ndi umunyeshuri` means I am a student. `Ndi` is useful for simple "
            "I am sentences.",
        ),
        (
            "`Witwa nde?`",
            "`Witwa nde?` asks What is your name? A simple answer is "
            "`Nitwa ...`.",
        ),
    ]
    records = []
    for phrase, explanation in topics:
        records.append(
            base_record(
                "grammar-explanation",
                f"Explain {phrase} word by word for a beginner.",
                explanation,
            )
        )
        records.append(
            base_record(
                "grammar-explanation",
                f"Give a beginner-friendly grammar note about {phrase}.",
                explanation + " Keep the explanation short for a new learner.",
            )
        )
        records.append(
            base_record(
                "grammar-explanation",
                f"Why might {phrase} be useful for a Kinyarwanda learner?",
                explanation,
            )
        )
        records.append(
            base_record(
                "grammar-explanation",
                f"Create a tiny lesson around {phrase}.",
                f"Lesson: {explanation}\nPractice: Ask the learner to make one "
                "similar sentence.",
            )
        )
        records.append(
            base_record(
                "grammar-explanation",
                f"Explain {phrase}, but say what should be checked by a reviewer.",
                explanation
                + " A fluent speaker should confirm the final wording before training.",
            )
        )
    return records


def build_corrections() -> list[dict]:
    corrections = [
        (
            "`Ndi kwiga Ikinyarwanda.`",
            "`Ndiga Ikinyarwanda.`",
            "Use `Ndiga` for I am learning/I study.",
        ),
        (
            "`Jyewe yitwa Jonathan.`",
            "`Nitwa Jonathan.`",
            "`Nitwa` is the simple pattern for My name is.",
        ),
        (
            "`Ikinyarwanda ndiga.`",
            "`Ndiga Ikinyarwanda.`",
            "The safer beginner order is I am learning Kinyarwanda.",
        ),
        (
            "`Mfite ikibazo?` when asking Do I have a question?",
            "`Mfite ikibazo?` may be understandable, but the reviewer should "
            "confirm whether the learner needs a clearer question form.",
            "Mark the uncertainty instead of overcorrecting.",
        ),
        (
            "`Murakoze hello.`",
            "`Muraho` means hello. `Murakoze` means thank you.",
            "The learner mixed two different greeting words.",
        ),
        (
            "`Nitwa umunyeshuri.` when the learner means I am a student.",
            "`Ndi umunyeshuri.`",
            "`Nitwa` is for names. `Ndi` is for I am.",
        ),
        (
            "`I am Aline` translated as `Ndi Aline.`",
            "`Nitwa Aline.`",
            "For giving a name, `Nitwa Aline` is safer for beginners.",
        ),
        (
            "`Please slowly repeat` translated word-for-word.",
            "`Subiramo buhoro, ndakwinginze.`",
            "Teach a natural classroom phrase instead of copying English order.",
        ),
        (
            "`Ndashaka explain iri jambo.`",
            "`Sobanura iri jambo, ndakwinginze.`",
            "Avoid mixing English into the Kinyarwanda sentence unless the task "
            "asks for code-switching.",
        ),
    ]
    records = []
    for wrong, corrected, note in corrections:
        records.append(
            base_record(
                "sentence-correction",
                f"Correct this learner sentence: {wrong}",
                f"Suggested correction: {corrected}\nReason: {note}",
            )
        )
        records.append(
            base_record(
                "sentence-correction",
                f"Explain what is wrong with {wrong}",
                f"A better version is: {corrected}\n{note}",
            )
        )
        records.append(
            base_record(
                "sentence-correction",
                f"Correct politely and briefly: {wrong}",
                f"Try: {corrected}\n{note}",
            )
        )
        records.append(
            base_record(
                "sentence-correction",
                f"Give a correction and one teaching note for {wrong}",
                f"Correction: {corrected}\nTeaching note: {note}",
            )
        )
        records.append(
            base_record(
                "sentence-correction",
                f"If a beginner wrote {wrong}, how should the tutor respond?",
                f"The tutor can say: {corrected}\nThen explain: {note}",
            )
        )
    return records


def build_translation_en_rw() -> list[dict]:
    pairs = [
        ("Hello.", "Muraho."),
        ("Good morning.", "Mwaramutse."),
        ("Thank you.", "Murakoze."),
        ("Thank you very much.", "Murakoze cyane."),
        ("My name is Aline.", "Nitwa Aline."),
        ("I am a student.", "Ndi umunyeshuri."),
        ("I am a teacher.", "Ndi umwarimu."),
        ("I am learning Kinyarwanda.", "Ndiga Ikinyarwanda."),
        ("Please repeat slowly.", "Subiramo buhoro, ndakwinginze."),
        ("I need help.", "Nkeneye ubufasha."),
        ("I have a question.", "Mfite ikibazo."),
        ("Explain this word.", "Sobanura iri jambo."),
        ("The teacher helps the student.", "Umwarimu afasha umunyeshuri."),
        ("The book is on the table.", "Igitabo kiri ku meza."),
        ("What is your name?", "Witwa nde?"),
        ("I am going home.", "Ngiye mu rugo."),
        ("I want water.", "Ndashaka amazi."),
    ]
    records = []
    for english, kinyarwanda in pairs:
        records.append(
            base_record(
                "translation-en-rw",
                f"Translate into Kinyarwanda: {english}",
                f"`{kinyarwanda}`",
            )
        )
        records.append(
            base_record(
                "translation-en-rw",
                f"Translate and briefly explain: {english}",
                f"`{kinyarwanda}` This is a short beginner-friendly translation.",
            )
        )
    return records


def build_translation_rw_en() -> list[dict]:
    pairs = [
        ("Muraho.", "Hello."),
        ("Mwaramutse.", "Good morning."),
        ("Murakoze.", "Thank you."),
        ("Murakoze cyane.", "Thank you very much."),
        ("Nitwa Aline.", "My name is Aline."),
        ("Ndi umunyeshuri.", "I am a student."),
        ("Ndi umwarimu.", "I am a teacher."),
        ("Ndiga Ikinyarwanda.", "I am learning Kinyarwanda."),
        ("Subiramo buhoro.", "Repeat slowly."),
        ("Nkeneye ubufasha.", "I need help."),
        ("Mfite ikibazo.", "I have a question."),
        ("Sobanura iri jambo.", "Explain this word."),
        ("Umwarimu afasha umunyeshuri.", "The teacher helps the student."),
        ("Igitabo kiri ku meza.", "The book is on the table."),
        ("Witwa nde?", "What is your name?"),
        ("Ngiye mu rugo.", "I am going home."),
        ("Ndashaka amazi.", "I want water."),
    ]
    records = []
    for kinyarwanda, english in pairs:
        records.append(
            base_record(
                "translation-rw-en",
                f"Translate into English: `{kinyarwanda}`",
                english,
            )
        )
        records.append(
            base_record(
                "translation-rw-en",
                f"Translate and explain for a beginner: `{kinyarwanda}`",
                f"{english} This is a short sentence a beginner can memorize.",
            )
        )
    return records


def build_dialogues() -> list[dict]:
    scenarios = [
        (
            "greeting a teacher",
            "Student: `Mwaramutse, mwarimu.`\nTeacher: `Mwaramutse. Amakuru?`\n"
            "Student: `Ni meza, murakoze.`",
        ),
        (
            "asking for repetition",
            "Learner: `Subiramo buhoro, ndakwinginze.`\n"
            "Tutor: `Yego. Ndabisubiramo buhoro.`",
        ),
        (
            "introducing a name",
            "A: `Muraho. Nitwa Aline.`\nB: `Muraho Aline. Nitwa Eric.`",
        ),
        (
            "asking what a word means",
            "Learner: `Iri jambo risobanura iki?`\nTutor: `Risobanura ...`",
        ),
        (
            "asking for help",
            "Learner: `Nkeneye ubufasha.`\nTutor: `Ni iki ushaka gusobanukirwa?`",
        ),
        (
            "classroom thanks",
            "Student: `Murakoze, mwarimu.`\nTeacher: `Murakoze nawe.`",
        ),
        (
            "checking understanding",
            "Tutor: `Urumva?`\nLearner: `Yego, ndumva.`",
        ),
    ]
    records = []
    for scenario, dialogue in scenarios:
        styles = [
            "short",
            "beginner-friendly",
            "with an English note",
            "with a practice question",
            "polite",
        ]
        for style in styles:
            assistant = dialogue
            if style == "with an English note":
                assistant += "\nEnglish note: This is a simple classroom dialogue."
            if style == "with a practice question":
                assistant += "\nPractice: Replace the name with your own name."
            records.append(
                base_record(
                    "dialogue",
                    f"Write a {style} Kinyarwanda dialogue about {scenario}.",
                    assistant,
                )
            )
    return records


def build_quizzes() -> list[dict]:
    quiz_specs = [
        (
            "greetings",
            [
                "What does `Muraho` mean?",
                "When can you use `Mwaramutse`?",
                "Translate `Murakoze`.",
                "What does `Amakuru?` ask?",
                "Give one simple answer to `Amakuru?`",
            ],
            ["Hello.", "In the morning.", "Thank you.", "How are you?", "`Ni meza.`"],
        ),
        (
            "classroom words",
            [
                "What does `ishuri` mean?",
                "What does `umwarimu` mean?",
                "What does `igitabo` mean?",
                "Translate `Mfite ikibazo`.",
                "Translate `Nkeneye ubufasha`.",
            ],
            ["School.", "Teacher.", "Book.", "I have a question.", "I need help."],
        ),
        (
            "introductions",
            [
                "Translate `Nitwa Aline`.",
                "How do you ask What is your name?",
                "Translate My name is Eric.",
                "How do you say I am a student?",
                "Which form is used to give a name: `Ndi` or `Nitwa`?",
            ],
            [
                "My name is Aline.",
                "`Witwa nde?`",
                "`Nitwa Eric.`",
                "`Ndi umunyeshuri.`",
                "`Nitwa`.",
            ],
        ),
        (
            "corrections",
            [
                "Fix `Ndi kwiga Ikinyarwanda.`",
                "Fix `Jyewe yitwa Aline.`",
                "Choose hello: `Muraho` or `Murakoze`?",
                "Fix `Nitwa umunyeshuri.`",
                "Fix `Ngiye isoko.`",
            ],
            [
                "`Ndiga Ikinyarwanda.`",
                "`Nitwa Aline.`",
                "`Muraho`.",
                "`Ndi umunyeshuri.`",
                "`Ngiye ku isoko.`",
            ],
        ),
    ]
    records = []
    for topic, questions, answers in quiz_specs:
        for count in range(1, 6):
            body = "\n".join(
                f"{i}. {question}\nAnswer: {answer}"
                for i, (question, answer) in enumerate(
                    zip(questions[:count], answers[:count], strict=True), start=1
                )
            )
            records.append(
                base_record(
                    "quiz-generation",
                    f"Create a {count}-question beginner quiz about {topic}.",
                    body,
                )
            )
    return records


def build_culture_register() -> list[dict]:
    prompts = [
        (
            "Explain why a tutor should be careful with formal and informal greetings.",
            "A tutor should avoid claiming one phrase is always right. It can say "
            "`Muraho` is a general polite greeting and ask a fluent speaker to "
            "confirm local context when register matters.",
        ),
        (
            "Teach a safe note about cultural explanations.",
            "The tutor should explain language use without making broad claims "
            "about all Rwandans. If unsure, it should say the phrase needs "
            "fluent-speaker confirmation.",
        ),
        (
            "How should the tutor handle a proverb it is not sure about?",
            "It should not invent a translation. It can say it is not sure, give "
            "a tentative meaning only if marked clearly, and ask for reviewer "
            "confirmation.",
        ),
        (
            "Explain a respectful classroom tone.",
            "A respectful tutor corrects mistakes gently, gives the better form, "
            "and avoids shaming the learner.",
        ),
        (
            "Give a register warning for beginner phrases.",
            "Some phrases may sound more formal or casual depending on context. "
            "The tutor should mark uncertainty and avoid overconfident rules.",
        ),
    ]
    return [
        base_record("culture-register", user, assistant)
        for user, assistant in prompts
    ]


def build_uncertainty() -> list[dict]:
    prompts = [
        (
            "What should the tutor do if it is unsure about a Kinyarwanda "
            "grammar rule?",
            "It should say it is not fully sure, avoid presenting a guess as fact, "
            "and suggest checking with a fluent speaker.",
        ),
        (
            "The learner asks for a proverb translation the tutor does not know. "
            "Respond safely.",
            "I am not fully sure about that proverb, so I do not want to invent a "
            "translation. If you share the context, I can help analyze the words, "
            "but a fluent speaker should confirm the final meaning.",
        ),
        (
            "How should the tutor flag uncertain spelling?",
            "It should say the spelling needs review and avoid using the uncertain "
            "form as a training example until checked.",
        ),
        (
            "The model is asked for cultural advice it cannot verify. What "
            "should it say?",
            "It should avoid broad cultural claims. It can explain that usage may "
            "depend on region, age, setting, and relationship.",
        ),
        (
            "Write a humble correction when the tutor might be wrong.",
            "A safe response is: `I think a better form may be ...`, followed by "
            "`please confirm with a fluent speaker before relying on it.`",
        ),
    ]
    return [base_record("uncertainty", user, assistant) for user, assistant in prompts]


def validate_records(records: list[dict]) -> None:
    failures = [result for result in validate_sft_records(records) if not result.ok]
    if failures:
        for result in failures[:20]:
            for error in result.errors:
                print(f"record {result.line_number}: {error}", file=sys.stderr)
        raise SystemExit(f"generated invalid records: {len(failures)} failures")


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_tsv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "id",
        "task_type",
        "generation_profile",
        "content_key",
        "variant",
        "requested_question_count",
        "user_prompt",
        "assistant_response",
        "review_status",
        "correctness_1_5",
        "naturalness_1_5",
        "helpfulness_1_5",
        "failure_tags",
        "reviewer",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "task_type": record["task_type"],
                    "generation_profile": record.get("generation_profile", ""),
                    "content_key": record.get("content_key", ""),
                    "variant": record.get("variant", ""),
                    "requested_question_count": record.get(
                        "requested_question_count", ""
                    ),
                    "user_prompt": one_line(record["messages"][0]["content"]),
                    "assistant_response": one_line(record["messages"][1]["content"]),
                    "review_status": record["review_status"],
                    "correctness_1_5": "",
                    "naturalness_1_5": "",
                    "helpfulness_1_5": "",
                    "failure_tags": "",
                    "reviewer": "",
                    "reviewer_notes": "",
                }
            )


def write_review_shards(
    records: list[dict], *, review_path: Path, shard_count: int
) -> list[Path]:
    for stale_path in review_path.parent.glob(
        f"{review_path.stem}.part-*-of-*.tsv"
    ):
        stale_path.unlink()
    if shard_count == 1:
        return []
    shard_paths = []
    for shard_index in range(shard_count):
        shard_path = review_path.with_name(
            f"{review_path.stem}.part-{shard_index + 1:02d}-of-{shard_count:02d}.tsv"
        )
        write_review_tsv(records[shard_index::shard_count], shard_path)
        shard_paths.append(shard_path)
    return shard_paths


def one_line(text: str) -> str:
    return text.replace("\r\n", "\\n").replace("\n", "\\n")


def write_summary(
    records: list[dict],
    path: Path,
    *,
    profile_id: str = "foundation-v1",
    compared_batches: int = 0,
    review_shards: int = 1,
) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["task_type"]] = counts.get(record["task_type"], 0) + 1
    lines = [
        "# SFT Draft Batch Summary",
        "",
        "Status: generated draft only. Not approved for training.",
        "",
        f"Rows: {len(records)}",
        f"Generation profile: `{profile_id}`",
        f"Older batches checked for exact conversation collisions: {compared_batches}",
        f"Review TSV shards: {review_shards}",
        "",
        "| Task Type | Rows |",
        "| --- | ---: |",
    ]
    for task_type, count in sorted(counts.items()):
        lines.append(f"| {task_type} | {count} |")
    lines.extend(
        [
            "",
            "Review rule: promote rows only after fluent-speaker review. Rows that",
            "remain `split=draft` or `review_status=needs-review` must not train.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert a reviewed distillation-queue export into project SFT JSONL.

The distillation review sheet exports one row per multi-turn conversation with
columns like ``original_conversation``, ``my_flag``, ``suggested_revision`` and
``critic_feedback``. That shape is a *review artifact*, not the trainable SFT
schema in ``docs/data/sft-data-schema.md`` (which needs ``id``, ``task_type``,
``split``, ``messages`` of exactly two turns, etc.).

This script turns an approved review export into schema-valid SFT rows:

* rows flagged ``Keep`` train on ``original_conversation`` (reviewer judged it
  correct and declined the critic's revision);
* rows flagged to take the revision train on ``suggested_revision``;
* each multi-turn conversation is split into adjacent (user, assistant) pairs,
  because the first-run schema requires exactly two messages per row;
* the result is split deterministically into train/validation.

Usage:

    python scripts/convert_distillation_review_to_sft.py \
        --review-jsonl data/reviewed/tessy_distillation_queue.jsonl \
        --out-prefix data/sft/tessy-distill-review \
        --reviewer "Tessy Mugisha" \
        --train-ratio 0.9

Then validate and train:

    python scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.train.jsonl
    python scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.validation.jsonl
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.sft import validate_sft_records  # noqa: E402

# Map the distillation task_family labels onto the project's allowed task_type
# vocabulary. Several are approximate; the team can refine labels later without
# regenerating the text. Kept explicit so the mapping is auditable.
TASK_FAMILY_TO_TYPE = {
    "conversation-practice": "dialogue",
    "learner-correction-feedback": "sentence-correction",
    "translation-with-explanation": "translation-en-rw",
    "grammar-and-structure": "grammar-explanation",
    "pronunciation-and-orthography": "grammar-explanation",
    "multi-turn-consistency": "dialogue",
    "reading-comprehension": "dialogue",
    "vocabulary-definition-usage": "vocabulary",
    "register-culture-code-switching": "culture-register",
    "sentence-generation": "dialogue",
    "ambiguity-and-hallucination-resistance": "uncertainty",
}

TURN_RE = re.compile(r"^(USER|ASSISTANT):\s*$", re.MULTILINE)


def parse_turns(text: str) -> list[tuple[str, str]]:
    """Parse a 'USER:\\n...\\nASSISTANT:\\n...' block into (role, content)."""
    parts = TURN_RE.split(text)
    # re.split with one capture group yields: [pre, ROLE, body, ROLE, body, ...]
    turns: list[tuple[str, str]] = []
    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip().lower()
        body = parts[i + 1].strip()
        if role in ("user", "assistant") and body:
            turns.append((role, body))
        i += 2
    return turns


def pairs_from_turns(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Emit adjacent (user_content, assistant_content) pairs."""
    out: list[tuple[str, str]] = []
    for idx in range(len(turns) - 1):
        role, body = turns[idx]
        nrole, nbody = turns[idx + 1]
        if role == "user" and nrole == "assistant":
            out.append((body, nbody))
    return out


def split_for(example_id: str, train_ratio: float) -> str:
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "validation"


def build_rows(review_row: dict, reviewer: str, train_ratio: float) -> list[dict]:
    flag = str(review_row.get("my_flag", "")).strip().lower()
    if flag in ("", "keep"):
        source_text = review_row.get("original_conversation", "")
        basis = "original (reviewer flagged Keep; critic revision declined)"
    else:
        source_text = review_row.get("suggested_revision") or review_row.get(
            "original_conversation", ""
        )
        basis = f"revision applied (flag={flag})"

    conv_id = str(review_row.get("conversation_id", "")).strip()
    family = str(review_row.get("task_family", "")).strip()
    task_type = TASK_FAMILY_TO_TYPE.get(family, "dialogue")

    turns = parse_turns(source_text)
    pairs = pairs_from_turns(turns)

    rows: list[dict] = []
    for p_idx, (user, assistant) in enumerate(pairs, start=1):
        example_id = f"{conv_id}-p{p_idx:02d}"
        rows.append(
            {
                "id": example_id,
                "task_type": task_type,
                "split": split_for(example_id, train_ratio),
                "source": "kinyalm-distillation-queue",
                "source_status": "team-authored",
                "review_status": "approved",
                "language_mix": "kinyarwanda+english",
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "reviewer_notes": (
                    f"Reviewer: {reviewer}. Basis: {basis}. "
                    f"task_family={family}. conversation_id={conv_id}."
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--reviewer", default="Tessy Mugisha")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    review_path = Path(args.review_jsonl).expanduser()
    review_rows = [
        json.loads(line)
        for line in review_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    all_rows: list[dict] = []
    skipped = 0
    for r in review_rows:
        rows = build_rows(r, args.reviewer, args.train_ratio)
        if not rows:
            skipped += 1
        all_rows.extend(rows)

    results = validate_sft_records(all_rows)
    bad = [res for res in results if not res.ok]
    if bad:
        for res in bad[:20]:
            print(f"row {res.line_number}: {', '.join(res.errors)}", file=sys.stderr)
        raise SystemExit(f"conversion produced {len(bad)} invalid rows")

    train = [r for r in all_rows if r["split"] == "train"]
    validation = [r for r in all_rows if r["split"] == "validation"]

    out_prefix = Path(args.out_prefix).expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path = out_prefix.with_suffix(".train.jsonl")
    val_path = out_prefix.with_suffix(".validation.jsonl")

    for path, rows in ((train_path, train), (val_path, validation)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"review conversations read: {len(review_rows)} (skipped {skipped})")
    print(f"SFT rows written: {len(all_rows)} "
          f"({len(train)} train / {len(validation)} validation)")
    print(f"train: {train_path}")
    print(f"validation: {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Convert a reviewed distillation-queue export into project SFT JSONL.

The distillation review sheet exports one row per multi-turn conversation with
columns like ``original_conversation``, ``my_flag``, ``suggested_revision`` and
``critic_feedback``. That shape is a *review artifact*, not the trainable SFT
schema in ``docs/data/sft-data-schema.md``.

This script turns an approved review export into schema-valid SFT rows:

* critic-accepted rows flagged ``Keep`` use ``original_conversation``;
* rows explicitly flagged to use a revision use ``suggested_revision``;
* critic-disputed rows stay out of training until explicitly adjudicated;
* each multi-turn conversation remains intact in one record;
* train/validation assignment is deterministic at conversation level.

Usage:

    python scripts/convert_distillation_review_to_sft.py \
        --review-jsonl data/reviewed/tessy_distillation_queue.jsonl \
        --out-prefix data/sft/tessy-distill-review \
        --reviewer "Tessy Mugisha" \
        --train-ratio 0.9 \
        --mlx-data-dir data/mlx-data

Then validate and train:

    python scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.train.jsonl
    python scripts/validate_sft_jsonl.py data/sft/tessy-distill-review.validation.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

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
    "pronunciation-and-orthography": "pronunciation",
    "multi-turn-consistency": "dialogue",
    "reading-comprehension": "reading-comprehension",
    "vocabulary-definition-usage": "vocabulary",
    "register-culture-code-switching": "culture-register",
    "sentence-generation": "sentence-generation",
    "ambiguity-and-hallucination-resistance": "uncertainty",
}

TURN_RE = re.compile(r"^(USER|ASSISTANT):\s*$", re.MULTILINE)
KNOWN_PRIORITIES = {"critic accepted", "repair first"}
REVISION_FLAGS = {
    "accept revision",
    "apply revision",
    "revision",
    "use revision",
    "use suggested revision",
}
WITHHOLD_FLAGS = {"needs fix", "not sure", "reject", "rejected", "skip"}


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


def split_for(conversation_id: str, train_ratio: float) -> str:
    """Assign an entire conversation to one deterministic data split."""

    digest = hashlib.sha1(conversation_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "validation"


def select_reviewed_text(
    review_row: dict,
    *,
    accept_disputed_keeps: bool,
) -> tuple[str, str] | None:
    """Select approved text or withhold a row whose decision is unresolved."""

    flag = str(review_row.get("my_flag", "")).strip().lower()
    priority = str(review_row.get("priority", "")).strip().lower()
    if priority not in KNOWN_PRIORITIES:
        raise ValueError(f"unknown critic priority: {priority!r}")

    if flag == "keep":
        if priority == "repair first" and not accept_disputed_keeps:
            return None
        source_text = str(review_row.get("original_conversation", "")).strip()
        basis = f"original (reviewer flagged Keep; critic priority={priority})"
        return source_text, basis

    if flag in REVISION_FLAGS:
        source_text = str(review_row.get("suggested_revision", "")).strip()
        if not source_text:
            raise ValueError("reviewer selected a revision but none is present")
        return source_text, f"suggested revision (reviewer flag={flag})"

    if flag in WITHHOLD_FLAGS or not flag:
        return None
    raise ValueError(f"unknown reviewer flag: {flag!r}")


def build_row(
    review_row: dict,
    reviewer: str,
    train_ratio: float,
    *,
    accept_disputed_keeps: bool = False,
) -> dict | None:
    """Build one full-conversation SFT record when the review gate passes."""

    selection = select_reviewed_text(
        review_row,
        accept_disputed_keeps=accept_disputed_keeps,
    )
    if selection is None:
        return None
    source_text, basis = selection

    conv_id = str(review_row.get("conversation_id", "")).strip()
    family = str(review_row.get("task_family", "")).strip()
    try:
        task_type = TASK_FAMILY_TO_TYPE[family]
    except KeyError as exc:
        raise ValueError(f"unknown task family: {family!r}") from exc

    turns = parse_turns(source_text)
    if not turns:
        raise ValueError("conversation contains no parseable turns")

    return {
        "id": conv_id,
        "task_type": task_type,
        "split": split_for(conv_id, train_ratio),
        "source": "kinyalm-distillation-queue",
        "source_status": "team-authored",
        "review_status": "approved",
        "language_mix": "kinyarwanda+english",
        "messages": [{"role": role, "content": content} for role, content in turns],
        "reviewer_notes": (
            f"Reviewer: {reviewer}. Basis: {basis}. "
            f"task_family={family}. conversation_id={conv_id}."
        ),
    }


def convert_review_rows(
    review_rows: list[dict],
    reviewer: str,
    train_ratio: float,
    *,
    accept_disputed_keeps: bool = False,
) -> tuple[list[dict], list[str]]:
    """Convert approved rows and return IDs withheld from training."""

    approved: list[dict] = []
    withheld: list[str] = []
    for review_row in review_rows:
        conversation_id = str(review_row.get("conversation_id", "")).strip()
        try:
            row = build_row(
                review_row,
                reviewer,
                train_ratio,
                accept_disputed_keeps=accept_disputed_keeps,
            )
        except ValueError as exc:
            raise ValueError(f"{conversation_id or '<missing id>'}: {exc}") from exc
        if row is None:
            withheld.append(conversation_id or "<missing id>")
        else:
            approved.append(row)
    return approved, withheld


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--reviewer", default="Tessy Mugisha")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument(
        "--accept-disputed-keeps",
        action="store_true",
        help=(
            "Promote rows marked Keep even when the critic said Repair first. "
            "Use only after an explicit adjudication decision."
        ),
    )
    parser.add_argument(
        "--mlx-data-dir",
        default=None,
        help="Also write MLX-LM-compatible train.jsonl and valid.jsonl files.",
    )
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be greater than 0 and less than 1")

    review_path = Path(args.review_jsonl).expanduser()
    review_rows = [
        json.loads(line)
        for line in review_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    try:
        all_rows, withheld = convert_review_rows(
            review_rows,
            args.reviewer,
            args.train_ratio,
            accept_disputed_keeps=args.accept_disputed_keeps,
        )
    except ValueError as exc:
        raise SystemExit(f"conversion failed: {exc}") from exc

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

    write_jsonl(train_path, train)
    write_jsonl(val_path, validation)

    print(
        f"review conversations read: {len(review_rows)} "
        f"({len(withheld)} withheld for adjudication)"
    )
    print(
        f"SFT conversations written: {len(all_rows)} "
        f"({len(train)} train / {len(validation)} validation)"
    )
    print(f"train: {train_path}")
    print(f"validation: {val_path}")

    if args.mlx_data_dir:
        mlx_data_dir = Path(args.mlx_data_dir).expanduser()
        mlx_train = mlx_data_dir / "train.jsonl"
        mlx_valid = mlx_data_dir / "valid.jsonl"
        write_jsonl(mlx_train, train)
        write_jsonl(mlx_valid, validation)
        print(f"MLX train: {mlx_train}")
        print(f"MLX validation: {mlx_valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

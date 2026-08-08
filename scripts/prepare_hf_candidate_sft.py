#!/usr/bin/env python3
"""Build deterministic experimental splits from a pinned HF candidate set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.sft import validate_sft_records  # noqa: E402

DEFAULT_REPO_ID = "kinyalm/kinyalm-data-lake"
DEFAULT_DATA_PATH = (
    "data/candidates/kinyalm-sft-10k-v4/"
    "kinyalm-sft-10k-v4-candidates.jsonl"
)
DEFAULT_MANIFEST_PATH = "data/candidates/kinyalm-sft-10k-v4/manifest.json"
TRAINING_TIER = "experimental-candidate-unreviewed"
QUALITY_POLICIES = ("unflagged", "strict-script-clean", "core-direct")
CORE_DIRECT_FAMILIES = {
    "translation-with-explanation",
    "grammar-and-structure",
    "vocabulary-definition-usage",
    "pronunciation-and-orthography",
    "learner-correction-feedback",
    "conversation-practice",
    "multi-turn-consistency",
    "register-culture-code-switching",
}
ALLOWED_NON_ASCII = frozenset("’‘“”–—…×÷≤≥°")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def stable_key(seed: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def unsupported_script_characters(row: dict[str, Any]) -> set[str]:
    """Return non-Latin or invisible characters from generated row text."""

    texts = [str(row.get("lesson_focus", ""))]
    texts.extend(
        str(message.get("content", ""))
        for message in row.get("messages", [])
        if isinstance(message, dict)
    )
    unsupported: set[str] = set()
    for text in texts:
        for character in text:
            if ord(character) < 128 or character in ALLOWED_NON_ASCII:
                continue
            if unicodedata.name(character, "").startswith("LATIN "):
                continue
            unsupported.add(character)
    return unsupported


def candidate_rejection_reasons(
    row: dict[str, Any], *, include_flagged: bool, quality_policy: str
) -> list[str]:
    reasons = []
    if row.get("candidate_flags") and not include_flagged:
        reasons.append("candidate-flagged")
    if quality_policy in {"strict-script-clean", "core-direct"}:
        if unsupported_script_characters(row):
            reasons.append("unsupported-script-character")
    if quality_policy == "core-direct":
        family = str(row.get("task_family", ""))
        if family not in CORE_DIRECT_FAMILIES:
            reasons.append(f"excluded-task-family:{family or 'missing'}")
    return reasons


def build_candidate_splits(
    rows: list[dict[str, Any]],
    *,
    split_seed: str,
    include_flagged: bool,
    quality_policy: str = "unflagged",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if quality_policy not in QUALITY_POLICIES:
        raise ValueError(f"unknown quality policy: {quality_policy}")

    rejection_counts: Counter[str] = Counter()
    rejected_rows = 0
    selected = []
    for row in rows:
        reasons = candidate_rejection_reasons(
            row,
            include_flagged=include_flagged,
            quality_policy=quality_policy,
        )
        if reasons:
            rejected_rows += 1
            rejection_counts.update(reasons)
            continue
        selected.append(json.loads(json.dumps(row, ensure_ascii=False)))
    if not selected:
        raise ValueError("candidate selection is empty")

    ids = [str(row.get("id", "")) for row in selected]
    if any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate rows contain empty or duplicate ids")
    conversation_keys = [
        json.dumps(
            row.get("messages"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in selected
    ]
    if len(set(conversation_keys)) != len(conversation_keys):
        raise ValueError("candidate rows contain exact duplicate conversations")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        family = str(row.get("task_family", ""))
        if not family:
            raise ValueError(f"{row['id']}: task_family is missing")
        if row.get("review_status") != "candidate-unreviewed":
            raise ValueError(f"{row['id']}: row is not a candidate-unreviewed row")
        grouped[family].append(row)

    split_names = (
        "experimental-train",
        "experimental-validation",
        "experimental-test",
    )
    for family_rows in grouped.values():
        family_rows.sort(key=lambda row: stable_key(split_seed, row["id"]))
        train_count = int(len(family_rows) * 0.90)
        validation_count = int(len(family_rows) * 0.05)
        boundaries = (train_count, train_count + validation_count)
        for index, row in enumerate(family_rows):
            split = (
                split_names[0]
                if index < boundaries[0]
                else split_names[1]
                if index < boundaries[1]
                else split_names[2]
            )
            row["split"] = split
            row["source_status"] = "model-generated"
            row["training_tier"] = TRAINING_TIER

    selected.sort(key=lambda row: row["id"])
    failures = [result for result in validate_sft_records(selected) if not result.ok]
    if failures:
        preview = "; ".join(
            f"row {result.line_number}: {', '.join(result.errors)}"
            for result in failures[:10]
        )
        raise ValueError(f"candidate split validation failed: {preview}")

    split_counts = Counter(row["split"] for row in selected)
    family_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        family_split_counts[row["task_family"]][row["split"]] += 1
    return selected, {
        "selected_rows": len(selected),
        "rejected_rows": rejected_rows,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "include_flagged": include_flagged,
        "quality_policy": quality_policy,
        "flagged_rows": sum(bool(row.get("candidate_flags")) for row in selected),
        "split_seed": split_seed,
        "split_counts": dict(sorted(split_counts.items())),
        "task_family_split_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_split_counts.items())
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_metadata(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": len(rows),
        "assistant_responses": sum(
            message["role"] == "assistant"
            for row in rows
            for message in row["messages"]
        ),
        "multi_turn_conversations": sum(
            sum(message["role"] == "assistant" for message in row["messages"]) > 1
            for row in rows
        ),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def materialize_hf_candidate_sft(
    *,
    repo_id: str,
    revision: str,
    data_path: str,
    source_manifest_path: str,
    output_dir: Path,
    split_seed: str,
    include_flagged: bool,
    quality_policy: str,
) -> dict[str, Any]:
    api = HfApi()
    resolved_revision = api.dataset_info(repo_id, revision=revision).sha
    local_data = Path(
        hf_hub_download(
            repo_id,
            data_path,
            repo_type="dataset",
            revision=resolved_revision,
        )
    )
    local_source_manifest = Path(
        hf_hub_download(
            repo_id,
            source_manifest_path,
            repo_type="dataset",
            revision=resolved_revision,
        )
    )
    source_manifest = json.loads(local_source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("data_file_sha256") != file_sha256(local_data):
        raise ValueError("candidate data does not match its source manifest")

    rows, build = build_candidate_splits(
        load_jsonl(local_data),
        split_seed=split_seed,
        include_flagged=include_flagged,
        quality_policy=quality_policy,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for label, split in (
        ("train", "experimental-train"),
        ("validation", "experimental-validation"),
        ("test", "experimental-test"),
    ):
        split_rows = [row for row in rows if row["split"] == split]
        path = output_dir / f"{label}.jsonl"
        write_jsonl(path, split_rows)
        split_paths[label] = path
        outputs[label] = output_metadata(path, split_rows)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_tier": TRAINING_TIER,
        "human_reviewed": False,
        "production_eligible": False,
        "owner_authorized_for_experimental_training": True,
        "intended_use": (
            "Controlled Gemma 4 QLoRA experiment with held-out evaluation; "
            "candidate flags remain available for analysis and later review."
        ),
        "source": {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "data_path": data_path,
            "manifest_path": source_manifest_path,
            "data_sha256": file_sha256(local_data),
            "source_manifest_sha256": file_sha256(local_source_manifest),
        },
        "build": build,
        "outputs": outputs,
    }
    (output_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--source-manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", default="kinyalm-sft-10k-v4-split-v1")
    parser.add_argument(
        "--include-flagged",
        action="store_true",
        help="Include explicitly flagged candidates in this experimental run.",
    )
    parser.add_argument(
        "--quality-policy",
        choices=QUALITY_POLICIES,
        default="unflagged",
        help="Deterministic candidate filtering policy (default: unflagged).",
    )
    args = parser.parse_args()

    try:
        manifest = materialize_hf_candidate_sft(
            repo_id=args.repo_id,
            revision=args.revision,
            data_path=args.data_path,
            source_manifest_path=args.source_manifest_path,
            output_dir=args.output_dir.expanduser(),
            split_seed=args.split_seed,
            include_flagged=args.include_flagged,
            quality_policy=args.quality_policy,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"candidate split build failed: {error}") from error

    print(f"HF revision: {manifest['source']['resolved_revision']}")
    print(f"Selected rows: {manifest['build']['selected_rows']}")
    print(f"Splits: {manifest['build']['split_counts']}")
    print(f"Output: {args.output_dir.expanduser()}")
    print("Tier: experimental candidate-unreviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

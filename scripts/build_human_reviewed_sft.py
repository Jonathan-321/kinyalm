#!/usr/bin/env python3
"""Build the native-reviewed KinyaLM recovery SFT dataset from pinned HF data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.human_reviewed import (  # noqa: E402
    DATASET_ID,
    build_dataset,
    prepare_existing_rows,
    prepare_tessy_rows,
    write_jsonl,
)
from kinyalm.data.sft import load_jsonl  # noqa: E402
from scripts.convert_distillation_review_to_sft import (  # noqa: E402
    convert_review_rows,
)

DEFAULT_REPO_ID = "kinyalm/kinyalm-data-lake"
DEFAULT_REVISION = "9e599494681e30beac36e5d5b95ffc193d3bb99c"
SOURCE_PATHS = {
    "drafts": (
        "data/drafts/sft-distillation-production-1000-v3-final/"
        "distillation-drafts.jsonl"
    ),
    "tessy_reviews": (
        "incoming/tessymugisha/distillation-review/"
        "tessy_distillation_queue.jsonl"
    ),
    "bonheur_train": (
        "incoming/bonheurbyiringiro/bonheur-batch-001-review/"
        "promoted/train.jsonl"
    ),
    "bonheur_validation": (
        "incoming/bonheurbyiringiro/bonheur-batch-001-review/"
        "promoted/validation.jsonl"
    ),
    "gemma_corrections": (
        "incoming/bonheurbyiringiro/gemma4-corrections-batch1/"
        "corrections.jsonl"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", default=DATASET_ID)
    parser.add_argument("--minimum-records", type=int, default=400)
    parser.add_argument(
        "--include-human-approved-critic-disagreements",
        action="store_true",
        help=(
            "Include rows Tessy marked Keep when the automated critic requested "
            "repair. The manifest records this decision."
        ),
    )
    args = parser.parse_args()

    try:
        source_dir, resolved_revision = resolve_source_dir(
            repo_id=args.repo_id,
            revision=args.revision,
            source_dir=args.source_dir,
        )
        paths = {name: source_dir / path for name, path in SOURCE_PATHS.items()}
        _require_files(paths)

        tessy_reviews = load_jsonl(paths["tessy_reviews"])
        converted_tessy, withheld = convert_review_rows(
            tessy_reviews,
            "Tessy Mugisha",
            args.train_ratio,
            accept_disputed_keeps=(
                args.include_human_approved_critic_disagreements
            ),
        )
        tessy_rows = prepare_tessy_rows(
            converted_tessy,
            tessy_reviews,
            load_jsonl(paths["drafts"]),
        )

        bonheur_rows = prepare_existing_rows(
            load_jsonl(paths["bonheur_train"])
            + load_jsonl(paths["bonheur_validation"]),
            tier="human-reviewed-foundation",
            reviewer="Bonheur Byiringiro",
        )
        correction_rows = prepare_existing_rows(
            load_jsonl(paths["gemma_corrections"]),
            tier="observed-model-failure-correction",
            reviewer="Bonheur Byiringiro",
        )
        rows, report = build_dataset(
            [tessy_rows, bonheur_rows, correction_rows],
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
            minimum_records=args.minimum_records,
            include_critic_disagreements=(
                args.include_human_approved_critic_disagreements
            ),
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"build failed: {error}") from error

    output_dir = args.output_dir.expanduser()
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "dataset-manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_tier": "human-reviewed-recovery-sft",
        "human_reviewed": True,
        "training_eligible": True,
        "source": {
            "repo_id": args.repo_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "files": {
                name: {
                    "path": SOURCE_PATHS[name],
                    "sha256": sha256_path(path),
                }
                for name, path in sorted(paths.items())
            },
        },
        "build": {
            **report,
            "split_seed": args.split_seed,
            "train_ratio": args.train_ratio,
            "withheld_review_ids": withheld,
        },
        "quality_note": (
            "Every row was approved by a named fluent reviewer. Automated critic "
            "agreement is recorded as a separate curation tier, not treated as a "
            "substitute for human judgment."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"HF revision: {resolved_revision}")
    print(f"Conversations: {len(rows)}")
    print(f"Assistant turns: {report['assistant_turn_count']}")
    print(f"Multi-turn conversations: {report['multi_turn_conversation_count']}")
    print(f"Splits: {report['split_counts']}")
    print(f"Curation tiers: {report['curation_tier_counts']}")
    print(f"Output: {output_dir}")
    return 0


def resolve_source_dir(
    *,
    repo_id: str,
    revision: str,
    source_dir: Path | None,
) -> tuple[Path, str]:
    if source_dir is not None:
        return source_dir.expanduser().resolve(), revision

    from huggingface_hub import HfApi, snapshot_download

    resolved_revision = HfApi().dataset_info(repo_id, revision=revision).sha
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        allow_patterns=list(SOURCE_PATHS.values()),
    )
    return Path(downloaded), resolved_revision


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing source files: {', '.join(missing)}")


if __name__ == "__main__":
    raise SystemExit(main())

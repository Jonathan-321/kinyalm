import json
from pathlib import Path

import pytest

from scripts.upload_hf_datalake import (
    dataset_card,
    index_json,
    stage_batches,
)

BATCH_1 = "sft-drafts-2026-07-13-batch-001"
BATCH_2 = "sft-drafts-2026-07-13-batch-002"


def create_batch_files(data_root: Path, manifest_dir: Path, batch_id: str, rows: int):
    paths = [
        data_root / "drafts" / f"{batch_id}.jsonl",
        data_root / "drafts" / f"{batch_id}.summary.md",
        data_root / "reviewed" / f"{batch_id}.review.tsv",
        data_root / "packages" / f"{batch_id}-review-package.zip",
        manifest_dir / f"{batch_id}-review-sheet.json",
        manifest_dir / f"{batch_id}-review-package.json",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test\n", encoding="utf-8")
    manifest_path = manifest_dir / f"{batch_id}.json"
    manifest_path.write_text(json.dumps({"row_count": rows}), encoding="utf-8")


def test_multi_batch_card_and_index_report_one_thousand_rows():
    batch_ids = [BATCH_1, BATCH_2]
    row_counts = {BATCH_1: 286, BATCH_2: 714}
    shard_counts = {BATCH_1: 0, BATCH_2: 3}

    card = dataset_card(batch_ids=batch_ids, row_counts=row_counts)
    index = json.loads(
        index_json(
            batch_ids=batch_ids,
            row_counts=row_counts,
            shard_counts=shard_counts,
        )
    )

    assert "1K<n<10K" in card
    assert "1,000 draft rows total" in card
    assert index["total_draft_rows"] == 1000
    assert len(index["batches"]) == 2
    assert index["batches"][1]["review_shard_count"] == 3


def test_stage_batches_keeps_both_batches_and_review_shards(tmp_path):
    data_root = tmp_path / "data-root"
    manifest_dir = tmp_path / "manifests"
    stage_dir = tmp_path / "stage"
    create_batch_files(data_root, manifest_dir, BATCH_1, 286)
    create_batch_files(data_root, manifest_dir, BATCH_2, 714)
    for index in range(1, 4):
        shard = (
            data_root
            / "reviewed"
            / f"{BATCH_2}.review.part-{index:02d}-of-03.tsv"
        )
        shard.write_text("test\n", encoding="utf-8")

    staged = stage_batches(
        [BATCH_1, BATCH_2],
        stage_dir,
        local_data=data_root,
        manifest_dir=manifest_dir,
    )

    assert len(staged) == 20
    assert (stage_dir / "data" / "drafts" / BATCH_1 / f"{BATCH_1}.jsonl").exists()
    assert (stage_dir / "data" / "drafts" / BATCH_2 / f"{BATCH_2}.jsonl").exists()
    assert len(list((stage_dir / "review" / BATCH_2 / "shards").glob("*.tsv"))) == 3


def test_stage_batches_rejects_duplicate_batch_ids(tmp_path):
    with pytest.raises(ValueError, match="batch ids must be unique"):
        stage_batches([BATCH_1, BATCH_1], tmp_path / "stage")

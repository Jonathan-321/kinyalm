import json

from kinyalm.data.manifest import create_manifest, manifest_from_json, write_manifest


def test_create_manifest_for_jsonl_counts_rows_and_hashes_file(tmp_path):
    data_path = tmp_path / "draft.jsonl"
    data_path.write_text('{"id":"one"}\n\n{"id":"two"}\n', encoding="utf-8")

    manifest = create_manifest(
        data_path,
        dataset_id="draft-001",
        stage="draft",
        owner="Jonathan",
        storage="google-drive",
        storage_uri="KinyaLM Shared Data/02_generated_drafts/draft.jsonl",
        source_status="team-authored",
        review_status="needs-review",
        can_train=False,
        can_redistribute=False,
    )

    assert manifest.path_type == "file"
    assert manifest.file_count == 1
    assert manifest.row_count == 2
    assert manifest.sha256
    assert not manifest.can_train


def test_write_manifest_round_trips_json(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text("Muraho\n", encoding="utf-8")
    manifest = create_manifest(
        data_path,
        dataset_id="text-001",
        stage="raw",
        owner="Tessy",
        storage="local",
        storage_uri=str(data_path),
        source_status="approved",
        review_status="not-sure",
        can_train=False,
        can_redistribute=True,
        license="cc-by-4.0",
    )

    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    loaded = manifest_from_json(out)

    assert loaded["dataset_id"] == "text-001"
    assert loaded["license"] == "cc-by-4.0"
    assert json.loads(out.read_text(encoding="utf-8")) == loaded

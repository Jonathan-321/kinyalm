"""Create lightweight manifests for shared data artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class DataManifest:
    """Metadata record for a local, Drive, or Hugging Face data artifact."""

    dataset_id: str
    created_at_utc: str
    stage: str
    owner: str
    storage: str
    storage_uri: str
    local_path_hint: str
    path_type: str
    file_count: int
    total_bytes: int
    sha256: str | None
    row_count: int | None
    source_status: str
    review_status: str
    can_train: bool
    can_redistribute: bool
    contains_benchmark_rows: bool
    contains_private_or_sensitive_data: bool
    license: str
    attribution: str
    notes: str
    next_action: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n"


def create_manifest(
    path: str | Path,
    *,
    dataset_id: str,
    stage: str,
    owner: str,
    storage: str,
    storage_uri: str,
    source_status: str,
    review_status: str,
    can_train: bool,
    can_redistribute: bool,
    contains_benchmark_rows: bool = False,
    contains_private_or_sensitive_data: bool = False,
    license: str = "not-specified",
    attribution: str = "",
    notes: str = "",
    next_action: str = "",
    row_count: int | None = None,
) -> DataManifest:
    """Build a manifest for a file or directory without copying the data."""

    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(target)

    path_type = "directory" if target.is_dir() else "file"
    files = list(_iter_files(target))
    total_bytes = sum(item.stat().st_size for item in files)
    inferred_row_count = row_count
    if inferred_row_count is None and target.is_file() and target.suffix == ".jsonl":
        inferred_row_count = count_jsonl_rows(target)

    return DataManifest(
        dataset_id=dataset_id,
        created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        stage=stage,
        owner=owner,
        storage=storage,
        storage_uri=storage_uri,
        local_path_hint=str(target),
        path_type=path_type,
        file_count=len(files),
        total_bytes=total_bytes,
        sha256=sha256_file(target) if target.is_file() else None,
        row_count=inferred_row_count,
        source_status=source_status,
        review_status=review_status,
        can_train=can_train,
        can_redistribute=can_redistribute,
        contains_benchmark_rows=contains_benchmark_rows,
        contains_private_or_sensitive_data=contains_private_or_sensitive_data,
        license=license,
        attribution=attribution,
        notes=notes,
        next_action=next_action,
    )


def count_jsonl_rows(path: str | Path) -> int:
    """Count non-empty JSONL rows without validating the schema."""

    count = 0
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum for one file."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(manifest: DataManifest, path: str | Path) -> None:
    """Write a manifest JSON file, creating the parent directory if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def manifest_from_json(path: str | Path) -> dict[str, Any]:
    """Load a manifest JSON file as a dictionary."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())

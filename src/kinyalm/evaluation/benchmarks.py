"""Load and validate benchmark metadata without downloading datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


ALLOWED_PRIORITIES = {"high", "medium", "low"}
ALLOWED_STATUSES = {"benchmark-only"}
REQUIRED_FIELDS = {
    "id",
    "name",
    "source",
    "source_url",
    "task_type",
    "license",
    "status",
    "priority",
    "splits",
    "metrics",
    "notes",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark entry from the manifest."""

    id: str
    name: str
    source: str
    source_url: str
    task_type: str
    license: str
    status: str
    priority: str
    splits: dict[str, int | None]
    metrics: tuple[str, ...]
    notes: str
    subset: str | None = None


@dataclass(frozen=True)
class BenchmarkManifestResult:
    """Validation result for the whole benchmark manifest."""

    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_benchmark_manifest(path: str | Path) -> list[BenchmarkSpec]:
    """Load benchmark specs from a project manifest JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark manifest must be a JSON object")
    entries = raw.get("benchmarks")
    if not isinstance(entries, list):
        raise ValueError("benchmark manifest must contain a benchmarks list")
    return [_to_spec(entry, index=index) for index, entry in enumerate(entries)]


def validate_benchmark_manifest(specs: list[BenchmarkSpec]) -> BenchmarkManifestResult:
    """Check that benchmark metadata is safe to use as held-out evaluation."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, spec in enumerate(specs, start=1):
        if spec.id in seen_ids:
            errors.append(f"duplicate benchmark id: {spec.id}")
        seen_ids.add(spec.id)

        if spec.status not in ALLOWED_STATUSES:
            errors.append(f"{spec.id}: status must be benchmark-only")
        if spec.priority not in ALLOWED_PRIORITIES:
            errors.append(f"{spec.id}: priority must be high, medium, or low")
        if not spec.source_url.startswith(("https://", "http://")):
            errors.append(f"{spec.id}: source_url must be an absolute URL")
        if not spec.metrics:
            errors.append(f"{spec.id}: at least one metric is required")
        if not spec.splits:
            errors.append(f"{spec.id}: at least one split count is required")
        known_counts = 0
        for split, count in spec.splits.items():
            if not split:
                errors.append(f"{spec.id}: split name cannot be empty")
            if count is None:
                continue
            known_counts += 1
            if count <= 0:
                errors.append(f"{spec.id}: split count for {split} must be positive")
        if known_counts == 0 and "confirmation" not in spec.notes.lower():
            errors.append(
                f"{spec.id}: unknown split counts require a confirmation note"
            )

        if index == 1 and spec.priority != "high":
            errors.append("first benchmark should be a high-priority starter")

    return BenchmarkManifestResult(errors=tuple(errors))


def _to_spec(entry: Any, *, index: int) -> BenchmarkSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"benchmark entry {index} must be a JSON object")

    missing = sorted(REQUIRED_FIELDS.difference(entry))
    if missing:
        raise ValueError(
            f"benchmark entry {index} missing required fields: {', '.join(missing)}"
        )

    splits = entry["splits"]
    if not isinstance(splits, dict):
        raise ValueError(f"benchmark entry {index} splits must be an object")
    normalized_splits = {
        str(split): _optional_positive_int(
            count, f"benchmark entry {index} split {split}"
        )
        for split, count in splits.items()
    }

    metrics = entry["metrics"]
    if not isinstance(metrics, list):
        raise ValueError(f"benchmark entry {index} metrics must be a list")

    return BenchmarkSpec(
        id=_string(entry, "id", index=index),
        name=_string(entry, "name", index=index),
        source=_string(entry, "source", index=index),
        source_url=_string(entry, "source_url", index=index),
        subset=_optional_string(entry, "subset", index=index),
        task_type=_string(entry, "task_type", index=index),
        license=_string(entry, "license", index=index),
        status=_string(entry, "status", index=index),
        priority=_string(entry, "priority", index=index),
        splits=normalized_splits,
        metrics=tuple(
            _string_value(metric, f"benchmark entry {index} metric")
            for metric in metrics
        ),
        notes=_string(entry, "notes", index=index),
    )


def _string(entry: dict[str, Any], field: str, *, index: int) -> str:
    return _string_value(entry.get(field), f"benchmark entry {index} {field}")


def _optional_string(entry: dict[str, Any], field: str, *, index: int) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    return _string_value(value, f"benchmark entry {index} {field}")


def _string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be null or a positive integer")
    return value

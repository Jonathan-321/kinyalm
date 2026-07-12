from pathlib import Path

from kinyalm.evaluation import load_benchmark_manifest, validate_benchmark_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "evaluation" / "kinyarwanda_benchmarks.json"


def test_benchmark_manifest_is_valid_and_held_out():
    specs = load_benchmark_manifest(MANIFEST)
    result = validate_benchmark_manifest(specs)

    assert result.ok, result.errors
    assert len(specs) >= 7
    assert {spec.status for spec in specs} == {"benchmark-only"}


def test_benchmark_manifest_has_priority_starters():
    specs = load_benchmark_manifest(MANIFEST)
    high_priority = {spec.id for spec in specs if spec.priority == "high"}

    assert {"belebele-kin-latn", "flores200-kin-latn", "afrixnli-kin"}.issubset(
        high_priority
    )

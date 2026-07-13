"""Lightweight project health check."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from kinyalm.data.examples import load_tsv_examples
    from kinyalm.evaluation import load_benchmark_manifest, validate_benchmark_manifest

    examples_path = ROOT / "docs" / "tokenizer" / "eval-examples.tsv"
    examples = load_tsv_examples(examples_path)
    task_bank_path = ROOT / "docs" / "evaluation" / "learning-task-bank.md"
    schema_path = ROOT / "docs" / "data" / "sft-data-schema.md"
    storage_workflow_path = ROOT / "docs" / "data" / "shared-storage-workflow.md"
    hf_datalake_path = ROOT / "docs" / "data" / "huggingface-datalake.md"
    manifest_template_path = (
        ROOT / "data" / "manifests" / "dataset-manifest.template.json"
    )
    benchmark_manifest_path = (
        ROOT / "configs" / "evaluation" / "kinyarwanda_benchmarks.json"
    )
    task_count, benchmark_count = count_task_bank_rows(task_bank_path)
    benchmark_specs = load_benchmark_manifest(benchmark_manifest_path)
    benchmark_result = validate_benchmark_manifest(benchmark_specs)

    if task_count < 50:
        raise SystemExit(f"expected at least 50 tutor tasks, found {task_count}")
    if benchmark_count < 15:
        raise SystemExit(
            f"expected at least 15 benchmark-only tasks, found {benchmark_count}"
        )
    if not schema_path.exists():
        raise SystemExit(f"missing SFT schema: {schema_path}")
    if not storage_workflow_path.exists():
        raise SystemExit(f"missing shared storage workflow: {storage_workflow_path}")
    if not hf_datalake_path.exists():
        raise SystemExit(f"missing Hugging Face datalake guide: {hf_datalake_path}")
    if not manifest_template_path.exists():
        raise SystemExit(f"missing dataset manifest template: {manifest_template_path}")
    if not benchmark_result.ok:
        errors = "; ".join(benchmark_result.errors)
        raise SystemExit(f"invalid benchmark manifest: {errors}")

    print(f"Loaded {len(examples)} tokenizer evaluation examples.")
    print(
        "Loaded "
        f"{task_count} tutor evaluation tasks "
        f"({benchmark_count} benchmark-only)."
    )
    print(f"Found SFT schema at {schema_path.relative_to(ROOT)}.")
    print(
        "Found shared storage workflow at "
        f"{storage_workflow_path.relative_to(ROOT)}."
    )
    print(f"Found HF datalake guide at {hf_datalake_path.relative_to(ROOT)}.")
    print(
        "Loaded "
        f"{len(benchmark_specs)} external benchmark specs from "
        f"{benchmark_manifest_path.relative_to(ROOT)}."
    )
    print("First implementation target: tokenizer analysis on reviewed examples.")
    print(
        "Reminder: keep raw training data, benchmark test rows, and experiment "
        "artifacts out of this repo."
    )
    return 0


def count_task_bank_rows(path: Path) -> tuple[int, int]:
    task_count = 0
    benchmark_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| T"):
            continue
        task_count += 1
        if "| benchmark-only |" in line:
            benchmark_count += 1
    return task_count, benchmark_count


if __name__ == "__main__":
    sys.exit(main())

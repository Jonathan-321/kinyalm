"""Stage a teammate data contribution for the HF datalake incoming folder."""

from __future__ import annotations

from pathlib import Path
import argparse
import shutil


DEFAULT_OUT_ROOT = Path("~/KinyaLMData/hf_contributions").expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File or directory to stage.")
    parser.add_argument("--contributor", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--hf-username", default="")
    parser.add_argument("--github-username", default="")
    parser.add_argument("--source-note", default="unknown")
    parser.add_argument(
        "--training-permission",
        choices=("yes", "no", "unknown"),
        default="unknown",
    )
    parser.add_argument(
        "--redistribution-permission",
        choices=("yes", "no", "unknown"),
        default="unknown",
    )
    parser.add_argument(
        "--contains-sensitive-data",
        choices=("yes", "no", "unknown"),
        default="unknown",
    )
    parser.add_argument(
        "--contains-benchmark-rows",
        choices=("yes", "no", "unknown"),
        default="unknown",
    )
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.exists():
        raise SystemExit(f"source does not exist: {source}")

    stage_dir = (
        args.out_root.expanduser()
        / "incoming"
        / normalize_path_part(args.contributor)
        / normalize_path_part(args.batch_id)
    )
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    copy_source(source, stage_dir)
    contribution_path = stage_dir / "CONTRIBUTION.md"
    contribution_path.write_text(
        contribution_md(
            contributor=args.contributor,
            hf_username=args.hf_username,
            github_username=args.github_username,
            batch_id=args.batch_id,
            source_note=args.source_note,
            training_permission=args.training_permission,
            redistribution_permission=args.redistribution_permission,
            contains_sensitive_data=args.contains_sensitive_data,
            contains_benchmark_rows=args.contains_benchmark_rows,
        ),
        encoding="utf-8",
    )

    print(f"staged contribution: {stage_dir}")
    print("Upload this folder to the HF datalake under the same incoming/ path.")
    return 0


def normalize_path_part(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in value.strip()
    ).strip("-")
    if not normalized:
        raise SystemExit(f"invalid path value: {value!r}")
    return normalized


def copy_source(source: Path, stage_dir: Path) -> None:
    if source.is_dir():
        destination = stage_dir / source.name
        shutil.copytree(source, destination)
        return
    shutil.copy2(source, stage_dir / source.name)


def contribution_md(
    *,
    contributor: str,
    hf_username: str,
    github_username: str,
    batch_id: str,
    source_note: str,
    training_permission: str,
    redistribution_permission: str,
    contains_sensitive_data: str,
    contains_benchmark_rows: str,
) -> str:
    return f"""# Contribution

Contributor: {contributor}
HF username: {hf_username}
GitHub username: {github_username}
Batch ID: {batch_id}

## Source

{source_note}

## Permission

Can this be used for training? {training_permission}
Can this be redistributed? {redistribution_permission}
License or permission note:

## Review

Who should review this?
What should reviewers check?

## Safety

Does it contain private or sensitive data? {contains_sensitive_data}
Does it include benchmark-only rows? {contains_benchmark_rows}
"""


if __name__ == "__main__":
    raise SystemExit(main())

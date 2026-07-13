"""Create a tracked manifest for a shared data file or directory."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinyalm.data.manifest import create_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Local file or directory to describe.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--storage-uri", required=True)
    parser.add_argument("--source-status", required=True)
    parser.add_argument("--review-status", required=True)
    parser.add_argument("--can-train", choices=["yes", "no"], required=True)
    parser.add_argument("--can-redistribute", choices=["yes", "no"], required=True)
    parser.add_argument("--contains-benchmark-rows", action="store_true")
    parser.add_argument("--contains-private-or-sensitive-data", action="store_true")
    parser.add_argument("--license", default="not-specified")
    parser.add_argument("--attribution", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--next-action", default="")
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = create_manifest(
        args.path,
        dataset_id=args.dataset_id,
        stage=args.stage,
        owner=args.owner,
        storage=args.storage,
        storage_uri=args.storage_uri,
        source_status=args.source_status,
        review_status=args.review_status,
        can_train=args.can_train == "yes",
        can_redistribute=args.can_redistribute == "yes",
        contains_benchmark_rows=args.contains_benchmark_rows,
        contains_private_or_sensitive_data=args.contains_private_or_sensitive_data,
        license=args.license,
        attribution=args.attribution,
        notes=args.notes,
        next_action=args.next_action,
        row_count=args.row_count,
    )
    write_manifest(manifest, args.out)
    print(f"wrote manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

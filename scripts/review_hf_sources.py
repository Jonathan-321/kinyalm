"""Review Hugging Face dataset metadata without downloading dataset files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import argparse
import json
import ssl
import sys


CANDIDATES = [
    "mbazaNLP/kinyarwanda-language-model-dataset",
    "RogerB/Kinyarwanda_wikipedia20230920",
    "DigitalUmuganda/common-voice-kinyarwanda-text-dataset",
    "DigitalUmuganda/kinyarwanda-english-machine-translation-dataset",
    "mbazaNLP/Kinyarwanda_English_parallel_dataset",
    "Mikecyane/Kinyarwanda_chat",
    "saillab/alpaca-kinyarwanda-cleaned",
    "ChrisToukmaji/kinyarwanda_instruction_tuning",
]


@dataclass(frozen=True)
class DatasetReview:
    dataset_id: str
    url: str
    ok: bool
    error: str
    downloads: int | None
    likes: int | None
    last_modified: str
    license_values: list[str]
    language_values: list[str]
    size_values: list[str]
    task_values: list[str]
    files: list[str]
    readme_present: bool

    @property
    def has_clear_license(self) -> bool:
        return bool(self.license_values)

    @property
    def first_pass_status(self) -> str:
        if not self.ok:
            return "blocked"
        if not self.has_clear_license:
            return "blocked pending license review"
        return "investigate"


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_json(url: str, context: ssl.SSLContext) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "kinyalm-source-review/0.1"})
    with urlopen(request, timeout=30, context=context) as response:
        return json.load(response)


def list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def review_dataset(dataset_id: str, context: ssl.SSLContext) -> DatasetReview:
    url = f"https://huggingface.co/datasets/{dataset_id}"
    api_url = f"https://huggingface.co/api/datasets/{dataset_id}"
    try:
        metadata = fetch_json(api_url, context)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return DatasetReview(
            dataset_id=dataset_id,
            url=url,
            ok=False,
            error=repr(error),
            downloads=None,
            likes=None,
            last_modified="",
            license_values=[],
            language_values=[],
            size_values=[],
            task_values=[],
            files=[],
            readme_present=False,
        )

    tags = list_value(metadata.get("tags"))
    card_data = metadata.get("cardData") or {}
    siblings = metadata.get("siblings") or []

    license_values = list_value(card_data.get("license"))
    license_values.extend(
        tag.removeprefix("license:")
        for tag in tags
        if str(tag).startswith("license:")
    )
    license_values = sorted(set(license_values))

    language_values = list_value(card_data.get("language"))
    language_values.extend(
        tag.removeprefix("language:")
        for tag in tags
        if str(tag).startswith("language:")
    )
    language_values = sorted(set(language_values))

    size_values = [
        tag.removeprefix("size_categories:")
        for tag in tags
        if str(tag).startswith("size_categories:")
    ]

    task_values = list_value(card_data.get("task_categories"))
    task_values.extend(list_value(card_data.get("task_ids")))
    task_values = sorted(set(task_values))

    files = [item.get("rfilename", "") for item in siblings]
    files = [item for item in files if item]

    return DatasetReview(
        dataset_id=dataset_id,
        url=url,
        ok=True,
        error="",
        downloads=metadata.get("downloads"),
        likes=metadata.get("likes"),
        last_modified=str(metadata.get("lastModified") or ""),
        license_values=license_values,
        language_values=language_values,
        size_values=size_values,
        task_values=task_values,
        files=files,
        readme_present="README.md" in files,
    )


def markdown_list(values: list[str]) -> str:
    if not values:
        return "none visible"
    return ", ".join(value.replace("<", "&lt;").replace(">", "&gt;") for value in values)


def render_report(reviews: list[DatasetReview]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Hugging Face Source Review",
        "",
        f"Generated: {generated}",
        "",
        "This report reviews Hugging Face dataset metadata and dataset cards. It",
        "does not download dataset contents and does not approve any dataset for",
        "training by itself.",
        "",
        "## Summary",
        "",
        "| Dataset | Visible License | Languages | Size | First-Pass Status |",
        "| --- | --- | --- | --- | --- |",
    ]

    for review in reviews:
        lines.append(
            "| "
            f"[{review.dataset_id}]({review.url}) | "
            f"{markdown_list(review.license_values)} | "
            f"{markdown_list(review.language_values)} | "
            f"{markdown_list(review.size_values)} | "
            f"{review.first_pass_status} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Notes",
            "",
        ]
    )

    for review in reviews:
        lines.extend(
            [
                f"### [{review.dataset_id}]({review.url})",
                "",
                f"- First-pass status: `{review.first_pass_status}`",
                f"- Downloads: {review.downloads if review.downloads is not None else 'unknown'}",
                f"- Likes: {review.likes if review.likes is not None else 'unknown'}",
                f"- Last modified: {review.last_modified or 'unknown'}",
                f"- Visible license: {markdown_list(review.license_values)}",
                f"- Visible languages: {markdown_list(review.language_values)}",
                f"- Visible size: {markdown_list(review.size_values)}",
                f"- Visible tasks: {markdown_list(review.task_values)}",
                f"- Files visible from metadata: {', '.join(review.files[:8]) if review.files else 'none visible'}",
                f"- Dataset card README visible: {'yes' if review.readme_present else 'no'}",
            ]
        )
        if review.error:
            lines.append(f"- Error: {review.error}")
        lines.append("")

    lines.extend(
        [
            "## Working Interpretation",
            "",
            "- Datasets with visible permissive or Creative Commons licenses can move to",
            "  deeper review, but still need source/provenance and quality checks.",
            "- Datasets with no visible license should stay blocked for training until a",
            "  license or permission is confirmed.",
            "- Chat or instruction datasets need extra privacy, consent, translation, and",
            "  quality review before use.",
            "- This report should be treated as a starting point for `docs/data/source-log.md`,",
            "  not as legal approval.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional Markdown output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = ssl_context()
    reviews = [review_dataset(dataset_id, context) for dataset_id in CANDIDATES]
    report = render_report(reviews)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Turn two baseline-probe outputs into a before/after comparison document.

Both inputs are JSONL of {"prompt", "completion"} records (as written by
scripts/baseline_probe.py). Rows are matched by prompt text. Produces a
Markdown document (paste into GitHub, a Google Doc, or the report) and,
with --html, a standalone self-contained HTML page for easy sharing.

    python scripts/compare_probes.py \
        --before baseline-2b-preFT.jsonl \
        --after  baseline-2b-postFT.jsonl \
        --output docs/model/before-after-2b.md --html
"""

from __future__ import annotations

from pathlib import Path
import argparse
import html
import json


def load(path: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        rows[record["prompt"]] = record.get("completion", "")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--title", default="Kinyarwanda Tutor: Before vs After Fine-Tuning")
    parser.add_argument("--before-label", default="Base model (no fine-tuning)")
    parser.add_argument("--after-label", default="After fine-tuning")
    parser.add_argument("--html", action="store_true",
                        help="Also write a standalone .html next to the markdown")
    return parser.parse_args()


def build_markdown(args, before: dict[str, str], after: dict[str, str]) -> str:
    lines = [f"# {args.title}", ""]
    lines.append(f"Same prompts run through the model before and after fine-tuning. "
                 f"Left is **{args.before_label}**, right is **{args.after_label}**.")
    lines.append("")
    for index, prompt in enumerate(before, start=1):
        lines.append(f"## {index}. {prompt}")
        lines.append("")
        lines.append(f"**{args.before_label}:**")
        lines.append("")
        lines.append("> " + before[prompt].replace("\n", "\n> "))
        lines.append("")
        lines.append(f"**{args.after_label}:**")
        lines.append("")
        after_text = after.get(prompt, "_(no matching prompt in the after file)_")
        lines.append("> " + after_text.replace("\n", "\n> "))
        lines.append("")
    return "\n".join(lines) + "\n"


def build_html(args, before: dict[str, str], after: dict[str, str]) -> str:
    def esc(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    cards = []
    for index, prompt in enumerate(before, start=1):
        after_text = after.get(prompt, "(no matching prompt in the after file)")
        cards.append(f"""
      <div class="card">
        <div class="q">{index}. {esc(prompt)}</div>
        <div class="cols">
          <div class="col before"><h3>{esc(args.before_label)}</h3><p>{esc(before[prompt])}</p></div>
          <div class="col after"><h3>{esc(args.after_label)}</h3><p>{esc(after_text)}</p></div>
        </div>
      </div>""")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }}
  h1 {{ font-size: 1.5rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .q {{ font-weight: 600; margin-bottom: .75rem; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .col {{ padding: .75rem; border-radius: 6px; }}
  .before {{ background: #fbeaea; }}
  .after {{ background: #eaf5ec; }}
  .col h3 {{ margin: 0 0 .5rem; font-size: .8rem; text-transform: uppercase;
             letter-spacing: .04em; color: #555; }}
  .col p {{ margin: 0; white-space: normal; }}
  @media (max-width: 640px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
  <h1>{html.escape(args.title)}</h1>
  <p>Same prompts, before and after fine-tuning.</p>
  {"".join(cards)}
</body></html>
"""


def main() -> int:
    args = parse_args()
    before = load(args.before)
    after = load(args.after)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(args, before, after), encoding="utf-8")
    print(f"markdown written: {out_path}")

    if args.html:
        html_path = out_path.with_suffix(".html")
        html_path.write_text(build_html(args, before, after), encoding="utf-8")
        print(f"html written: {html_path}")

    matched = sum(1 for p in before if p in after)
    print(f"prompts: {len(before)} before, {len(after)} after, {matched} matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

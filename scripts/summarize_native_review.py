#!/usr/bin/env python3
"""Aggregate native-speaker blind reviews and enforce base-model gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinyalm.evaluation.native_review import (
    render_native_review_markdown,
    summarize_native_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--baseline-candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--morphology-pass-rate", type=float, default=0.7)
    parser.add_argument("--morphology-correctness", type=float, default=4.0)
    parser.add_argument("--minimum-candidates-for-cpt", type=int, default=2)
    parser.add_argument(
        "--review-kind",
        choices=("native", "model-assisted"),
        default="native",
        help="Label the evidence source accurately in generated reports.",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--gate-candidate-id",
        help="Exit nonzero unless this candidate meets or exceeds the base gates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.review_kind == "model-assisted":
        report_title = "Normalized Model-Assisted Evaluation Results"
        scope = (
            "Preliminary scores from a blinded model judge; native-speaker "
            "validation is still required."
        )
    else:
        report_title = "Normalized Native-Review Results"
        scope = "Native-speaker scores only; no model-generated quality grades."
    summary = summarize_native_review(
        args.review_csv,
        args.blind_key,
        baseline_candidate_id=args.baseline_candidate_id,
        morphology_pass_rate=args.morphology_pass_rate,
        morphology_correctness=args.morphology_correctness,
        minimum_candidates_for_cpt=args.minimum_candidates_for_cpt,
        report_title=report_title,
        scope=scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_native_review_markdown(summary), encoding="utf-8"
        )
        print(f"Wrote {args.markdown_output}")
    if args.require_complete and not summary["complete"]:
        print("Review is incomplete.")
        return 2
    if args.gate_candidate_id:
        decision = summary["candidate_decisions"].get(args.gate_candidate_id)
        if not decision:
            raise SystemExit(
                f"gate candidate not present in review: {args.gate_candidate_id}"
            )
        if decision["decision"] != "continue":
            print(
                f"Candidate gate failed: {args.gate_candidate_id} "
                f"({decision['decision']})"
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

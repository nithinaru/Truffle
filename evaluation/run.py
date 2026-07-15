"""Score a file of validated offline predictions.

Usage::

    python -m evaluation.run --predictions predictions.jsonl

Each prediction line is ``{"case_id": "...", "result": <ParseResult>}``.
This command intentionally has no live-client option and performs no network
access.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.io import load_benchmark, load_predictions, report_json
from evaluation.score import score_predictions

DEFAULT_BENCHMARK = Path(__file__).with_name("benchmark.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score validated Truffle parse predictions.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Benchmark JSONL path (defaults to the starter corpus).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Offline prediction JSONL path; this command never calls an LLM.",
    )
    parser.add_argument("--output", type=Path, help="Write report JSON here instead of stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = load_benchmark(args.benchmark)
    records = load_predictions(args.predictions)
    report = score_predictions(cases, {case_id: row.result for case_id, row in records.items()})
    rendered = report_json(report) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

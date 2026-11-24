"""Command-line interface: score files, build rewrite plans, compare drafts.

The CLI is deliberately independent of the MCP server so it can be used in
scripts and CI without starting a protocol loop. ``open-detector-mcp`` remains
the MCP entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from . import __version__
from .analysis import classify, compare, rewrite_plan
from .detector import (
    APPROX_MB,
    MODEL_LABEL,
    MODEL_REPO,
    get_detector,
    is_model_cached,
    prefetch_model,
    preload_dependencies,
)
from .extract import extract_file
from .text import split_document


def _paragraphs_from_text(text: str, keep_references: bool):
    paragraphs = split_document(text, exclude_references=not keep_references)
    return paragraphs, {"type": "text", "paragraphs_total": len(paragraphs)}


def _paragraphs_from_path(path: str, keep_references: bool):
    if path == "-":
        return _paragraphs_from_text(sys.stdin.read(), keep_references)
    return extract_file(path, exclude_references=not keep_references)


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _print_score(report: dict, source: dict, include_text: bool) -> None:
    overall = report["overall"]
    bands = report["bands"]
    header = source.get("path") or "text"
    print(f"Open-Detector  |  {MODEL_REPO}")
    print(f"{header}")
    print(
        f"Overall: {_fmt_pct(overall['ai_pct'])} AI  "
        f"[{overall['verdict']}]   "
        f"{overall['paragraphs_scored']}/{overall['paragraphs_total']} paragraphs scored, "
        f"{overall['words']:,} words"
    )
    print(
        f"Bands: high {bands['high']}  medium {bands['medium']}  "
        f"low {bands['low']}  human {bands['human']}"
    )
    if report.get("style_flag_totals"):
        flags = ", ".join(
            f"{name} ×{count}" for name, count in
            sorted(report["style_flag_totals"].items(), key=lambda kv: -kv[1])
        )
        print(f"Style flags: {flags}")
    print()

    rows = report["paragraphs"]
    if not rows:
        print("No paragraphs found.")
        return
    width = max(4, len(str(rows[-1]["index"])))
    print(f"{'#':>{width}}  {'AI%':>6}  {'words':>5}  paragraph")
    for entry in rows:
        if entry["label"] == "Skipped":
            print(
                f"{entry['index']:>{width}}  {'skip':>6}  {entry['words']:>5}  "
                f"{entry['preview']}"
            )
            continue
        text = entry.get("text") if include_text else entry["preview"]
        suffix = ""
        if entry["style_flags"]:
            suffix = f"  [{', '.join(entry['style_flags'])}]"
        print(
            f"{entry['index']:>{width}}  {entry['ai_pct']:>6.1f}  "
            f"{entry['words']:>5}  {text}{suffix}"
        )

    spans = report.get("span_analysis")
    if spans and spans["candidates"]:
        print("\nCandidate spans (weak localisation signal):")
        for span in spans["candidates"]:
            print(
                f"  p{span['paragraph_index']} s{span['sentence_start']}-"
                f"{span['sentence_end']}  {span['ai_pct']:>5.1f}%  {span['preview']}"
            )


def _print_plan(plan: dict, source: dict) -> None:
    overall = plan["overall"]
    print(f"Open-Detector rewrite plan  |  {source.get('path') or 'text'}")
    print(
        f"Overall: {_fmt_pct(overall['ai_pct'])} AI [{overall['verdict']}]  ·  "
        f"{plan['target_count']} paragraph(s) at or above threshold, "
        f"showing {plan['returned']}"
    )
    for target in plan["targets"]:
        print()
        print(
            f"--- Priority {target['priority']} · paragraph {target['index']} · "
            f"{target['ai_pct']:.1f}% AI · {target['words']} words ---"
        )
        body = target.get("text") or target["preview"]
        print(textwrap.fill(body, width=100))
        for issue in target["style_issues"]:
            print(f"  • {issue['tag']}: {issue['detail']}")
            print(textwrap.fill(f"    → {issue['hint']}", width=100,
                                initial_indent="", subsequent_indent="      "))
    print()
    for step in plan["next_steps"]:
        print(textwrap.fill(f"- {step}", width=100, subsequent_indent="  "))


def _print_compare(result: dict) -> None:
    print(f"Before: {_fmt_pct(result['before']['ai_pct'])} AI "
          f"[{result['before']['verdict']}]")
    print(f"After:  {_fmt_pct(result['after']['ai_pct'])} AI "
          f"[{result['after']['verdict']}]")
    delta = result["delta_ai_pct"]
    print(f"Delta:  {'n/a' if delta is None else f'{delta:+.1f}'}  ({result['result']})")
    counts = result["paragraphs"]
    print(
        f"Paragraphs: {counts['compared']} compared · {counts['improved']} improved · "
        f"{counts['regressed']} regressed · {counts['unchanged']} unchanged · "
        f"{counts['added']} added · {counts['removed']} removed"
    )
    if result["paragraph_deltas"]:
        print()
        print(f"{'#':>7}  {'before':>7}  {'after':>7}  {'delta':>7}  paragraph")
        for pair in result["paragraph_deltas"]:
            index = (
                str(pair["before_index"])
                if pair["before_index"] == pair["after_index"]
                else f"{pair['before_index']}→{pair['after_index']}"
            )
            print(
                f"{index:>7}  {pair['before_ai_pct']:>6.1f}%  "
                f"{pair['after_ai_pct']:>6.1f}%  {pair['delta']:>+6.1f}  "
                f"{pair['preview']}"
            )
    for added in result["added_paragraphs"]:
        print(f"  added   p{added['index']}  {added['ai_pct']:>5.1f}%  {added['preview']}")
    for removed in result["removed_paragraphs"]:
        print(f"  removed p{removed['index']}  {removed['ai_pct']:>5.1f}%  {removed['preview']}")
    for warning in result["warnings"]:
        print(textwrap.fill(f"warning: {warning}", width=100))


def _dump(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _cmd_score(args) -> int:
    if args.text is not None and args.path:
        raise ValueError("Pass either a path or --text, not both.")
    if args.text is None and not args.path:
        raise ValueError("Pass a file path, '-' for stdin, or --text.")
    if args.text is not None:
        paragraphs, source = _paragraphs_from_text(args.text, args.keep_references)
    else:
        paragraphs, source = _paragraphs_from_path(args.path or "-", args.keep_references)
    report = classify(
        paragraphs,
        get_detector(),
        detail=args.detail,
        include_text=args.include_text,
    )
    if args.json:
        _dump({"source": source, **report})
    else:
        _print_score(report, source, args.include_text)
    return 0


def _cmd_plan(args) -> int:
    paragraphs, source = _paragraphs_from_path(args.path, args.keep_references)
    plan = rewrite_plan(
        paragraphs,
        get_detector(),
        top_n=args.top,
        min_ai_pct=args.min_ai,
        include_text=not args.no_text,
    )
    if args.json:
        _dump({"source": source, **plan})
    else:
        _print_plan(plan, source)
    return 0


def _cmd_compare(args) -> int:
    before, _ = _paragraphs_from_path(args.a, args.keep_references)
    after, _ = _paragraphs_from_path(args.b, args.keep_references)
    result = compare(before, after, get_detector())
    if args.json:
        _dump(result)
    else:
        _print_compare(result)
    return 0


def _cmd_status(_args) -> int:
    cached = is_model_cached()
    print(f"Model:    {MODEL_LABEL}")
    print(f"Repo:     {MODEL_REPO}")
    if APPROX_MB:
        print(f"Size:     ~{APPROX_MB} MB (one-time download)")
    print(f"Cached:   {'yes' if cached else 'no'}")
    print(f"Device:   {os.environ.get('OPEN_DETECTOR_DEVICE', 'auto')}")
    if not cached:
        print("Run 'open-detector download' to fetch the weights, or the first score "
              "call will download them.")
    return 0


def _cmd_download(_args) -> int:
    preload_dependencies()
    ok = prefetch_model()
    print("Model ready." if ok else "Download failed; the first score call will retry.")
    return 0 if ok else 1


def _cmd_mcp(_args) -> int:
    from .server import main as server_main

    server_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-detector",
        description=(
            "Paragraph-level AI-text detection for academic papers "
            "(Open-Detector BERT model)."
        ),
        epilog=(
            "MCP server: open-detector-mcp\n"
            "Examples:\n"
            "  open-detector score paper.docx\n"
            "  open-detector plan paper.docx --top 5\n"
            "  open-detector compare original.txt revised.txt\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_score = sub.add_parser("score", help="score a file or text")
    p_score.add_argument("path", nargs="?", help="file path, or - for stdin")
    p_score.add_argument("--text", help="score this text instead of a file")
    p_score.add_argument("--json", action="store_true", help="machine-readable output")
    p_score.add_argument("--detail", choices=("paragraph", "windows"),
                         default="paragraph")
    p_score.add_argument("--include-text", action="store_true",
                         help="print each paragraph's full text")
    p_score.add_argument("--keep-references", action="store_true",
                         help="score the bibliography instead of excluding it")
    p_score.set_defaults(func=_cmd_score)

    p_plan = sub.add_parser("plan", help="rank paragraphs to rewrite")
    p_plan.add_argument("path", help="paper file (.txt/.md/.tex/.docx/.pdf)")
    p_plan.add_argument("--top", type=int, default=5, help="number of targets (default 5)")
    p_plan.add_argument("--min-ai", type=float, default=50.0,
                        help="minimum ai_pct to include (default 50)")
    p_plan.add_argument("--json", action="store_true", help="machine-readable output")
    p_plan.add_argument("--no-text", action="store_true",
                        help="omit full target text from the output")
    p_plan.add_argument("--keep-references", action="store_true",
                        help="include the bibliography")
    p_plan.set_defaults(func=_cmd_plan)

    p_compare = sub.add_parser("compare", help="compare two versions of a paper")
    p_compare.add_argument("a", help="original file")
    p_compare.add_argument("b", help="revised file")
    p_compare.add_argument("--json", action="store_true", help="machine-readable output")
    p_compare.add_argument("--keep-references", action="store_true",
                           help="include the bibliography")
    p_compare.set_defaults(func=_cmd_compare)

    p_status = sub.add_parser("status", help="show model cache/device status")
    p_status.set_defaults(func=_cmd_status)

    p_download = sub.add_parser("download", help="download model weights")
    p_download.set_defaults(func=_cmd_download)

    p_mcp = sub.add_parser("mcp", help="run the MCP server (same as open-detector-mcp)")
    p_mcp.set_defaults(func=_cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return
    try:
        code = args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

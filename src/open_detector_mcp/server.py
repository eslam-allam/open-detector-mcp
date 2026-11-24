"""MCP server for Open-Detector.

Exposes paragraph-level AI-text detection and a humanization loop over stdio so
any MCP client (opencode, Claude Desktop/Code, Codex, etc.) can score academic
papers, locate the paragraphs worth rewriting, and verify that a rewrite actually
helped.

Tools:
  * get_model_status  - is the model cached, downloading, loaded; optional warmup
  * detect_ai_text    - score a string, paragraph by paragraph
  * detect_ai_file    - score .txt/.md/.tex/.docx/.pdf without loading it into context
  * get_rewrite_plan  - worst paragraphs, full text + concrete style issues
  * read_paragraphs   - fetch exact paragraph text by stable index
  * compare_texts     - before/after scores for two strings
  * compare_files     - before/after scores for two files

Run: open-detector-mcp   (or: python -m open_detector_mcp)
"""

from __future__ import annotations

import functools
import os
import sys
import threading

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from . import __version__
from .analysis import classify, compare, rewrite_plan
from .detector import (
    APPROX_MB,
    MAX_TOKENS,
    MIN_WORDS,
    MODEL_LABEL,
    MODEL_REPO,
    get_detector,
    is_model_cached,
    prefetch_model,
    preload_dependencies,
)
from .extract import extract_file
from .text import split_document, word_count

INSTRUCTIONS = """\
Open-Detector scores English academic text for AI-like style, paragraph by paragraph.

Reliable unit: the paragraph. Single-sentence scores are noisy; only use
detail="windows" to locate candidate spans inside a paragraph that is already
flagged, and re-score the whole paragraph after editing.

Humanizing workflow:
1. get_model_status - first run downloads ~440 MB. Pass warmup=true to load the
   model before the first scoring call.
2. detect_ai_file(path) or detect_ai_text(text) - overall ai_pct and per-paragraph
   scores. Paragraph indices are stable for the same input.
3. get_rewrite_plan(path=... or text=...) - ranked paragraphs with full text and
   concrete style issues. Rewrite structure and specificity, not just synonyms;
   light paraphrases do not move the score.
4. Preserve claims, numbers, citations and section structure. Never invent evidence.
5. compare_texts(original, rewrite) or compare_files(before_path, after_path) - the
   delta tells you whether the rewrite helped. Repeat until every paragraph is
   below 50% AI, ideally below 20%.

References are excluded by default (exclude_references=true). Scores are stylistic
evidence, not proof of misconduct.
"""

mcp = FastMCP("open-detector", instructions=INSTRUCTIONS, log_level="WARNING")

# Heavy imports (torch/transformers) happen in one dedicated startup thread so the
# MCP initialize handshake returns immediately and first-use imports never run on a
# request worker thread (a known deadlock hazard for C-extension stacks).
_preload_done = threading.Event()
_preload_error: Exception | None = None
_model_state = "unknown"
_state_lock = threading.Lock()

READY_TIMEOUT = float(os.environ.get("OPEN_DETECTOR_READY_TIMEOUT", "900"))

_STATE_NOTES = {
    "ready": "Ready - weights are cached locally.",
    "downloading": (
        "Downloading in the background now. A detect call will wait for it to "
        "finish; tell the user this may take a few minutes."
    ),
    "absent": (
        "Not downloaded. The next detect call fetches it from Hugging Face "
        "(one-time, ~440 MB) and blocks until it is cached. Warn the user first."
    ),
    "unknown": "Server is still starting up; call again in a moment.",
}


def _set_state(state: str) -> None:
    global _model_state
    with _state_lock:
        _model_state = state


def _preload_in_background() -> None:
    global _preload_error
    try:
        _set_state("ready" if is_model_cached() else "absent")
        preload_dependencies()
        if not is_model_cached():
            _set_state("downloading")
            _set_state("ready" if prefetch_model() else "absent")
    except Exception as exc:  # defensive: keep the server alive, surface on first call
        _preload_error = exc
    finally:
        _preload_done.set()


def _await_ready() -> None:
    """Block until startup preload finished. Call from a worker thread only."""
    timeout = READY_TIMEOUT if READY_TIMEOUT > 0 else None
    if not _preload_done.wait(timeout):
        raise TimeoutError(
            f"open-detector: model startup did not finish within {READY_TIMEOUT:.0f}s "
            f"(state: {_model_state}). Check the network and restart the server, or "
            "raise OPEN_DETECTOR_READY_TIMEOUT."
        )
    if _preload_error is not None:
        raise RuntimeError(f"open-detector: model startup failed: {_preload_error}")


async def _offload(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def _text_source(paragraphs: list[str]) -> dict:
    return {"type": "text", "paragraphs_total": len(paragraphs)}


def _detect_text(text: str, detail: str, include_text: bool, exclude_references: bool) -> dict:
    _await_ready()
    paragraphs = split_document(text, exclude_references=exclude_references)
    report = classify(paragraphs, get_detector(), detail=detail, include_text=include_text)
    return {"source": _text_source(paragraphs), **report}


def _detect_file(path: str, detail: str, include_text: bool, exclude_references: bool) -> dict:
    # Read before waiting on the model: a bad path should fail in milliseconds.
    paragraphs, source = extract_file(path, exclude_references=exclude_references)
    _await_ready()
    report = classify(paragraphs, get_detector(), detail=detail, include_text=include_text)
    return {"source": source, **report}


def _plan(text: str | None, path: str | None, top_n: int, min_ai_pct: float,
          include_text: bool, exclude_references: bool) -> dict:
    if (text is None) == (path is None):
        raise ValueError("Pass exactly one of 'text' or 'path'.")
    if path is not None:
        paragraphs, source = extract_file(path, exclude_references=exclude_references)
    else:
        paragraphs = split_document(text or "", exclude_references=exclude_references)
        source = _text_source(paragraphs)
    _await_ready()
    result = rewrite_plan(
        paragraphs,
        get_detector(),
        top_n=top_n,
        min_ai_pct=min_ai_pct,
        include_text=include_text,
    )
    return {"source": source, **result}


def _compare_texts(text_a: str, text_b: str, exclude_references: bool) -> dict:
    _await_ready()
    before = split_document(text_a, exclude_references=exclude_references)
    after = split_document(text_b, exclude_references=exclude_references)
    result = compare(before, after, get_detector())
    return {
        "before": {**result["before"], "source": _text_source(before)},
        "after": {**result["after"], "source": _text_source(after)},
        **{k: v for k, v in result.items() if k not in ("before", "after")},
    }


def _compare_files(path_a: str, path_b: str, exclude_references: bool) -> dict:
    before, source_a = extract_file(path_a, exclude_references=exclude_references)
    after, source_b = extract_file(path_b, exclude_references=exclude_references)
    _await_ready()
    result = compare(before, after, get_detector())
    return {
        "before": {**result["before"], "source": source_a},
        "after": {**result["after"], "source": source_b},
        **{k: v for k, v in result.items() if k not in ("before", "after")},
    }


def _read_paragraphs(path: str, indices: list[int], exclude_references: bool) -> dict:
    if not indices:
        raise ValueError("Pass at least one paragraph index.")
    if len(indices) > 50:
        raise ValueError("Request at most 50 paragraphs per call to bound response size.")
    paragraphs, source = extract_file(path, exclude_references=exclude_references)
    total = len(paragraphs)
    bad = [i for i in indices if not isinstance(i, int) or i < 0 or i >= total]
    if bad:
        raise ValueError(f"Paragraph index out of range: {bad}. Valid range is 0..{total - 1}.")
    return {
        "source": source,
        "paragraphs": [
            {"index": i, "words": word_count(paragraphs[i]), "text": paragraphs[i]}
            for i in indices
        ],
    }


def _status(warmup: bool) -> dict:
    state = _model_state
    if _preload_done.is_set():
        state = "ready" if is_model_cached() else "absent"
        _set_state(state)

    result = {
        "model": MODEL_LABEL,
        "repo": MODEL_REPO,
        "approx_mb": APPROX_MB,
        "max_tokens": MAX_TOKENS,
        "min_paragraph_words": MIN_WORDS,
        "status": state,
        "downloaded": state == "ready",
        "note": _STATE_NOTES.get(state, _STATE_NOTES["unknown"]),
        "device_preference": os.environ.get("OPEN_DETECTOR_DEVICE", "auto"),
    }
    if warmup:
        _await_ready()
        detector = get_detector()
        detector.warmup()
        state = "ready" if is_model_cached() else state
        _set_state(state)
        result.update(
            status=state,
            downloaded=state == "ready",
            loaded=True,
            device=str(detector.device),
            note="Model loaded and warmed up; detection will start immediately.",
        )
    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_model_status(warmup: bool = False) -> dict:
    """Check whether the detector is cached, downloading, or loaded.

    Returns instantly unless warmup=true. Call this before the first detection so
    you can warn the user about the one-time ~440 MB Hugging Face download.
    Set warmup=true to load the model into memory before scoring (adds a few
    seconds, avoids a slow first detection).
    """
    return await _offload(_status, warmup)


@mcp.tool()
async def detect_ai_text(
    text: str,
    detail: str = "paragraph",
    include_text: bool = False,
    exclude_references: bool = True,
) -> dict:
    """Score a string of academic text for AI-like style, paragraph by paragraph.

    Args:
        text: The text to analyse.
        detail: 'paragraph' (default, reliable) or 'windows' (adds weak
            3-sentence candidate spans inside flagged paragraphs; doubles cost).
        include_text: Include each paragraph's full text in the response. Off by
            default to keep context small.
        exclude_references: Drop everything after a References/Bibliography
            heading (default true; reference lists skew the average).

    Returns overall ai_pct (0-100, higher = more AI-like), verdict, paragraph
    bands, per-paragraph scores with stable indices, and deterministic style flags.
    """
    return await _offload(_detect_text, text, detail, include_text, exclude_references)


@mcp.tool()
async def detect_ai_file(
    path: str,
    detail: str = "paragraph",
    include_text: bool = False,
    exclude_references: bool = True,
) -> dict:
    """Score a local paper (.txt, .md, .tex, .docx, .pdf) without loading it into context.

    Args:
        path: Path to the file (~ is expanded).
        detail: 'paragraph' (default) or 'windows'.
        include_text: Include each paragraph's full text (large; usually use
            get_rewrite_plan or read_paragraphs instead).
        exclude_references: Drop the bibliography (default true).

    DOCX body paragraphs and PDF page text are extracted; DOCX tables are skipped.
    Same response shape as detect_ai_text, plus source metadata.
    """
    return await _offload(_detect_file, path, detail, include_text, exclude_references)


@mcp.tool()
async def get_rewrite_plan(
    path: str | None = None,
    text: str | None = None,
    top_n: int = 5,
    min_ai_pct: float = 50.0,
    include_text: bool = True,
    exclude_references: bool = True,
) -> dict:
    """Rank the paragraphs most worth rewriting and explain what to change.

    Pass exactly one of 'path' or 'text'. Returns the overall score plus targets
    sorted by ai_pct, each with full paragraph text, deterministic style issues
    (uniform sentence length, stock transitions, buzzwords, em dashes, etc.) and
    concrete hints. This is the best tool to start a humanization pass.

    Args:
        path: Paper file (.txt/.md/.tex/.docx/.pdf).
        text: Text to analyse instead of a file.
        top_n: Maximum number of targets to return (default 5).
        min_ai_pct: Only include paragraphs at or above this score (default 50).
        include_text: Include each target's full text (default true).
        exclude_references: Drop the bibliography (default true).
    """
    return await _offload(
        _plan, text, path, top_n, min_ai_pct, include_text, exclude_references
    )


@mcp.tool()
async def read_paragraphs(
    path: str,
    indices: list[int],
    exclude_references: bool = True,
) -> dict:
    """Fetch exact paragraph text by index (for rewriting without loading the whole paper).

    Indices come from detect_ai_file / get_rewrite_plan and are stable for the
    same file and exclude_references setting. Request at most 50 at a time.
    """
    return await _offload(_read_paragraphs, path, indices, exclude_references)


@mcp.tool()
async def compare_texts(
    text_a: str,
    text_b: str,
    exclude_references: bool = True,
) -> dict:
    """Score two versions of a text and report the before/after delta.

    Paragraphs are paired by index, so use it for paragraph-level rewrites or
    in-place edits that preserve paragraph boundaries. Negative delta_ai_pct is
    an improvement. Also returns per-paragraph deltas and improved/regressed counts.
    """
    return await _offload(_compare_texts, text_a, text_b, exclude_references)


@mcp.tool()
async def compare_files(
    path_a: str,
    path_b: str,
    exclude_references: bool = True,
) -> dict:
    """Compare two versions of a paper on disk (e.g. original.docx vs revised.docx).

    Same response as compare_texts. Paragraph counts should match for index
    pairing to be meaningful; the response warns when they differ.
    """
    return await _offload(_compare_files, path_a, path_b, exclude_references)


@mcp.prompt()
def humanize_paper(path: str) -> str:
    """Guided workflow for lowering a paper's detected AI score without changing its claims."""
    return f"""\
You are helping the user revise the academic paper at: {path}

Work in this order and do not skip the re-scoring step:
1. Call get_model_status(warmup=true) once so the model is ready.
2. Call get_rewrite_plan(path="{path}", top_n=5). Read the style_issues carefully.
3. For each target, rewrite the paragraph so that:
   - the claims, numbers, citations and hedging stay accurate;
   - sentence lengths and openings vary;
   - stock connectives and LLM-favoured wording are replaced with plain, specific prose;
   - no evidence, citation or result is invented.
   Make structural edits (merge/split/reorder sentences, lead with the finding).
   Synonym swaps alone will not change the score.
4. Call compare_texts(original_paragraph, rewritten_paragraph) after each rewrite,
   or compare_files(original_path, edited_path) after a batch. If a paragraph
   regresses or stays above 50%, rewrite it again rather than reverting blindly.
5. Finish when every paragraph is below 50% AI, ideally below 20%, and report the
   before/after scores. Remind the user that this is stylistic evidence, not proof
   of misconduct, and that they remain responsible for the content.
"""


@mcp.resource("open-detector://guide")
def guide() -> str:
    """Scoring semantics, thresholds, and the humanization loop."""
    return INSTRUCTIONS + """\
Score bands: high >= 80, medium 50-79, low 20-49, human < 20 (ai_pct).
Verdicts: LIKELY AI >= 60, MIXED 40-59, LIKELY HUMAN < 40.

The model is followsci/bert-ai-text-detector (BERT-base, MIT), trained on
paragraph-level academic text. It is deliberately insensitive to light
paraphrase: substantive edits are required. Long paragraphs are split on sentence
boundaries at 480 tokens and combined by token count; paragraph-level scores are
the reliable unit. References are excluded by default.
"""


def main() -> None:
    args = sys.argv[1:]
    if "--version" in args:
        print(f"open-detector-mcp {__version__}")
        return
    if "-h" in args or "--help" in args:
        print(
            "open-detector-mcp - MCP stdio server for Open-Detector\n\n"
            "This command speaks the Model Context Protocol on stdin/stdout; it\n"
            "takes no options during normal use. Configure your MCP client with:\n"
            '  {"command": "open-detector-mcp"}\n\n'
            "CLI equivalent: open-detector --help"
        )
        return
    threading.Thread(
        target=_preload_in_background, name="open-detector-preload", daemon=True
    ).start()
    mcp.run()


if __name__ == "__main__":
    main()

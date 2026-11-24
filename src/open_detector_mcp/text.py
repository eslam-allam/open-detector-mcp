"""Text normalisation, paragraph splitting and sentence splitting.

Everything here is dependency-free so the MCP server can start, and answer
status questions, without importing torch/transformers.

The paragraph splitter is intentionally deterministic: the same input always
yields the same paragraph indices, which is what lets agents move between
``detect_ai_*``, ``get_rewrite_plan`` and ``read_paragraphs``.
"""

from __future__ import annotations

import re

# Characters that survive PDF/Word extraction but hurt tokenisation.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")

# Bullets / numbered / bracketed list items. Single newlines between these are
# real boundaries, not hard-wrapped prose.
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•‣–]|\(?\d{1,3}[.)\]]|\[\d{1,3}\]|[a-zA-Z][.)])\s+\S"
)

_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited)"
    r"\s*:?\s*$",
    re.I,
)

# `.` after these is not a sentence boundary. Lower-case; matching is case-insensitive.
_ABBREVIATIONS = (
    "e.g", "i.e", "et al", "etc", "vs", "cf", "ca", "approx", "resp",
    "fig", "figs", "eq", "eqs", "tab", "sec", "secs", "ref", "refs",
    "no", "nos", "vol", "vols", "pp", "p", "ch", "suppl", "al",
    "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "corp", "st",
    "phd", "dept", "univ", "assoc", "sci", "proc", "conf", "j", "rev",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "u.s", "u.k", "e.u",
)
_ABBREV_RE = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.",
    re.I,
)
_INITIAL_RE = re.compile(r"(?<![A-Za-z])([A-Z])\.(?=\s+[A-Z])")
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")
_SENTINEL = "\ue000"  # private-use char; restored to "." after splitting


def normalize_paragraph(text: str) -> str:
    """Collapse whitespace and strip control characters, preserving wording."""
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS_RE.sub(" ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _join_lines(lines: list[str]) -> str:
    """Join hard-wrapped lines, repairing PDF hyphenation (``trans-\\nformer``)."""
    out = lines[0]
    for line in lines[1:]:
        if (
            out.endswith("-")
            and len(out) > 2
            and out[-2].isalpha()
            and line[:1].islower()
        ):
            out = out[:-1] + line
        else:
            out = out + " " + line
    return out


def _block_paragraphs(block: str) -> list[str]:
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        return lines
    if all(_LIST_ITEM_RE.match(line) for line in lines):
        return lines
    # PDFs / plain-text exports often emit one sentence per line. Keep those as
    # separate paragraphs; otherwise the block is hard-wrapped prose and joining
    # it gives the model a proper paragraph to score.
    if len(lines) >= 3:
        short = sum(len(line) < 60 for line in lines) / len(lines)
        punctuated = sum(line.endswith((".", "!", "?", ":")) for line in lines) / len(lines)
        if short >= 0.7 and punctuated >= 0.6:
            return lines
    return [_join_lines(lines)]


def paragraphs_from_text(text: str) -> list[str]:
    """Split free text into paragraphs (blank lines and list items are boundaries)."""
    text = _FENCE_RE.sub("\n\n", text)  # code blocks are not prose
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\u00a0", " ")
    paragraphs: list[str] = []
    for block in _BLANK_LINE_RE.split(text):
        for para in _block_paragraphs(block):
            normalized = normalize_paragraph(para)
            if normalized:
                paragraphs.append(normalized)
    return paragraphs


def split_document(text: str, exclude_references: bool = True) -> list[str]:
    """Paragraphs for scoring; optionally drop everything from a References heading.

    Bibliographies are long, formulaic and rarely worth "humanizing", and scoring
    them drags the document average around. The cut is heading-based and only
    triggers on a line that is *only* a reference heading (optionally numbered).
    """
    paragraphs = paragraphs_from_text(text)
    if not exclude_references:
        return paragraphs
    for i, para in enumerate(paragraphs):
        if _REFERENCE_HEADING_RE.match(para):
            return paragraphs[:i]
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """Sentence splitter tuned for academic prose (abbreviations, initials, decimals)."""
    if not text:
        return []

    masked = _ABBREV_RE.sub(lambda m: m.group(1) + _SENTINEL, text)
    masked = _INITIAL_RE.sub(lambda m: m.group(1) + _SENTINEL, masked)
    masked = _DECIMAL_RE.sub(r"\1" + _SENTINEL + r"\2", masked)

    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?][\"'\u201d\u2019)\]]*(?=\s|$)", masked):
        piece = masked[start:match.end()].strip()
        if piece:
            sentences.append(piece.replace(_SENTINEL, "."))
        start = match.end()
    tail = masked[start:].strip()
    if tail:
        sentences.append(tail.replace(_SENTINEL, "."))
    return sentences


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019\-][A-Za-z0-9]+)*")


def words(text: str) -> list[str]:
    """Word-ish tokens, including hyphenated and numeric tokens."""
    return _WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def preview(text: str, max_words: int = 18) -> str:
    """A short, stable identifier for a paragraph, without dragging full text around."""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= max_words:
        return text.strip()
    # Cut on the original string so spacing and punctuation survive.
    return text[: matches[max_words - 1].end()].rstrip() + " …"

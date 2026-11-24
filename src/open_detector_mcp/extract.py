"""File readers for the formats papers actually arrive in.

Text, Markdown, LaTeX, DOCX and PDF are supported out of the box. DOCX/PDF
readers are imported lazily so status/help paths stay fast, and every reader
returns the same thing: a list of paragraphs plus a small metadata dict.
"""

from __future__ import annotations

import re
from pathlib import Path

from .text import split_document

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".rst", ".org"}
SUPPORTED = sorted(TEXT_SUFFIXES | {".tex", ".latex", ".docx", ".pdf"})


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _read_docx(path: Path) -> tuple[str, dict]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ValueError("Reading .docx needs python-docx (pip install python-docx).") from exc

    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    meta = {"tables_skipped": len(document.tables)}
    return "\n\n".join(paragraphs), meta


def _read_pdf(path: Path) -> tuple[str, dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ValueError("Reading .pdf needs pypdf (pip install pypdf).") from exc

    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"PDF is encrypted and cannot be read: {path}") from exc
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages), {"pages": len(reader.pages)}


def _clean_tex(text: str) -> str:
    """Best-effort LaTeX → prose. Keeps section/caption text, drops markup and math."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "\n\n", text, flags=re.S
    )
    text = re.sub(r"\\(?:bibliography|bibliographystyle)\{[^}]*\}", "", text)
    text = re.sub(
        r"\\(?:section|subsection|subsubsection|paragraph|chapter)\*?\{([^{}]*)\}",
        r"\n\n\1.\n\n",
        text,
    )
    text = re.sub(r"\\(?:caption|title|abstract)\*?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r" \1 ", text)
    text = re.sub(
        r"\\(?:cite[a-zA-Z]*|ref|eqref|autoref|label|url|footnote|thanks)\*?"
        r"(?:\[[^\]]*\])?\{[^{}]*\}",
        " ",
        text,
    )
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    for env in ("equation", "align", "eqnarray", "figure", "table", "tabular", "verbatim", "lstlisting"):
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", text, flags=re.S
        )
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "\n\n", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[~^_]", " ", text)
    return text


def extract_file(path: str, exclude_references: bool = True) -> tuple[list[str], dict]:
    """Read a paper file into (paragraphs, metadata). Raises ValueError on unsupported input."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    suffix = file_path.suffix.lower()
    extra: dict = {}

    if suffix == ".docx":
        text, extra = _read_docx(file_path)
        fmt = "docx"
    elif suffix == ".pdf":
        text, extra = _read_pdf(file_path)
        fmt = "pdf"
    elif suffix in (".tex", ".latex"):
        text = _clean_tex(_read_text(file_path))
        fmt = "latex"
    elif suffix in TEXT_SUFFIXES or suffix == "":
        text = _read_text(file_path)
        fmt = "text" if suffix != ".md" else "markdown"
    elif suffix == ".doc":
        raise ValueError(
            "Legacy .doc is not supported. Save the file as .docx or .txt and retry."
        )
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED)}"
        )

    paragraphs = split_document(text, exclude_references=exclude_references)
    meta = {
        "type": "file",
        "path": str(file_path.resolve()),
        "format": fmt,
        "paragraphs_total": len(paragraphs),
        **extra,
    }
    return paragraphs, meta

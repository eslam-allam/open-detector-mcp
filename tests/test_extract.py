from pathlib import Path

import pytest

from open_detector_mcp.extract import extract_file


def test_extract_plain_text(tmp_path: Path):
    path = tmp_path / "paper.txt"
    path.write_text(
        "A Title\n\nBody paragraph that is long enough to be scored by the model.\n\n"
        "References\n\n[1] A. Author. A paper. 2020.\n",
        encoding="utf-8",
    )
    paragraphs, meta = extract_file(str(path))
    assert meta["format"] == "text"
    assert meta["path"].endswith("paper.txt")
    assert paragraphs == ["A Title", "Body paragraph that is long enough to be scored by the model."]
    assert meta["paragraphs_total"] == 2


def test_extract_docx_including_references_cut(tmp_path: Path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "paper.docx"
    document = docx.Document()
    document.add_paragraph("A Title")
    document.add_paragraph("Body paragraph that is long enough to be scored by the model.")
    document.add_paragraph("References")
    document.add_paragraph("[1] A. Author. A paper. 2020.")
    document.save(str(path))

    paragraphs, meta = extract_file(str(path))
    assert meta["format"] == "docx"
    assert paragraphs == ["A Title", "Body paragraph that is long enough to be scored by the model."]


def test_extract_latex_strips_markup_and_bibliography(tmp_path: Path):
    path = tmp_path / "paper.tex"
    path.write_text(
        r"""\documentclass{article}
\begin{document}
\section{Introduction}
We study \cite{smith2020} retrieval with $x^2$ terms and \textbf{bold} claims.
% a comment that should disappear
\begin{thebibliography}{9}
\bibitem{smith2020} Smith. A paper. 2020.
\end{thebibliography}
\end{document}
""",
        encoding="utf-8",
    )
    paragraphs, meta = extract_file(str(path))
    joined = " ".join(paragraphs)
    assert meta["format"] == "latex"
    assert "Introduction" in joined
    assert "smith2020" not in joined
    assert "comment" not in joined
    assert "x^2" not in joined
    assert "bold" in joined


def test_unsupported_extension_is_a_clear_error(tmp_path: Path):
    path = tmp_path / "paper.xyz"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_file(str(path))


def test_missing_file_is_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_file(str(tmp_path / "missing.docx"))

# Open-Detector MCP

An MCP server that scores academic papers for AI-like style, paragraph by paragraph, and gives an agent a concrete loop for revising them. It wraps the open-source **[Open-Detector](https://github.com/Imalwayshere/Open-Detector)** BERT model ([`followsci/bert-ai-text-detector`](https://huggingface.co/followsci/bert-ai-text-detector), MIT) and is built to be installed as a `uv` tool and used by coding/research agents.

Why paragraph-level: the model was trained on academic paragraphs. In testing it is confident and stable there (100% AI on formulaic LLM abstracts, 0% on twenty pre-2023 arXiv abstracts), but noisy on isolated sentences. This server therefore makes the paragraph the unit of scoring, skips headings/titles and bibliographies, and treats sentence-level spans as an explicitly weak localisation hint.

## Install

```bash
# from GitHub
uv tool install "git+https://github.com/eslam-allam/open-detector-mcp"

# CPU-only (much smaller install: skips CUDA wheels)
uv tool install "git+https://github.com/eslam-allam/open-detector-mcp" --torch-backend cpu

# from a local clone
uv tool install .
```

This installs two commands:

| Command | Purpose |
| --- | --- |
| `open-detector-mcp` | the MCP stdio server |
| `open-detector` | CLI: `score`, `plan`, `compare`, `status`, `download` |

The first detection downloads ~440 MB of model weights from Hugging Face (one-time, cached in `~/.cache/huggingface`). Pre-fetch it with:

```bash
open-detector download
```

## Wire it into an MCP client

opencode (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "servers": {
      "open-detector": {
        "type": "local",
        "command": ["open-detector-mcp"]
      }
    }
  }
}
```

Claude Desktop:

```json
{
  "mcpServers": {
    "open-detector": {
      "command": "open-detector-mcp"
    }
  }
}
```

## Tools

| Tool | What it does |
| --- | --- |
| `get_model_status(warmup=false)` | Cache/download state; `warmup=true` loads and warms the model. Returns instantly otherwise. |
| `detect_ai_text(text, detail="paragraph", include_text=false, exclude_references=true)` | Overall `ai_pct`, verdict, paragraph bands, per-paragraph scores + style flags. |
| `detect_ai_file(path, ...)` | Same for `.txt`, `.md`, `.rst`, `.tex`, `.docx`, `.pdf` — the server reads the file, so large drafts stay out of the model's context. |
| `get_rewrite_plan(path=... or text=..., top_n=5, min_ai_pct=50, include_text=true)` | Ranked rewrite targets with full text, deterministic style issues and edit-level hints. |
| `read_paragraphs(path, indices)` | Exact paragraph text by stable index (max 50 per call). |
| `compare_texts(text_a, text_b)` | Before/after scores, per-paragraph deltas, improved/regressed counts. |
| `compare_files(path_a, path_b)` | Same for two files; aligns paragraphs by similarity when counts differ. |

There is also a `humanize_paper` prompt and an `open-detector://guide` resource with the same workflow.

## Agent workflow

```text
get_model_status(warmup=true)
detect_ai_file("paper.docx")
get_rewrite_plan(path="paper.docx", top_n=5)
  → rewrite paragraph 8, 3, 12 ...
compare_texts(original_paragraph, rewritten_paragraph)
  → repeat until every paragraph is < 50%, ideally < 20%
```

`get_rewrite_plan` is the entry point for humanizing: it returns the exact paragraph text plus issues such as uniform sentence length, stock transitions (`moreover`, `furthermore`, `ultimately`), LLM-favoured vocabulary (`delve`, `seamless`, `pivotal`, `holistic`), formulaic constructions (`not only … but also`, `it is important to note`), em-dash overuse and nominalisation density. Hints are edit-level, e.g. *"3 generic connectives in 86 words (34.9 per 1000): moreover ×1, furthermore ×1, ultimately ×1 — delete most and let argument order carry the logic."*

Light paraphrases do **not** move this model: in testing, a synonym-swap edit stayed at 100% AI while a structural rewrite dropped to 0%. Agents should change sentence order, specificity and rhythm, not just wording.

## Scoring semantics

- `ai_pct` is 0–100, higher = more AI-like. Verdicts: `LIKELY AI` ≥ 60, `MIXED` 40–59, `LIKELY HUMAN` < 40.
- Bands per paragraph: `high` ≥ 80, `medium` 50–79, `low` 20–49, `human` < 20.
- Paragraphs below 8 words and heading-like fragments (short, no terminal punctuation) are `Skipped` but keep their index.
- The overall score is the mean of scored paragraph scores. A document with a few 100% paragraphs and many 0% paragraphs reports a middling average; use `bands` and `paragraphs_flagged` for the real picture.
- Long paragraphs (> 480 tokens) are split on sentence boundaries and combined by token count; `chunked: true` exposes the per-chunk scores.
- `exclude_references=true` (default) cuts everything after a References/Bibliography/Works Cited heading. Reference lists are formulaic and would skew the average.
- `detail="windows"` adds 3-sentence candidate spans inside already-flagged paragraphs. Treat them as weak hints only; the paragraph score governs. Re-score the whole paragraph after editing.

## CLI

```bash
open-detector score paper.docx                 # table
open-detector score paper.docx --json          # full report
open-detector score - < paper.txt              # stdin
open-detector plan paper.docx --top 5          # rewrite targets + hints
open-detector compare before.txt after.txt     # before/after delta
open-detector status                           # model cache / device
open-detector download                         # pre-fetch weights
```

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPEN_DETECTOR_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `OPEN_DETECTOR_MODEL` | `followsci/bert-ai-text-detector` | Alternative 2-class Hugging Face text-classification model |
| `OPEN_DETECTOR_READY_TIMEOUT` | `900` | Seconds a tool waits for the startup model download |

## Caveats

- This is **stylometric** evidence, not proof of misconduct. The upstream project makes the same point: academic integrity should be judged on content authenticity, not writing style.
- English academic prose only; the model is `bert-base-uncased`.
- False positives happen, especially on formulaic human writing (reviews that enumerate "First… Next… Finally…"). Treat a high score as a prompt to edit, not an accusation.
- Paragraph scores are not calibrated probabilities; they are highly saturated (often 0% or 100%). Use `compare_texts` deltas for measuring an edit.
- DOCX body paragraphs are read; tables are skipped. PDF extraction depends on the PDF's text layer. Legacy `.doc` is not supported — export to `.docx` or `.txt`.
- No writing tools: the server reads and scores files, but the agent edits with its own tools. This keeps your paper's source-of-truth editing under your control.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run open-detector-mcp
```

## Credits & license

- Model and dataset: [Imalwayshere/Open-Detector](https://github.com/Imalwayshere/Open-Detector) and [`followsci/bert-ai-text-detector`](https://huggingface.co/followsci/bert-ai-text-detector), MIT.
- This server: MIT, see [LICENSE](LICENSE).

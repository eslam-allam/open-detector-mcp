"""Report shaping: paragraph scores, rewrite plans and before/after comparisons."""

from __future__ import annotations

from difflib import SequenceMatcher

from .detector import MIN_WORDS
from .styles import analyze_style, style_flags
from .text import preview, split_sentences, word_count, words

BAND_HIGH = 80.0
BAND_MEDIUM = 50.0
BAND_LOW = 20.0
FLAG_THRESHOLD = 50.0
CHANGE_THRESHOLD = 5.0

SPAN_NOTE = (
    "Window scores are a weak localisation signal, not a verdict: the model was "
    "trained on whole paragraphs and is noisy on 1–3 sentence fragments. Use them "
    "only to find candidate spans inside a paragraph the paragraph-level score "
    "already flags, and always re-score the whole paragraph after editing."
)


def verdict(pct: float | None) -> str:
    if pct is None:
        return "INSUFFICIENT TEXT"
    if pct >= 60:
        return "LIKELY AI"
    if pct < 40:
        return "LIKELY HUMAN"
    return "MIXED"


def _band(pct: float) -> str:
    if pct >= BAND_HIGH:
        return "high"
    if pct >= BAND_MEDIUM:
        return "medium"
    if pct >= BAND_LOW:
        return "low"
    return "human"


def _looks_like_heading(text: str) -> bool:
    """Titles, section headings and captions score badly out of context; skip them.

    A short paragraph with no terminal punctuation is much more likely to be a
    heading/caption than running prose. This keeps titles from dominating the
    document average.
    """
    stripped = text.strip()
    if len(stripped) > 160 or word_count(stripped) > 20:
        return False
    return not stripped.endswith((".", "!", "?", ":", ";", ","))


def _score_paragraph(index: int, text: str, detector, include_text: bool) -> dict:
    wc = word_count(text)
    if wc < MIN_WORDS:
        entry = {
            "index": index,
            "label": "Skipped",
            "reason": "too_short",
            "words": wc,
            "preview": preview(text),
        }
        if include_text:
            entry["text"] = text
        return entry
    if _looks_like_heading(text):
        entry = {
            "index": index,
            "label": "Skipped",
            "reason": "heading_or_fragment",
            "words": wc,
            "preview": preview(text),
        }
        if include_text:
            entry["text"] = text
        return entry

    prob, chunks, chunked = detector.score_paragraph(text)
    pct = round(prob * 100, 1)
    entry = {
        "index": index,
        "ai_pct": pct,
        "label": "AI" if pct >= FLAG_THRESHOLD else "Human",
        "words": wc,
        "tokens": sum(c["tokens"] for c in chunks),
        "preview": preview(text),
        "style_flags": style_flags(text),
    }
    if chunked:
        entry["chunked"] = True
        entry["chunks"] = chunks
    if include_text:
        entry["text"] = text
    return entry


def _candidate_spans(index: int, text: str, detector) -> list[dict]:
    """Non-overlapping 3-sentence windows, only inside already-flagged paragraphs."""
    sentences = split_sentences(text)
    if len(sentences) < 4:
        return []

    groups: list[tuple[int, int, list[str]]] = []
    for start in range(0, len(sentences), 3):
        groups.append((start, min(start + 3, len(sentences)), sentences[start:start + 3]))
    if len(groups) > 1 and word_count(" ".join(groups[-1][2])) < 12:
        _, last_end, last_sents = groups.pop()
        prev_start, _prev_end, prev_sents = groups[-1]
        groups[-1] = (prev_start, last_end, prev_sents + last_sents)

    texts = [" ".join(group[2]) for group in groups]
    probs = detector.score_texts(texts)
    spans = []
    for (start, end, _), prob, window in zip(groups, probs, texts):
        pct = round(prob * 100, 1)
        if pct < 60:
            continue
        spans.append({
            "paragraph_index": index,
            "sentence_start": start + 1,
            "sentence_end": end,
            "ai_pct": pct,
            "preview": preview(window, 24),
        })
    return spans


def classify(
    paragraphs: list[str],
    detector,
    detail: str = "paragraph",
    include_text: bool = False,
) -> dict:
    """Score every paragraph (and optionally candidate spans) and summarise."""
    if detail not in ("paragraph", "windows"):
        raise ValueError("detail must be 'paragraph' or 'windows'")

    entries = [_score_paragraph(i, text, detector, include_text) for i, text in enumerate(paragraphs)]
    scored = [entry for entry in entries if entry["label"] != "Skipped"]

    overall_pct = (
        round(sum(entry["ai_pct"] for entry in scored) / len(scored), 1) if scored else None
    )
    bands = {"high": 0, "medium": 0, "low": 0, "human": 0}
    flag_totals: dict[str, int] = {}
    for entry in scored:
        bands[_band(entry["ai_pct"])] += 1
        for flag in entry["style_flags"]:
            flag_totals[flag] = flag_totals.get(flag, 0) + 1

    overall = {
        "ai_pct": overall_pct,
        "verdict": verdict(overall_pct),
        "paragraphs_scored": len(scored),
        "paragraphs_flagged": sum(1 for entry in scored if entry["label"] == "AI"),
        "paragraphs_total": len(entries),
        "max_paragraph_ai_pct": max((entry["ai_pct"] for entry in scored), default=None),
        "words": sum(entry["words"] for entry in entries),
    }

    report = {
        "overall": overall,
        "bands": bands,
        "style_flag_totals": flag_totals,
        "paragraphs": entries,
    }

    if detail == "windows":
        candidates: list[dict] = []
        for entry in scored:
            if entry["ai_pct"] < FLAG_THRESHOLD:
                continue
            candidates.extend(_candidate_spans(entry["index"], paragraphs[entry["index"]], detector))
        report["span_analysis"] = {"note": SPAN_NOTE, "candidates": candidates}

    return report


def rewrite_plan(
    paragraphs: list[str],
    detector,
    top_n: int = 5,
    min_ai_pct: float = FLAG_THRESHOLD,
    include_text: bool = True,
) -> dict:
    """Rank paragraphs by AI score and attach full style issues + exact text."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if not 0 <= min_ai_pct <= 100:
        raise ValueError("min_ai_pct must be between 0 and 100")

    report = classify(paragraphs, detector, detail="paragraph", include_text=False)
    candidates = [
        entry for entry in report["paragraphs"]
        if entry["label"] != "Skipped" and entry["ai_pct"] >= min_ai_pct
    ]
    candidates.sort(key=lambda entry: entry["ai_pct"], reverse=True)

    targets = []
    for priority, entry in enumerate(candidates[:top_n], start=1):
        text = paragraphs[entry["index"]]
        style = analyze_style(text)
        target = {
            "priority": priority,
            "index": entry["index"],
            "ai_pct": entry["ai_pct"],
            "words": entry["words"],
            "preview": entry["preview"],
            "style_issues": style["issues"],
            "style_metrics": style["metrics"],
        }
        if include_text:
            target["text"] = text
        targets.append(target)

    if targets:
        next_steps = [
            (
                "Rewrite the targets in priority order. Change structure and "
                "specificity, not just synonyms: the detector ignores light paraphrases."
            ),
            (
                "Keep claims, numbers, citations and section structure intact; do not "
                "invent evidence or references."
            ),
            (
                "Re-score with compare_texts or compare_files. Aim for every paragraph "
                "below 50% AI, ideally below 20%."
            ),
        ]
    else:
        next_steps = [
            (
                f"No scored paragraph is at or above {min_ai_pct:.0f}% AI. If you still "
                "want to smooth the style, lower min_ai_pct or call detect_ai_text with "
                "include_text=true and inspect the highest-scoring paragraphs."
            ),
        ]

    return {
        "overall": report["overall"],
        "bands": report["bands"],
        "style_flag_totals": report["style_flag_totals"],
        "target_count": len(candidates),
        "returned": len(targets),
        "targets": targets,
        "next_steps": next_steps,
        "note": (
            "Paragraph scores are the reliable unit. Style issues are deterministic "
            "heuristics, not model output."
        ),
    }


def _similarity(a: str, b: str) -> float:
    aw = [w.lower() for w in words(a)]
    bw = [w.lower() for w in words(b)]
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    return SequenceMatcher(None, aw, bw, autojunk=False).ratio()


def _align_paragraphs(
    before_paragraphs: list[str], after_paragraphs: list[str]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Align paragraphs when counts differ (e.g. one paragraph was split or removed).

    Index pairing is used when counts match, because that is the in-place-rewrite
    workflow and survives complete rewording. With unequal counts, a small
    sequence alignment anchors headings and unchanged paragraphs, then reports the
    rest as added/removed instead of silently pairing unrelated text.
    """
    n, m = len(before_paragraphs), len(after_paragraphs)
    gap = 0.35
    sim = [[_similarity(before_paragraphs[i], after_paragraphs[j]) for j in range(m)]
           for i in range(n)]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i - 1][j] - gap,
                dp[i][j - 1] - gap,
                dp[i - 1][j - 1] + sim[i - 1][j - 1] - gap,
            )

    pairs: list[tuple[int, int, float]] = []
    before_only: list[int] = []
    after_only: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and abs(dp[i][j] - (dp[i - 1][j - 1] + sim[i - 1][j - 1] - gap)) < 1e-9
        ):
            if sim[i - 1][j - 1] >= 0.35:
                pairs.append((i - 1, j - 1, sim[i - 1][j - 1]))
            else:
                before_only.append(i - 1)
                after_only.append(j - 1)
            i -= 1
            j -= 1
        elif i > 0 and abs(dp[i][j] - (dp[i - 1][j] - gap)) < 1e-9:
            before_only.append(i - 1)
            i -= 1
        else:
            after_only.append(j - 1)
            j -= 1

    pairs.reverse()
    before_only.reverse()
    after_only.reverse()
    return pairs, before_only, after_only


def compare(
    before_paragraphs: list[str],
    after_paragraphs: list[str],
    detector,
) -> dict:
    """Score two versions and report the delta, paired by paragraph."""
    before = classify(before_paragraphs, detector)
    after = classify(after_paragraphs, detector)

    before_pct = before["overall"]["ai_pct"]
    after_pct = after["overall"]["ai_pct"]
    delta = round(after_pct - before_pct, 1) if before_pct is not None and after_pct is not None else None
    if delta is None:
        result = "INSUFFICIENT TEXT"
    elif delta <= -CHANGE_THRESHOLD:
        result = "IMPROVED"
    elif delta >= CHANGE_THRESHOLD:
        result = "REGRESSED"
    else:
        result = "UNCHANGED"

    before_entries = {entry["index"]: entry for entry in before["paragraphs"]}
    after_entries = {entry["index"]: entry for entry in after["paragraphs"]}

    if len(before_paragraphs) == len(after_paragraphs):
        pairs = [(i, i, 1.0) for i in range(len(before_paragraphs))]
        before_only: list[int] = []
        after_only: list[int] = []
    else:
        pairs, before_only, after_only = _align_paragraphs(before_paragraphs, after_paragraphs)

    deltas = []
    improved = regressed = unchanged = 0
    for before_index, after_index, similarity in pairs:
        b = before_entries[before_index]
        a = after_entries[after_index]
        if b["label"] == "Skipped" or a["label"] == "Skipped":
            continue
        pair_delta = round(a["ai_pct"] - b["ai_pct"], 1)
        if pair_delta <= -CHANGE_THRESHOLD:
            improved += 1
        elif pair_delta >= CHANGE_THRESHOLD:
            regressed += 1
        else:
            unchanged += 1
        deltas.append({
            "before_index": before_index,
            "after_index": after_index,
            "similarity": round(similarity, 2),
            "before_ai_pct": b["ai_pct"],
            "after_ai_pct": a["ai_pct"],
            "delta": pair_delta,
            "preview": b["preview"],
        })

    added = [
        {"index": index, "ai_pct": after_entries[index]["ai_pct"],
         "preview": after_entries[index]["preview"]}
        for index in after_only
        if after_entries[index]["label"] != "Skipped"
    ]
    removed = [
        {"index": index, "ai_pct": before_entries[index]["ai_pct"],
         "preview": before_entries[index]["preview"]}
        for index in before_only
        if before_entries[index]["label"] != "Skipped"
    ]

    warnings = []
    if len(before_paragraphs) != len(after_paragraphs):
        warnings.append(
            f"Paragraph counts differ (before {len(before_paragraphs)}, after "
            f"{len(after_paragraphs)}). Paragraphs were aligned by text similarity; "
            "check added/removed entries, and compare paragraph-by-paragraph with "
            "detect_ai_text if the pairing matters."
        )
    if not deltas:
        warnings.append("No paragraph pair could be scored in both versions.")
    if added or removed:
        warnings.append(
            "Completely rewritten paragraphs have little word overlap and are "
            "reported as removed/added rather than paired. Use "
            "compare_texts(original_paragraph, rewritten_paragraph) for a direct "
            "per-paragraph delta."
        )

    return {
        "before": {"ai_pct": before_pct, "verdict": before["overall"]["verdict"]},
        "after": {"ai_pct": after_pct, "verdict": after["overall"]["verdict"]},
        "delta_ai_pct": delta,
        "result": result,
        "paragraphs": {
            "compared": len(deltas),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "added": len(added),
            "removed": len(removed),
        },
        "paragraph_deltas": deltas,
        "added_paragraphs": added,
        "removed_paragraphs": removed,
        "warnings": warnings,
        "note": (
            "delta_ai_pct is after minus before, so negative is better. "
            "Per-paragraph deltas below -5 are improvements."
        ),
    }

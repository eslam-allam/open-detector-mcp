"""Deterministic style diagnostics that explain *why* a paragraph reads AI-like.

The BERT backdrop gives a probability; these checks give an agent something
actionable to edit. They are heuristics, not detector output, and are labelled
as such in every response.
"""

from __future__ import annotations

import re
import statistics

from .text import split_sentences, word_count

_TRANSITIONS = {
    "moreover": r"\bmoreover\b",
    "furthermore": r"\bfurthermore\b",
    "additionally": r"\badditionally\b",
    "consequently": r"\bconsequently\b",
    "therefore": r"\btherefore\b",
    "thus": r"\bthus\b",
    "hence": r"\bhence\b",
    "in addition": r"\bin addition\b",
    "in contrast": r"\bin contrast\b",
    "by contrast": r"\bby contrast\b",
    "on the other hand": r"\bon the other hand\b",
    "as a result": r"\bas a result\b",
    "for instance": r"\bfor instance\b",
    "for example": r"\bfor example\b",
    "specifically": r"\bspecifically\b",
    "notably": r"\bnotably\b",
    "importantly": r"\bimportantly\b",
    "overall": r"\boverall\b",
    "in conclusion": r"\bin conclusion\b",
    "in summary": r"\bin summary\b",
    "to summarize": r"\bto summarize\b",
    "ultimately": r"\bultimately\b",
    "first and foremost": r"\bfirst and foremost\b",
    "firstly": r"\bfirstly\b",
    "secondly": r"\bsecondly\b",
    "thirdly": r"\bthirdly\b",
    "finally": r"\bfinally\b",
    "meanwhile": r"\bmeanwhile\b",
    "subsequently": r"\bsubsequently\b",
    "accordingly": r"\baccordingly\b",
    "nevertheless": r"\bnevertheless\b",
    "nonetheless": r"\bnonetheless\b",
    "indeed": r"\bindeed\b",
    "in particular": r"\bin particular\b",
    "particularly": r"\bparticularly\b",
    "significantly": r"\bsignificantly\b",
    "crucially": r"\bcrucially\b",
    "conversely": r"\bconversely\b",
    "similarly": r"\bsimilarly\b",
    "likewise": r"\blikewise\b",
    "in essence": r"\bin essence\b",
    "to this end": r"\bto this end\b",
    "with this in mind": r"\bwith this in mind\b",
}
_TRANSITION_RES = {name: re.compile(rx, re.I) for name, rx in _TRANSITIONS.items()}

_BUZZWORDS = (
    "delve", "crucial", "pivotal", "robust", "seamless", "leverage", "underscore",
    "holistic", "multifaceted", "realm", "landscape", "tapestry", "testament",
    "navigate", "foster", "showcase", "intricate", "meticulous", "comprehensive",
    "paradigm", "synergy", "utilize", "harness", "illuminate", "elucidate",
    "paramount", "invaluable", "profound", "cutting-edge", "transformative",
    "empower", "streamline", "cutting edge",
)
_STRONG_BUZZWORDS = {
    "delve", "tapestry", "testament", "seamless", "multifaceted", "holistic",
    "leverage", "realm", "synergy", "elucidate", "paramount",
}
_BUZZWORD_RES = {w: re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in _BUZZWORDS}

_FORMULAIC = {
    "not only … but also": r"\bnot only\b[^.!?]{0,120}?\bbut also\b",
    "it is important/worth noting": (
        r"\bit is (?:important|worth|essential|critical|interesting) to "
        r"(?:note|mention|emphasize|highlight|recognize)\b"
    ),
    "plays a crucial role": (
        r"\bplays? an? (?:crucial|pivotal|vital|key|important|significant) role\b"
    ),
    "in today's landscape": (
        r"\bin today'?s\b[^.!?]{0,60}\b(?:landscape|world|era|environment)\b"
    ),
    "a wide range of": r"\ba (?:wide|broad|diverse) range of\b",
    "increasingly important": r"\b(?:increasingly|ever[- ]more) (?:important|critical|relevant|prevalent)\b",
    "serves as a testament": r"\b(?:serves?|stands?) as a testament\b",
}
_FORMULAIC_RES = {name: re.compile(rx, re.I) for name, rx in _FORMULAIC.items()}

_NOMINALIZATION_RE = re.compile(
    r"\b[a-z]{5,}(?:tion|sion|ment|ness|ity|ance|ence|ization|isation)s?\b", re.I
)
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|get|gets|got)\s+(?:\w+ly\s+)?"
    r"(?:[a-z]+(?:ed|en)|shown|given|taken|made|done|found|known|seen|held|built|drawn)\b",
    re.I,
)
_EM_DASH_RE = re.compile(r"\u2014|\s--\s")

_GENERIC_OPENERS = (
    "this paper", "this study", "this work", "this research", "overall",
    "in conclusion", "in summary", "ultimately", "taken together",
)

_MIN_WORDS_FOR_METRICS = 35


def _phrase_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    hits = {}
    for name, regex in patterns.items():
        count = len(regex.findall(text))
        if count:
            hits[name] = count
    return hits


def analyze_style(text: str) -> dict:
    """Return metrics plus a list of concrete, edit-level issues.

    Each issue has ``tag`` (stable machine name), ``detail`` (what was found and
    where) and ``hint`` (what to change). Empty issues means no strong style
    smell; it does not mean the model will call the paragraph human.
    """
    sentences = [s for s in split_sentences(text) if s.strip()]
    lengths = [word_count(s) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    wc = word_count(text)
    n = len(lengths)

    mean_len = statistics.fmean(lengths) if lengths else 0.0
    stdev = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    cv = (stdev / mean_len) if mean_len else 0.0
    long_sentences = [i + 1 for i, length in enumerate(lengths) if length > 35]

    transition_hits = _phrase_hits(text, _TRANSITION_RES)
    transition_total = sum(transition_hits.values())
    transition_rate = (transition_total / wc * 1000) if wc else 0.0

    buzzword_hits = _phrase_hits(text, _BUZZWORD_RES)
    buzzword_total = sum(buzzword_hits.values())
    strong_buzzwords = sorted(set(buzzword_hits) & _STRONG_BUZZWORDS)

    formulaic_hits = _phrase_hits(text, _FORMULAIC_RES)
    em_dashes = len(_EM_DASH_RE.findall(text))
    nominalizations = len(_NOMINALIZATION_RE.findall(text))
    nominalization_rate = (nominalizations / wc * 100) if wc else 0.0
    passive = len(_PASSIVE_RE.findall(text))
    passive_rate = (passive / n) if n else 0.0

    openers: dict[str, int] = {}
    for sentence in sentences:
        tokens = re.findall(r"[A-Za-z']+", sentence.lower())
        if len(tokens) >= 2:
            key = " ".join(tokens[:2])
            openers[key] = openers.get(key, 0) + 1
    repeated_openers = {k: v for k, v in openers.items() if v >= 3}

    generic_openers = 0
    generic_examples = []
    for sentence in sentences:
        low = sentence.lower().lstrip("\"'(")
        for prefix in _GENERIC_OPENERS:
            if low.startswith(prefix):
                generic_openers += 1
                generic_examples.append(prefix)
                break

    metrics: dict = {
        "words": wc,
        "sentences": n,
        "mean_sentence_words": round(mean_len, 1),
        "sentence_length_cv": round(cv, 2),
        "long_sentences": len(long_sentences),
        "transitions_per_1000_words": round(transition_rate, 1),
        "em_dashes": em_dashes,
        "nominalizations_per_100_words": round(nominalization_rate, 1),
        "passive_constructions": passive,
    }
    if transition_hits:
        metrics["transition_hits"] = transition_hits
    if buzzword_hits:
        metrics["buzzword_hits"] = buzzword_hits
    if repeated_openers:
        metrics["repeated_openers"] = repeated_openers
    if formulaic_hits:
        metrics["formulaic_hits"] = formulaic_hits

    issues: list[dict] = []
    long_enough = wc >= _MIN_WORDS_FOR_METRICS

    if long_enough and n >= 4 and cv < 0.22:
        issues.append({
            "tag": "uniform_sentence_length",
            "detail": (
                f"{n} sentences, mean {mean_len:.1f} words, length CV {cv:.2f} "
                f"(typical human academic prose is above ~0.30)."
            ),
            "hint": (
                "Vary rhythm: merge two short sentences somewhere, split the "
                "longest one, and let one sentence run past 30 words if it reads "
                "naturally."
            ),
        })

    if long_enough and len(long_sentences) >= 2 and len(long_sentences) / n >= 0.3:
        issues.append({
            "tag": "long_sentences",
            "detail": f"{len(long_sentences)} of {n} sentences exceed 35 words (sentences {long_sentences}).",
            "hint": "Split the longest sentences at a real clause boundary; do not just insert commas.",
        })

    if long_enough and transition_rate >= 15:
        top = ", ".join(f"{k} ×{v}" for k, v in sorted(transition_hits.items(), key=lambda kv: -kv[1])[:5])
        issues.append({
            "tag": "heavy_transitions",
            "detail": f"{transition_total} generic connectives in {wc} words ({transition_rate:.1f} per 1000): {top}.",
            "hint": (
                "Delete most of these connectives and let the order of the "
                "argument carry the logic; keep at most one per few sentences."
            ),
        })

    if buzzword_total >= 3 or strong_buzzwords:
        top = ", ".join(f"{k} ×{v}" for k, v in sorted(buzzword_hits.items(), key=lambda kv: -kv[1])[:6])
        issues.append({
            "tag": "buzzwords",
            "detail": f"LLM-favoured wording: {top}.",
            "hint": (
                "Replace with plain, specific wording. Prefer a concrete verb "
                "over 'leverage/foster/showcase', and cut 'crucial/pivotal' "
                "unless the claim genuinely depends on it."
            ),
        })

    if formulaic_hits:
        top = ", ".join(f"{k} ×{v}" for k, v in sorted(formulaic_hits.items(), key=lambda kv: -kv[1])[:4])
        issues.append({
            "tag": "formulaic_phrases",
            "detail": f"Stock constructions: {top}.",
            "hint": "Rewrite the sentence directly; these templates are among the strongest stylistic tells.",
        })

    if repeated_openers:
        top = ", ".join(f"'{k}' ×{v}" for k, v in sorted(repeated_openers.items(), key=lambda kv: -kv[1])[:4])
        issues.append({
            "tag": "repeated_openers",
            "detail": f"Repeated sentence openings: {top}.",
            "hint": "Vary openings or combine adjacent sentences that share a subject.",
        })

    if generic_openers >= 2:
        issues.append({
            "tag": "generic_openers",
            "detail": f"{generic_openers} sentences start with generic framing ({', '.join(sorted(set(generic_examples)))}).",
            "hint": "Start with the concrete subject or finding instead of 'This paper/Overall/Ultimately'.",
        })

    if em_dashes >= 3:
        issues.append({
            "tag": "em_dashes",
            "detail": f"{em_dashes} em dashes.",
            "hint": "Keep at most one; use commas, parentheses or a second sentence.",
        })

    if long_enough and nominalization_rate >= 10:
        issues.append({
            "tag": "nominalization",
            "detail": f"{nominalizations} nominalisations in {wc} words ({nominalization_rate:.1f} per 100).",
            "hint": "Turn noun stacks back into verbs where accurate ('the implementation of' → 'we implemented').",
        })

    if long_enough and passive >= 3 and passive_rate >= 0.6:
        issues.append({
            "tag": "passive_voice",
            "detail": f"~{passive} passive constructions across {n} sentences.",
            "hint": "Use active voice where the actor is known; keep passive only when the actor is irrelevant.",
        })

    return {"metrics": metrics, "issues": issues}


def style_flags(text: str) -> list[str]:
    """Just the issue tags, for compact per-paragraph output."""
    return [issue["tag"] for issue in analyze_style(text)["issues"]]

"""Open-Detector model loading, chunking and scoring.

The model is a BERT-base binary classifier trained on academic paragraphs
(1 = AI-written, 0 = human-written) with a 512-token limit. Paragraphs are the
reliable unit; long paragraphs are split on sentence boundaries and their
chunk probabilities are combined by token count.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

from .text import split_sentences

DEFAULT_MODEL_REPO = "followsci/bert-ai-text-detector"
MODEL_REPO = os.environ.get("OPEN_DETECTOR_MODEL", DEFAULT_MODEL_REPO).strip() or DEFAULT_MODEL_REPO
MODEL_LABEL = "open-detector (BERT-base academic AI detector)"
APPROX_MB = 440 if MODEL_REPO == DEFAULT_MODEL_REPO else None

MAX_TOKENS = 480  # 512 minus [CLS]/[SEP] and a safety margin
BATCH_SIZE = 8
MIN_WORDS = 8

_CACHE_PROBES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class Chunk:
    text: str
    tokens: int


# ---------------------------------------------------------------------------
# Hugging Face cache helpers
# ---------------------------------------------------------------------------

def is_model_cached(repo: str | None = None) -> bool:
    """True when the model weights are already in the local HF cache."""
    repo = repo or MODEL_REPO
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    for probe in _CACHE_PROBES:
        try:
            if isinstance(try_to_load_from_cache(repo, probe), str):
                return True
        except Exception:
            continue
    return False


def prefetch_model(repo: str | None = None) -> bool:
    """Download weights/config/tokenizer without loading them onto a device.

    Never raises: a network failure is reported on stderr and the first detect
    call will retry the download the normal way.
    """
    repo = repo or MODEL_REPO
    if is_model_cached(repo):
        return True
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return False
    print(
        f"open-detector: downloading {repo} from Hugging Face "
        "(one-time, cached for later runs)...",
        file=sys.stderr,
    )
    try:
        snapshot_download(
            repo,
            allow_patterns=[
                "*.json", "*.txt", "*.safetensors", "*.model", "*.spm",
                "tokenizer*", "vocab*", "merges*", "special_tokens*",
            ],
        )
    except Exception as exc:
        print(
            f"open-detector: background model download failed ({exc}); "
            "the first detect call will download it instead.",
            file=sys.stderr,
        )
        return False
    return is_model_cached(repo)


# ---------------------------------------------------------------------------
# Device + dependency preload
# ---------------------------------------------------------------------------

def resolve_device(pref: str | None = None):
    """Resolve 'auto' | 'cpu' | 'cuda' | 'mps' to a torch.device."""
    import torch

    pref = (pref or os.environ.get("OPEN_DETECTOR_DEVICE") or "auto").strip().lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("open-detector: no CUDA device; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    if pref == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        print("open-detector: no MPS device; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preload_dependencies() -> None:
    """Import heavy libraries in one dedicated thread before serving requests.

    First-time imports of C-extension stacks from a request worker thread can
    deadlock; warming sys.modules in the startup preload thread avoids that.
    """
    import importlib

    for module in ("torch", "transformers", "huggingface_hub"):
        try:
            importlib.import_module(module)
        except Exception:
            pass
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Split a paragraph into <= max_tokens chunks on sentence boundaries."""
    if not text.strip():
        return []
    sentences = split_sentences(text) or [text]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            joined = " ".join(current).strip()
            if joined:
                chunks.append(Chunk(joined, max(current_tokens, 1)))
            current = []
            current_tokens = 0

    for sentence in sentences:
        tokens = count_tokens(tokenizer, sentence)
        if tokens > max_tokens:
            flush()
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            for start in range(0, len(ids), max_tokens):
                piece = ids[start:start + max_tokens]
                decoded = tokenizer.decode(
                    piece, skip_special_tokens=True, clean_up_tokenization_spaces=True
                ).strip()
                if decoded:
                    chunks.append(Chunk(decoded, len(piece)))
            continue
        if current and current_tokens + tokens + 1 > max_tokens:
            flush()
        current.append(sentence)
        current_tokens += tokens + 1
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class OpenDetectorBackend:
    """The followsci/bert-ai-text-detector binary classifier."""

    def __init__(self, repo: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.repo = repo or MODEL_REPO
        self.device = resolve_device(device)
        self._lock = threading.Lock()

        if not is_model_cached(self.repo):
            print(
                f"open-detector: {self.repo} is not cached; downloading it now "
                "(this can take a few minutes)...",
                file=sys.stderr,
            )
        print(f"open-detector: loading {self.repo} on {self.device}...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.repo)
        self.model.to(self.device).eval()

        config = self.model.config
        if int(getattr(config, "num_labels", 0)) != 2:
            raise ValueError(
                f"{self.repo} is not a binary classifier "
                f"(num_labels={getattr(config, 'num_labels', '?')}); open-detector "
                "expects a 2-class human/AI model."
            )
        self.ai_index = self._find_ai_index(config)
        print(
            f"open-detector: model ready ({self.repo}, labels: "
            f"0={config.id2label.get(0, '?')}, 1={config.id2label.get(1, '?')})",
            file=sys.stderr,
        )

    @staticmethod
    def _find_ai_index(config) -> int:
        """Find the AI class, falling back to the model card's 1 = AI mapping."""
        aliases = {
            "ai", "ai-generated", "ai_generated", "machine", "machine-generated",
            "generated", "fake", "chatgpt", "gpt",
        }
        id2label = getattr(config, "id2label", None) or {}
        for index, label in id2label.items():
            if str(label).strip().lower() in aliases:
                return int(index)
        return 1

    def score_texts(self, texts: list[str]) -> list[float]:
        """P(AI) for each text, batched and serialised per loaded model."""
        if not texts:
            return []
        torch = self._torch
        probs: list[float] = []
        with self._lock:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start:start + BATCH_SIZE]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    logits = self.model(**encoded).logits
                    batch_probs = torch.softmax(logits, dim=-1)[:, self.ai_index]
                probs.extend(float(p) for p in batch_probs.tolist())
        return probs

    def score_paragraph(self, text: str) -> tuple[float, list[dict], bool]:
        """Score a paragraph, chunking it if needed.

        Returns (ai_prob, chunk_reports, chunked). Chunk probabilities are
        weighted by token count so a short trailing chunk cannot dominate.
        """
        chunks = chunk_text(text, self.tokenizer)
        if not chunks:
            return 0.0, [], False
        probs = self.score_texts([c.text for c in chunks])
        total_tokens = sum(c.tokens for c in chunks)
        combined = (
            sum(p * c.tokens for p, c in zip(probs, chunks)) / total_tokens
            if total_tokens
            else 0.0
        )
        reports = [
            {
                "ai_pct": round(p * 100, 1),
                "tokens": c.tokens,
                "preview": c.text[:160] + (" …" if len(c.text) > 160 else ""),
            }
            for p, c in zip(probs, chunks)
        ]
        return combined, reports, len(chunks) > 1

    def warmup(self) -> None:
        """Load is lazy; run one tiny inference so the first real call is fast."""
        self.score_texts(["This sentence warms up the detector before real use."])


# ---------------------------------------------------------------------------
# Cached singleton
# ---------------------------------------------------------------------------

_DETECTORS: dict[tuple[str, str], OpenDetectorBackend] = {}
_DETECTOR_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_GUARD = threading.Lock()


def _device_key(device: str | None) -> str:
    value = (device or os.environ.get("OPEN_DETECTOR_DEVICE") or "auto").strip().lower()
    return "cuda" if value == "gpu" else value


def get_detector(device: str | None = None, repo: str | None = None) -> OpenDetectorBackend:
    """Return the cached backend for this (repo, device), loading it if needed."""
    key = (repo or MODEL_REPO, _device_key(device))
    cached = _DETECTORS.get(key)
    if cached is not None:
        return cached
    with _GUARD:
        lock = _DETECTOR_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key not in _DETECTORS:
            _DETECTORS[key] = OpenDetectorBackend(repo=key[0], device=key[1])
    return _DETECTORS[key]

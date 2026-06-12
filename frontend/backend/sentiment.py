"""News headline sentiment classification — VADER + FinBERT side-by-side.

VADER: rule-based, instant, with an oil-finance lexicon overlay so it
       understands "draw" / "build" / "outage" / "cut" / etc.
FinBERT (ProsusAI/finbert): transformer fine-tuned on financial news; deeper
       semantic understanding of phrases like "guidance lowered" or
       "inventory build". Lazy-loaded in a background thread at startup
       (~30s) so it never blocks the FastAPI request handler. Falls back to
       VADER if the model can't be loaded.

Both engines return ``{compound, label}`` with compound in [-1, +1] and
label in {bullish, bearish, neutral}. Per-item, news.py calls VADER
synchronously (microseconds) and FinBERT asynchronously via thread executor."""
from __future__ import annotations

import threading
from typing import Dict, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _AVAILABLE = True
    _analyzer = SentimentIntensityAnalyzer()

    # Oil-market lexicon overlay — values are valence scores in VADER's
    # range (-4 to +4). Positive = bullish for oil prices.
    _OIL_LEXICON: Dict[str, float] = {
        # ----- bullish for oil (supply tightens / demand rises) -----
        "draw":         2.5,
        "draws":        2.5,
        "drawdown":     2.5,
        "drawdowns":    2.5,
        "cut":          1.5,
        "cuts":         1.5,
        "outage":       3.0,
        "outages":      3.0,
        "halt":         2.5,
        "halts":        2.5,
        "shortage":     3.0,
        "shortages":    3.0,
        "tighten":      2.5,
        "tightens":     2.5,
        "tightening":   2.5,
        "sanction":     2.0,
        "sanctions":    2.0,
        "disruption":   3.0,
        "disruptions":  3.0,
        "strike":       2.0,
        "stoppage":     3.0,
        "attack":       3.0,
        "war":          2.5,
        "embargo":      3.0,
        "blockade":     3.0,
        "rally":        2.0,
        "surge":        2.5,
        "spike":        2.0,
        # ----- bearish for oil (supply rises / demand falls) -----
        "build":       -2.5,
        "builds":      -2.5,
        "buildup":     -2.5,
        "glut":        -3.0,
        "oversupply":  -3.0,
        "surplus":     -2.0,
        "weak":        -1.5,
        "slowdown":    -2.5,
        "slump":       -2.5,
        "plunge":      -2.5,
        "crash":       -3.0,
        "recession":   -2.5,
        "ease":        -1.0,
        "eases":       -1.0,
        "easing":      -1.0,
        "ramp":        -1.5,
        "increase":    -1.0,
        "boost":       -1.5,
        "fall":        -1.5,
        "falls":       -1.5,
        "drop":        -1.5,
        "drops":       -1.5,
        "decline":     -1.5,
        "declines":    -1.5,
        "loss":        -1.5,
        "losses":      -1.5,
    }
    _analyzer.lexicon.update(_OIL_LEXICON)
except Exception:
    _AVAILABLE = False
    _analyzer = None

# bounded LRU-ish cache so we don't re-score identical headlines every refresh
_cache: Dict[str, Dict] = {}
_CACHE_MAX = 5000


def classify(text: str) -> Dict[str, object]:
    """Score a headline. Returns ``{'compound': float, 'label': str}``.

    ``compound`` is VADER's normalized score in [-1, +1]:
      ≥ +0.15 → 'bullish' (supply concern / demand strength)
      ≤ -0.15 → 'bearish' (supply build / demand weakness)
      otherwise → 'neutral'"""
    if not text:
        return {"compound": 0.0, "label": "neutral"}
    if not _AVAILABLE:
        return {"compound": 0.0, "label": "neutral"}

    key = text.lower()[:200]
    cached = _cache.get(key)
    if cached is not None:
        return cached

    scores = _analyzer.polarity_scores(text)
    c = float(scores["compound"])
    if c >= 0.15:
        label = "bullish"
    elif c <= -0.15:
        label = "bearish"
    else:
        label = "neutral"

    result = {"compound": round(c, 3), "label": label}
    if len(_cache) < _CACHE_MAX:
        _cache[key] = result
    return result


# ──────────────────────────────────────────────────────────────────────
# FinBERT — ProsusAI/finbert, lazy-loaded in a background thread
# ──────────────────────────────────────────────────────────────────────
class _FinBertEngine:
    """Holds the FinBERT tokenizer and model. Loaded lazily because the
    weights are ~440 MB and tokenizer + model init takes ~30s on first
    use. Inference is CPU-bound and blocks the GIL, so news.py calls it
    via ``loop.run_in_executor`` to keep the async server responsive."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.torch = None

    def load(self) -> None:
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        import torch  # noqa
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert")
        self.model.eval()
        self.torch = torch

    def classify(self, text: str) -> Dict[str, object]:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        probs = self.torch.softmax(outputs.logits, dim=-1)[0]
        # ProsusAI/finbert label order: 0 = positive, 1 = negative, 2 = neutral
        p_pos = float(probs[0])
        p_neg = float(probs[1])
        p_neu = float(probs[2])
        compound = p_pos - p_neg                 # in [-1, +1]
        if compound >= 0.15:
            label = "bullish"
        elif compound <= -0.15:
            label = "bearish"
        else:
            label = "neutral"
        return {
            "compound": round(compound, 3),
            "label": label,
            "p_pos": round(p_pos, 3),
            "p_neg": round(p_neg, 3),
            "p_neu": round(p_neu, 3),
        }


_FINBERT: Optional[_FinBertEngine] = None
_FINBERT_LOCK = threading.Lock()
_FINBERT_STATE = {"ready": False, "loading": False, "error": None}
_FINBERT_CACHE: Dict[str, Dict] = {}


def _load_finbert_blocking() -> None:
    """Run in background thread by ``warm_finbert``."""
    global _FINBERT
    _FINBERT_STATE["loading"] = True
    try:
        eng = _FinBertEngine()
        eng.load()
        with _FINBERT_LOCK:
            _FINBERT = eng
            _FINBERT_STATE["ready"] = True
    except Exception as exc:           # transformers / torch import or model fail
        _FINBERT_STATE["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _FINBERT_STATE["loading"] = False


def warm_finbert() -> None:
    """Kick off FinBERT loading in a background thread. Idempotent.
    Call once at app startup so the model is ready by the time the first
    real classify request arrives. Does NOT block."""
    if _FINBERT_STATE["ready"] or _FINBERT_STATE["loading"]:
        return
    threading.Thread(target=_load_finbert_blocking, daemon=True).start()


def finbert_ready() -> bool:
    return bool(_FINBERT_STATE["ready"]) and _FINBERT is not None


def finbert_status() -> Dict[str, object]:
    """Snapshot of FinBERT load state for the UI/snapshot."""
    return {
        "ready": _FINBERT_STATE["ready"],
        "loading": _FINBERT_STATE["loading"],
        "error": _FINBERT_STATE["error"],
    }


def classify_finbert(text: str) -> Optional[Dict[str, object]]:
    """Synchronous FinBERT classify. Returns ``None`` if the model isn't
    loaded yet (caller should treat that as 'use VADER instead'). Caches
    results so identical headlines aren't re-tokenized every refresh."""
    if not text or not finbert_ready():
        return None
    key = text.lower()[:200]
    cached = _FINBERT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        assert _FINBERT is not None
        result = _FINBERT.classify(text)
    except Exception:
        return None
    if len(_FINBERT_CACHE) < 5000:
        _FINBERT_CACHE[key] = result
    return result


def aggregate(items: list) -> Dict[str, object]:
    """Aggregate sentiment across a batch of news items.

    Each item should have a ``sentiment_score`` field (compound from
    classify). Returns an overall label/score plus counts per category."""
    if not items:
        return {"label": "neutral", "compound": 0.0,
                "bullish": 0, "bearish": 0, "neutral": 0, "count": 0}

    scores = [float(it.get("sentiment_score", 0.0)) for it in items]
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for it in items:
        cat = str(it.get("sentiment", "neutral"))
        if cat in counts:
            counts[cat] += 1
        else:
            counts["neutral"] += 1

    avg = sum(scores) / len(scores) if scores else 0.0
    if avg >= 0.10:
        label = "bullish"
    elif avg <= -0.10:
        label = "bearish"
    else:
        label = "neutral"
    return {
        "label": label,
        "compound": round(avg, 3),
        "bullish": counts["bullish"],
        "bearish": counts["bearish"],
        "neutral": counts["neutral"],
        "count": len(items),
    }

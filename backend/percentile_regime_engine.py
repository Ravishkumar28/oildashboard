"""Per-product percentile-based regime tagger.

Loads pre-computed quintile cutoffs from backend/data/percentile_regime_cutoffs.json
(generated offline by tools/compare_regime_labels.py from the user's xlsx data).

For each product, applies the FROZEN cutoffs to today's live slope (M12 - M1
from real_curves.json or the live yfinance per-contract feed) to produce a
quintile label Q1..Q5. Returns both the percentile label and the existing
hard-cutoff label side-by-side for honest comparison.

This is NOT a prediction model — it's a more meaningful LABELING SCHEME that
adapts to each product's slope distribution rather than using one-size-fits-all
WTI-derived thresholds.

Output schema:
    {
      "available": True,
      "explainer": "...",
      "products": [
        {
          "product": "CL",
          "current_slope": -8.4,
          "hard_label": "Steep Backwardation",
          "percentile_label": "Q1 (deepest back)",
          "percentile_idx": 0,            # 0..4
          "hard_idx": 0,                  # 0..4
          "agreement_ordinal": True,
          "p20": -9.27, "p40": -5.79, "p60": -3.92, "p80": -2.41,
          "history_window": "2021-01 -> 2026-05",
        },
        ...
      ],
      "n_agreement": 4,
      "n_total": 5,
    }
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

_CUTOFFS_PATH = Path(__file__).parent / "data" / "percentile_regime_cutoffs.json"
_CURVES_PATH  = Path(__file__).parent / "data" / "real_curves.json"


HARD_CUTS = (-4.0, -1.0, 1.0, 4.0)
HARD_NAMES = ["Steep Backwardation", "Backwardation", "Flat",
               "Contango", "Steep Contango"]
PCT_NAMES = ["Q1 (deepest back)", "Q2", "Q3 (middle)", "Q4", "Q5 (deepest contango)"]
PCT_SHORT = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def _hard_label_idx(slope: float):
    for i, c in enumerate(HARD_CUTS):
        if slope <= c:
            return HARD_NAMES[i], i
    return HARD_NAMES[4], 4


def _pct_label_idx(slope: float, cuts):
    for i, c in enumerate(cuts):
        if slope <= c:
            return PCT_NAMES[i], i
    return PCT_NAMES[4], 4


_CUTOFFS_CACHE: Optional[Dict] = None
_CURVES_CACHE: Optional[Dict] = None


def _load_cutoffs() -> Dict:
    global _CUTOFFS_CACHE
    if _CUTOFFS_CACHE is None:
        if not _CUTOFFS_PATH.exists():
            _CUTOFFS_CACHE = {}
        else:
            try:
                _CUTOFFS_CACHE = json.loads(_CUTOFFS_PATH.read_text())
            except Exception:
                _CUTOFFS_CACHE = {}
    return _CUTOFFS_CACHE


def _load_curves() -> Dict:
    global _CURVES_CACHE
    if _CURVES_CACHE is None:
        if not _CURVES_PATH.exists():
            _CURVES_CACHE = {}
        else:
            try:
                _CURVES_CACHE = json.loads(_CURVES_PATH.read_text())
            except Exception:
                _CURVES_CACHE = {}
    return _CURVES_CACHE


def _current_slope(product_code: str) -> Optional[float]:
    """M12 - M1 from the latest curve snapshot for this product."""
    curves = _load_curves()
    d = curves.get(product_code)
    if not d:
        return None
    hist = d.get("history") or []
    if not hist:
        return None
    last = hist[-1]
    prices = last.get("prices") or []
    if len(prices) < 12 or prices[0] is None or prices[11] is None:
        return None
    try:
        return float(prices[11]) - float(prices[0])
    except (TypeError, ValueError):
        return None


def build_panel() -> Dict:
    cutoffs = _load_cutoffs()
    if not cutoffs:
        return {"available": False,
                "reason": "percentile_regime_cutoffs.json missing — run tools/compare_regime_labels.py"}

    products_out: List[Dict] = []
    n_agree = 0
    for prod_code, cf in cutoffs.items():
        slope = _current_slope(prod_code)
        if slope is None:
            continue
        cuts = (cf["p20"], cf["p40"], cf["p60"], cf["p80"])
        hard_lbl, hard_idx = _hard_label_idx(slope)
        pct_lbl,  pct_idx  = _pct_label_idx(slope, cuts)
        ordinal_agree = hard_idx == pct_idx
        if ordinal_agree:
            n_agree += 1
        products_out.append({
            "product":         prod_code,
            "current_slope":   round(slope, 3),
            "hard_label":      hard_lbl,
            "hard_idx":        hard_idx,
            "percentile_label": pct_lbl,
            "percentile_short": PCT_SHORT[pct_idx],
            "percentile_idx":   pct_idx,
            "agreement_ordinal": ordinal_agree,
            "p20": round(cuts[0], 3),
            "p40": round(cuts[1], 3),
            "p60": round(cuts[2], 3),
            "p80": round(cuts[3], 3),
            "history_window":  cf.get("fit_range", ""),
            "n_days_history":  cf.get("n_days_history"),
        })

    return {
        "available":  True,
        "products":   products_out,
        "n_agreement": n_agree,
        "n_total":    len(products_out),
        "explainer": (
            "Per-product quintile (Q1-Q5) regime labels derived from each "
            "product's M12-M1 slope distribution over its full xlsx history. "
            "Compared side-by-side with the legacy hard-cutoff labels "
            "(Steep Back / Back / Flat / Contango / Steep Contango). The "
            "hard cutoffs use WTI-style thresholds (-4, -1, +1, +4) which "
            "produce 90%+ class imbalance on products with different scale "
            "(LGO in $/mt, HO in $/gal) — percentile labels normalize this."),
    }

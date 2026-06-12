"""Multi-dimensional regime fingerprint.

Phase 2: instead of one regime label (just curve shape), classify the
market state along 5 independent dimensions. The TUPLE of these labels
is the "regime fingerprint" — two markets with the same fingerprint
should behave similarly, and historical observations sharing a
fingerprint are the right reference set for "what's normal here."

Dimensions:

  1. Inventory   — DOE crude stocks vs trailing 60d (z-score buckets)
  2. Volatility  — WTI 20d realized vol vs trailing 60d (z-score buckets)
  3. Seasonality — calendar phase from week-of-year:
                    1-9     winter heating
                    10-22   spring shoulder / refinery turnaround
                    23-35   summer driving
                    36-44   fall refinery turnaround
                    45-52   winter heating
  4. Curve       — M12-M1 slope buckets (existing Phase 1 rule)
  5. Macro       — DXY 20d change buckets (strong/neutral/weak dollar)

Each dimension returns (bucket_idx 0..K, bucket_label). The full
fingerprint is the tuple of 5 labels. Helper `regime_key()` returns a
canonical string used as the dictionary key into the historical DB.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple


# ----- bucket cutoffs (z-score thresholds for inv/vol/macro) ---------------
# Tuned so each bucket gets ~equal historical mass under a normal-ish
# distribution. ±0.5 and ±1.0 z give roughly 30/40/30 split for the
# inner three buckets and ~15% in each tail.
_BUCKETS_Z5 = [
    (-1.0,  "Very Low"),
    (-0.3,  "Low"),
    ( 0.3,  "Normal"),
    ( 1.0,  "High"),
    ( float("inf"), "Very High"),
]

_CURVE_CUTS: Tuple[float, float, float, float] = (-4.0, -1.0, 1.0, 4.0)
_CURVE_NAMES = ["Steep Backwardation", "Backwardation", "Flat",
                "Contango", "Steep Contango"]

_MACRO_CUTS_PCT = (-2.0, -0.5, 0.5, 2.0)   # 20-day DXY % change cutoffs
_MACRO_NAMES = ["Crashing Dollar", "Weak Dollar", "Neutral Dollar",
                "Strong Dollar", "Spiking Dollar"]


def _bucket_z(z: float, buckets=_BUCKETS_Z5) -> Tuple[int, str]:
    for i, (cut, label) in enumerate(buckets):
        if z <= cut:
            return i, label
    return len(buckets) - 1, buckets[-1][1]


def _zscore(series: List[float], window: int = 60) -> float:
    if not series or len(series) < 5:
        return 0.0
    tail = series[-window:]
    mean = sum(tail) / len(tail)
    var = sum((x - mean) ** 2 for x in tail) / len(tail)
    sd = var ** 0.5
    if sd == 0:
        return 0.0
    return (series[-1] - mean) / sd


def _realized_vol(series: List[float], window: int = 20) -> float:
    """Annualized log-return std over the window. Returns 0 if too short."""
    import math
    if not series or len(series) < window + 1:
        return 0.0
    rets = []
    for i in range(len(series) - window, len(series)):
        if i == 0 or series[i - 1] <= 0:
            continue
        rets.append(math.log(series[i] / series[i - 1]))
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    sd = var ** 0.5
    return sd * math.sqrt(252) * 100   # annualized vol in %


def _classify_inventory(inv_hist: List[float]) -> Tuple[int, str, float]:
    z = _zscore(inv_hist, window=60)
    idx, label = _bucket_z(z)
    return idx, label, round(z, 2)


def _classify_volatility(price_hist: List[float]) -> Tuple[int, str, float]:
    """Vol regime = realized vol vs trailing 60d of realized vols."""
    if len(price_hist) < 80:
        return 2, "Normal", 0.0
    vols: List[float] = []
    for t in range(20, len(price_hist) + 1):
        vols.append(_realized_vol(price_hist[:t], window=20))
    z = _zscore(vols, window=60)
    idx, label = _bucket_z(z)
    return idx, label, round(z, 2)


def _classify_seasonality(today: Optional[dt.date] = None) -> Tuple[int, str, int]:
    """Returns (idx, label, week_of_year)."""
    if today is None:
        try:
            today = dt.date.today()
        except Exception:
            today = dt.date(2026, 1, 1)
    wk = today.isocalendar().week
    if 1 <= wk <= 9 or wk >= 45:
        return 0, "Winter Heating", wk
    if 10 <= wk <= 16:
        return 1, "Spring Refinery Turnaround", wk
    if 17 <= wk <= 22:
        return 2, "Spring Shoulder", wk
    if 23 <= wk <= 35:
        return 3, "Summer Driving", wk
    if 36 <= wk <= 44:
        return 4, "Fall Refinery Turnaround", wk
    return 2, "Spring Shoulder", wk


def _classify_curve(curve_prices: List[float]) -> Tuple[int, str, float]:
    if not curve_prices or len(curve_prices) < 9:
        return 2, "Flat", 0.0
    slope = curve_prices[-1] - curve_prices[0]
    a, b, c, d = _CURVE_CUTS
    if slope < a:  return 0, _CURVE_NAMES[0], round(slope, 2)
    if slope < b:  return 1, _CURVE_NAMES[1], round(slope, 2)
    if slope <= c: return 2, _CURVE_NAMES[2], round(slope, 2)
    if slope <= d: return 3, _CURVE_NAMES[3], round(slope, 2)
    return 4, _CURVE_NAMES[4], round(slope, 2)


def _classify_macro(dxy_hist: List[float]) -> Tuple[int, str, float]:
    """Macro regime from DXY 20-day percent change."""
    if not dxy_hist or len(dxy_hist) < 21:
        return 2, "Neutral Dollar", 0.0
    pct = (dxy_hist[-1] - dxy_hist[-21]) / dxy_hist[-21] * 100.0
    a, b, c, d = _MACRO_CUTS_PCT
    if pct < a:  return 0, _MACRO_NAMES[0], round(pct, 2)
    if pct < b:  return 1, _MACRO_NAMES[1], round(pct, 2)
    if pct <= c: return 2, _MACRO_NAMES[2], round(pct, 2)
    if pct <= d: return 3, _MACRO_NAMES[3], round(pct, 2)
    return 4, _MACRO_NAMES[4], round(pct, 2)


# ----- public API ----------------------------------------------------------

def classify(price_hist: List[float], curve_prices: List[float],
             inv_hist: List[float], dxy_hist: List[float],
             today: Optional[dt.date] = None) -> Dict:
    """Full 5-dimension regime fingerprint."""
    inv_idx, inv_label, inv_z = _classify_inventory(inv_hist)
    vol_idx, vol_label, vol_z = _classify_volatility(price_hist)
    sea_idx, sea_label, week  = _classify_seasonality(today)
    cur_idx, cur_label, slope = _classify_curve(curve_prices)
    mac_idx, mac_label, dxy_pct = _classify_macro(dxy_hist)

    return {
        "dimensions": [
            {"name": "Inventory",  "bucket": inv_idx, "label": inv_label,
             "metric": "z-score vs 60d", "value": inv_z},
            {"name": "Volatility", "bucket": vol_idx, "label": vol_label,
             "metric": "realized-vol z vs 60d", "value": vol_z},
            {"name": "Seasonality","bucket": sea_idx, "label": sea_label,
             "metric": "week-of-year", "value": week},
            {"name": "Curve",      "bucket": cur_idx, "label": cur_label,
             "metric": "M12-M1 slope $/bbl", "value": slope},
            {"name": "Macro",      "bucket": mac_idx, "label": mac_label,
             "metric": "DXY 20d % change", "value": dxy_pct},
        ],
        "fingerprint": (inv_idx, vol_idx, sea_idx, cur_idx, mac_idx),
        "fingerprint_label": (
            f"{inv_label} Inv · {vol_label} Vol · "
            f"{sea_label} · {cur_label} · {mac_label}"
        ),
    }


def classify_from_arrays(price_hist: List[float], curve_prices: List[float],
                          inv_hist: List[float], dxy_hist: List[float],
                          historical_index: int,
                          today: Optional[dt.date] = None) -> Tuple:
    """Classify the regime AT a historical point in `price_hist[:historical_index+1]`.
    Used by regime_history.py to backfill the historical regime database."""
    p = price_hist[:historical_index + 1]
    inv = inv_hist[:historical_index + 1] if inv_hist else []
    dxy = dxy_hist[:historical_index + 1] if dxy_hist else []
    fp = classify(p, curve_prices, inv, dxy, today)
    return fp["fingerprint"]


def regime_key(fingerprint: Tuple) -> str:
    """Canonical dict key for the regime DB."""
    return "/".join(str(x) for x in fingerprint)

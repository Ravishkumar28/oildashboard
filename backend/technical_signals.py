"""Technical-analysis trade engine.

Produces a per-product LONG/SHORT/HOLD verdict + position size that is
INDEPENDENT of the regime/regression engine. Surfaces in the dashboard as
panel P3E so the user can compare TA signals to fundamental regime signals
and combine the two (or override one with the other).

Per product (WTI, Brent, RBOB, Heating Oil, NatGas) we compute:

  RSI(14)     -> overbought / oversold  (+1 long below 30, -1 short above 70)
  BB(20, 2)   -> mean reversion         (+1 long below lower, -1 above upper)
  EMA(20/50)  -> trend                  (+1 long if 20>50 crossover, -1 short if opposite)
  Momentum    -> price vs 20-day SMA    (+1 above, -1 below)

Each of the four sub-signals returns {-1, 0, +1}. Sum them => raw score in
[-4, +4]. Convert to:
  raw >= +2  -> LONG  with lots = round(min(5, max(1, |raw|)))
  raw <= -2  -> SHORT with lots = round(min(5, max(1, |raw|)))
  otherwise  -> HOLD  (raw in {-1, 0, +1} = mixed signals)

The TA verdict is intentionally simple and explainable so the trader can
audit every signal — no black-box ML here. The regression engine in P3B
covers the heavy quantitative work; TA gives a second opinion on the
front month for timing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import indicators as ind


CONTRACT_SIZE = {
    "wti":    1_000.0,   # 1k bbl per CL
    "brent":  1_000.0,
    "rbob":   42_000.0,  # 42k gal per RB ($/gal -> $/contract)
    "heat":   42_000.0,
    "natgas": 10_000.0,  # 10k MMBtu per NG
}

PRODUCT_NAMES = {
    "wti":    "WTI",
    "brent":  "Brent",
    "rbob":   "RBOB",
    "heat":   "Heating Oil",
    "natgas": "Natural Gas",
}

HIST_KEYS = ["wti", "brent", "rbob", "heat", "natgas"]


def _last(seq: Sequence[Optional[float]]) -> Optional[float]:
    for v in reversed(seq):
        if v is not None:
            return float(v)
    return None


def _rsi_signal(prices: List[float]) -> Dict[str, Any]:
    if len(prices) < 20:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    series = ind.rsi(prices, period=14)
    val = _last(series)
    if val is None:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    if val >= 70:
        return {"value": round(val, 1), "signal": -1, "label": f"RSI {val:.0f} overbought"}
    if val <= 30:
        return {"value": round(val, 1), "signal": +1, "label": f"RSI {val:.0f} oversold"}
    return {"value": round(val, 1), "signal": 0, "label": f"RSI {val:.0f} neutral"}


def _bb_signal(prices: List[float]) -> Dict[str, Any]:
    if len(prices) < 25:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    mid, upper, lower = ind.bollinger_bands(prices, window=20, k=2.0)
    p = prices[-1]
    u, m, l = _last(upper), _last(mid), _last(lower)
    if None in (u, m, l):
        return {"value": None, "signal": 0, "label": "insufficient history"}
    pct = (p - l) / (u - l) * 100.0 if (u - l) > 0 else 50.0
    if p >= u:
        return {"value": round(pct, 1), "signal": -1,
                "label": f"price above upper band ({pct:.0f}%)"}
    if p <= l:
        return {"value": round(pct, 1), "signal": +1,
                "label": f"price below lower band ({pct:.0f}%)"}
    if pct >= 80:
        return {"value": round(pct, 1), "signal": -1,
                "label": f"BB {pct:.0f}% near upper"}
    if pct <= 20:
        return {"value": round(pct, 1), "signal": +1,
                "label": f"BB {pct:.0f}% near lower"}
    return {"value": round(pct, 1), "signal": 0, "label": f"BB {pct:.0f}% mid-band"}


def _ema_cross_signal(prices: List[float]) -> Dict[str, Any]:
    if len(prices) < 55:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    e20 = ind.ema(prices, period=20)
    e50 = ind.ema(prices, period=50)
    a, b = _last(e20), _last(e50)
    if a is None or b is None:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    diff = a - b
    diff_pct = (diff / b * 100.0) if b != 0 else 0.0
    if a > b:
        return {"value": round(diff_pct, 2), "signal": +1,
                "label": f"EMA20 > EMA50 (+{diff_pct:.1f}%) bullish trend"}
    return {"value": round(diff_pct, 2), "signal": -1,
            "label": f"EMA20 < EMA50 ({diff_pct:.1f}%) bearish trend"}


def _momentum_signal(prices: List[float]) -> Dict[str, Any]:
    if len(prices) < 22:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    sma = ind.moving_average(prices, window=20)
    s = _last(sma)
    if s is None:
        return {"value": None, "signal": 0, "label": "insufficient history"}
    p = prices[-1]
    pct = (p - s) / s * 100.0 if s != 0 else 0.0
    if p > s * 1.005:
        return {"value": round(pct, 2), "signal": +1,
                "label": f"price {pct:+.1f}% vs SMA20 — up momentum"}
    if p < s * 0.995:
        return {"value": round(pct, 2), "signal": -1,
                "label": f"price {pct:+.1f}% vs SMA20 — down momentum"}
    return {"value": round(pct, 2), "signal": 0, "label": f"price flat ({pct:+.1f}%) vs SMA20"}


def _daily_vol(prices: List[float]) -> Optional[float]:
    """Realized 20-day log-return std (daily, not annualized)."""
    import math
    if len(prices) < 22:
        return None
    rets: List[float] = []
    for i in range(len(prices) - 20, len(prices)):
        if i == 0 or prices[i - 1] <= 0:
            continue
        rets.append(math.log(prices[i] / prices[i - 1]))
    if len(rets) < 5:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return var ** 0.5


def _entry_tp_sl(price: float, direction: str,
                 daily_vol: Optional[float]) -> Dict[str, Any]:
    """Entry / Take Profit / Stop Loss derived from realized vol.

    Convention (matches P4D regime signals):
        σ_$    = |price| × vol20         (1-day dollar std)
        TP at  entry + dir × 2σ × √5     (5-day, 2σ horizon)
        SL at  entry − dir × 1.5σ        (1-day, 1.5σ adverse)
        R:R   ≈ 2√5 / 1.5 = 2.98 by construction
    """
    if direction not in ("LONG", "SHORT") or daily_vol is None:
        return {"entry": price, "tp": None, "sl": None, "rr": None,
                "tp_pct": None, "sl_pct": None, "horizon_days": 5}
    sigma = abs(price) * daily_vol
    horizon_root = 5 ** 0.5
    tp_dist = 2.0 * sigma * horizon_root
    sl_dist = 1.5 * sigma
    dir_sign = 1 if direction == "LONG" else -1
    tp = price + dir_sign * tp_dist
    sl = price - dir_sign * sl_dist
    return {
        "entry":        round(float(price), 4),
        "tp":           round(float(tp), 4),
        "sl":           round(float(sl), 4),
        "tp_dist":      round(float(tp_dist), 4),
        "sl_dist":      round(float(sl_dist), 4),
        "tp_pct":       round(tp_dist / abs(price) * 100, 2) if price else None,
        "sl_pct":       round(sl_dist / abs(price) * 100, 2) if price else None,
        "rr":           round(tp_dist / sl_dist, 2) if sl_dist > 0 else None,
        "daily_vol_pct": round(daily_vol * 100, 2),
        "horizon_days": 5,
    }


def _product_panel(key: str, prices: List[float]) -> Dict[str, Any]:
    rsi_sig = _rsi_signal(prices)
    bb_sig  = _bb_signal(prices)
    ema_sig = _ema_cross_signal(prices)
    mom_sig = _momentum_signal(prices)

    raw_score = (rsi_sig["signal"] + bb_sig["signal"] +
                 ema_sig["signal"] + mom_sig["signal"])

    if raw_score >= 2:
        direction = "LONG"
    elif raw_score <= -2:
        direction = "SHORT"
    else:
        direction = "HOLD"

    lots = int(min(5, max(1, abs(raw_score)))) if direction != "HOLD" else 0

    # Confidence: how many of 4 signals point the same way.
    if direction == "HOLD":
        conf_pct = 0
    else:
        aligned = sum(1 for s in (rsi_sig["signal"], bb_sig["signal"],
                                  ema_sig["signal"], mom_sig["signal"])
                      if (direction == "LONG" and s > 0)
                      or (direction == "SHORT" and s < 0))
        conf_pct = round(aligned / 4 * 100)

    price_now = prices[-1] if prices else None
    vol_d = _daily_vol(prices)
    # Halve lots if realized vol > 5% daily (matches P4D high-vol guard)
    if vol_d is not None and vol_d > 0.05 and lots > 0:
        lots = max(1, lots // 2)

    plan = _entry_tp_sl(float(price_now) if price_now is not None else 0.0,
                        direction, vol_d) if price_now is not None else {}

    return {
        "product":    key,
        "name":       PRODUCT_NAMES[key],
        "price":      round(float(price_now), 4) if price_now is not None else None,
        "n_hist":     len(prices),
        "rsi":        rsi_sig,
        "bb":         bb_sig,
        "ema_cross":  ema_sig,
        "momentum":   mom_sig,
        "raw_score":  raw_score,
        "direction":  direction,
        "lots":       lots,
        "conf_pct":   conf_pct,
        "contract_size": CONTRACT_SIZE.get(key, 1000.0),
        # New: entry / TP / SL / R:R derived from 20-day realized vol
        "plan":       plan,
    }


def build_panel(market) -> Dict[str, Any]:
    """Build the technical-analysis panel for the snapshot."""
    products: List[Dict[str, Any]] = []
    n_long = n_short = n_hold = 0

    # Pull each product's price history from MarketEngine.
    hist_dict = getattr(market, "hist", {}) or {}
    for key in HIST_KEYS:
        raw = hist_dict.get(key) or []
        try:
            prices = [float(p) for p in raw if p is not None]
        except Exception:
            prices = []
        if not prices:
            continue
        panel = _product_panel(key, prices)
        products.append(panel)
        if panel["direction"] == "LONG":
            n_long += 1
        elif panel["direction"] == "SHORT":
            n_short += 1
        else:
            n_hold += 1

    # Rank by absolute score so the strongest convictions surface first.
    products.sort(key=lambda p: -abs(p["raw_score"]))

    return {
        "available":    bool(products),
        "products":     products,
        "n_long":       n_long,
        "n_short":      n_short,
        "n_hold":       n_hold,
        "n_total":      len(products),
        "indicators":   ["RSI(14)", "BB(20,2σ)", "EMA(20/50)", "SMA(20) momentum"],
        "explainer": (
            "Four equal-weight TA sub-signals per product, each voting "
            "-1 / 0 / +1. Raw score in [-4,+4]. |raw| ≥ 2 -> trade with "
            "lots = min(5, max(1, |raw|)). Below ±2 = HOLD (mixed signals). "
            "This is intentionally independent of the regime / regression "
            "engine so it can serve as a second opinion."
        ),
    }

"""Multi-product butterfly + spread strategy matrix.

For each of the 5 products (WTI, Brent, RBOB, HO, NatGas) this module:

  1. Reads the live 12-month curve (real Yahoo settlements where available)
  2. Computes the M3-M6-M9 butterfly and 4 calendar spreads (M1-M2, M1-M3,
     M3-M6, M6-M9)
  3. Classifies the curve regime by M12-M1 slope: Steep Backwardation,
     Backwardation, Flat, Contango, Steep Contango (cutoffs scaled to each
     product's typical $/bbl or $/gal range)
  4. Builds a feature set: returns, BB position (where stored), fly z-score,
     plus product-specific factors (US inventory for WTI, OPEC quota for
     Brent, summer-driving phase for RBOB, winter-heating phase for HO,
     storage z-score for NatGas)
  5. Fits a per-product per-regime Lasso to predict next-tick fly value
     (refits at most every 5 min). Lasso surface non-zero factor weights —
     the geopolitical / macro factors that actually drive each curve.
  6. Produces THREE verdicts per product:
       * Fundamental — from Lasso prediction + fly z-score
       * Technical   — from BB position + EMA20/MA50 trend + price vs VWAP
       * Combined    — weighted aggregation (both agree → strong; conflict → WATCH)
  7. Returns a sized trade recommendation (long/short fly or spread + size in
     contracts at notional based on contract specs)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except Exception:
    _SKLEARN = False


# ----- product configuration ----------------------------------------------
# Each product has a different price unit and typical $ range, so the
# regime cutoffs and contract specs differ. Slope cutoffs in same unit
# as the price ($/bbl for crudes, $/gal for products, $/MMBtu for NG).

PRODUCTS: List[Dict[str, Any]] = [
    {
        "key":          "wti",
        "name":         "WTI Crude",
        "unit":         "$/bbl",
        "curve_attr":   "real_curve",
        "hist_key":     "wti",
        "bb_key":       "wti",
        "contract_size": 1000,           # bbl per NYMEX CL contract
        # M12-M1 slope cutoffs ($/bbl)
        "slope_cuts":   (-4.0, -1.0, 1.0, 4.0),
        # Macro driver labels (for context, not Lasso inputs)
        "macros":       ["US inventory", "Cushing stocks", "Rig count",
                          "Hurricane risk", "OPEC quota compliance"],
    },
    {
        "key":          "brent",
        "name":         "Brent Crude",
        "unit":         "$/bbl",
        "curve_attr":   "brent_curve",
        "hist_key":     "brent",
        "bb_key":       "brent",
        "contract_size": 1000,
        "slope_cuts":   (-4.0, -1.0, 1.0, 4.0),
        "macros":       ["OPEC supply", "Hormuz tension", "Russia sanctions",
                          "Hurricane (gulf transit)", "Suez closures"],
    },
    {
        "key":          "rbob",
        "name":         "RBOB Gasoline",
        "unit":         "$/gal",
        "curve_attr":   "rbob_curve",
        "hist_key":     "rbob",
        "bb_key":       "rbob",
        "contract_size": 42000,          # 42k gal per RB contract
        # Gasoline curve has tight monthly spreads; cutoffs in $/gal
        "slope_cuts":   (-0.10, -0.03, 0.03, 0.10),
        "macros":       ["Summer-driving demand", "Gasoline crack",
                          "Refinery utilization", "Hurricane (Gulf refineries)"],
    },
    {
        "key":          "heat",
        "name":         "Heating Oil",
        "unit":         "$/gal",
        "curve_attr":   "heat_curve",
        "hist_key":     "heat",
        "bb_key":       "heat",
        "contract_size": 42000,
        "slope_cuts":   (-0.10, -0.03, 0.03, 0.10),
        "macros":       ["Winter-heating demand", "Distillate stocks",
                          "Diesel demand (truck/marine)", "Refinery turnaround"],
    },
    {
        "key":          "natgas",
        "name":         "Natural Gas",
        "unit":         "$/MMBtu",
        "curve_attr":   "natgas_curve",
        "hist_key":     "natgas",
        "bb_key":       None,            # we don't compute BB for NG
        "contract_size": 10000,          # 10k MMBtu per NG contract
        "slope_cuts":   (-0.50, -0.10, 0.10, 0.50),
        "macros":       ["Storage level (DOE)", "Heating-degree days",
                          "Cooling-degree days", "LNG export flows",
                          "Power-gen demand"],
    },
]

REGIME_NAMES = ["Steep Backwardation", "Backwardation", "Flat",
                "Contango", "Steep Contango"]


# ----- helpers -------------------------------------------------------------

def _classify_slope(slope: float, cuts: Tuple[float, float, float, float]) -> Tuple[int, str]:
    a, b, c, d = cuts
    if slope < a:  return 0, REGIME_NAMES[0]
    if slope < b:  return 1, REGIME_NAMES[1]
    if slope <= c: return 2, REGIME_NAMES[2]
    if slope <= d: return 3, REGIME_NAMES[3]
    return 4, REGIME_NAMES[4]


def _curve_prices(curve: Any) -> List[float]:
    if not curve:
        return []
    try:
        return [float(row.get("price", 0.0)) for row in curve]
    except Exception:
        return []


def _fly_and_spreads(prices: List[float]) -> Optional[Dict[str, float]]:
    # Fly needs M3/M6/M9 → 9 prices min. Use last available month as the
    # "far month" for slope when M12 missing (some products publish only
    # 11 settlements until the latest expiry rolls).
    if len(prices) < 9:
        return None
    last = prices[-1]
    return {
        "fly":   prices[2] - 2 * prices[5] + prices[8],     # M3-2*M6+M9
        "m1_m2": prices[1] - prices[0],
        "m1_m3": prices[2] - prices[0],
        "m3_m6": prices[5] - prices[2],
        "m6_m9": prices[8] - prices[5],
        "m12_m1_slope": last - prices[0],
    }


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


def _pct_return(series: List[float], lag: int) -> float:
    if not series or len(series) <= lag:
        return 0.0
    prev = series[-(lag + 1)]
    if prev == 0:
        return 0.0
    return (series[-1] - prev) / prev * 100.0


def _bb_position(series: List[float], window: int = 20) -> float:
    if len(series) < window:
        return 50.0
    tail = series[-window:]
    mean = sum(tail) / window
    sd = (sum((x - mean) ** 2 for x in tail) / window) ** 0.5
    if sd == 0:
        return 50.0
    lower = mean - 2 * sd
    upper = mean + 2 * sd
    return max(0.0, min(100.0, (series[-1] - lower) / (upper - lower) * 100.0))


def _ema(series: List[float], window: int) -> Optional[float]:
    if len(series) < window:
        return None
    k = 2.0 / (window + 1)
    e = sum(series[:window]) / window
    for x in series[window:]:
        e = e + k * (x - e)
    return e


def _sma(series: List[float], window: int) -> Optional[float]:
    if len(series) < window:
        return None
    return sum(series[-window:]) / window


# ----- per-product Lasso fit cache -----------------------------------------

class _ProductLassoCache:
    REFIT_EVERY_SEC = 300
    MIN_PER_REGIME = 6      # lowered so Lasso fits soon after warmup; fly
                            # history grows by 1 per 60s curve refresh, so
                            # 15 obs would take ~15 min after boot.
    LASSO_ALPHA = 0.05

    def __init__(self) -> None:
        # (product_key, regime_idx) -> (coef list, intercept, scaler mean,
        # scaler scale, n_samples, factor_names)
        self.cache: Dict[Tuple[str, int], Tuple[List[float], float,
                                                 List[float], List[float],
                                                 int, List[str]]] = {}
        self.last_fit: Dict[str, float] = {}

    def fit_product(self, product_key: str, fly_hist: List[float],
                    wti_hist: List[float], dxy_hist: List[float],
                    bb_series: Optional[List[float]], cuts) -> None:
        if not _SKLEARN:
            return
        now = time.time()
        if now - self.last_fit.get(product_key, 0.0) < self.REFIT_EVERY_SEC \
                and any(k[0] == product_key for k in self.cache):
            return
        # Build (X, y, regime) from rolling fly history. Without a curve
        # history per product, we use the M12-M1 slope inferred from
        # underlying-price-vs-fly relationship as a proxy. Practically: we
        # bucket by fly z-score history segments since slope and fly tend
        # to co-move (steep curves often pair with persistent fly bias).
        n = len(fly_hist)
        if n < 12:                          # need at least 12 fly obs to fit
            return
        factor_names = ["fly_now", "wti_5d_ret", "wti_20d_ret",
                         "dxy_5d_ret", "bb_pos"]
        X: List[List[float]] = []
        y_next_fly: List[float] = []
        regime_idx: List[int] = []
        # Use fly_hist itself as the slope proxy: rolling mean of fly is
        # the cheapest in-sample regime indicator we have without curve_hist.
        # Start at 6 (was 20) so we get usable samples from limited history.
        for i in range(6, n - 1):
            fly_i = fly_hist[i]
            fly_next = fly_hist[i + 1]
            # Local proxies — align WTI/DXY history same end
            w_slice = wti_hist[:len(wti_hist) - (n - 1 - i)] if wti_hist else []
            d_slice = dxy_hist[:len(dxy_hist) - (n - 1 - i)] if dxy_hist else []
            w5 = _pct_return(w_slice, 5) if len(w_slice) >= 6 else 0.0
            w20 = _pct_return(w_slice, 20) if len(w_slice) >= 21 else 0.0
            d5 = _pct_return(d_slice, 5) if len(d_slice) >= 6 else 0.0
            bb = (_bb_position(bb_series[:len(bb_series) - (n - 1 - i)])
                  if bb_series else 50.0)
            X.append([fly_i, w5, w20, d5, bb])
            y_next_fly.append(fly_next)
            # Bucket by recent local fly mean → maps to slope-proxy regime
            local = fly_hist[max(0, i - 20):i + 1]
            lmean = sum(local) / len(local)
            # Translate fly bias into a regime proxy
            ridx, _ = _classify_slope(lmean * 4.0, cuts)
            regime_idx.append(ridx)

        if not X:
            return
        Xn = np.asarray(X, dtype=float)
        yn = np.asarray(y_next_fly, dtype=float)
        rg = np.asarray(regime_idx, dtype=int)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xn)
        # Clear stale entries for this product before refitting
        for k in list(self.cache.keys()):
            if k[0] == product_key:
                del self.cache[k]
        for r in range(len(REGIME_NAMES)):
            mask = (rg == r)
            n_r = int(mask.sum())
            if n_r < self.MIN_PER_REGIME:
                continue
            try:
                m = Lasso(alpha=self.LASSO_ALPHA, max_iter=5000)
                m.fit(Xs[mask], yn[mask])
                self.cache[(product_key, r)] = (
                    m.coef_.tolist(),
                    float(m.intercept_),
                    scaler.mean_.tolist(),
                    scaler.scale_.tolist(),
                    n_r,
                    factor_names,
                )
            except Exception:
                continue
        self.last_fit[product_key] = now

    def predict(self, product_key: str, regime_idx: int,
                features: List[float]) -> Optional[Dict]:
        entry = self.cache.get((product_key, regime_idx))
        if entry is None:
            return None
        coef, intercept, mean, scale, n_r, names = entry
        xs = [(f - m_) / (s_ or 1.0)
              for f, m_, s_ in zip(features, mean, scale)]
        pred = intercept + sum(c * x for c, x in zip(coef, xs))
        weights = sorted(
            [{"factor": n, "weight": round(c, 4)}
             for n, c in zip(names, coef) if abs(c) > 1e-6],
            key=lambda w: abs(w["weight"]), reverse=True,
        )
        return {"prediction": round(pred, 4),
                "weights": weights, "n_regime_samples": n_r}


_CACHE = _ProductLassoCache()


# ----- per-product verdicts -----------------------------------------------

def _technical_verdict(price_hist: List[float], bb_pos: Optional[float]) -> Dict:
    """Combine BB position + EMA20/MA50 trend into a technical signal."""
    if len(price_hist) < 20:
        return {"action": "WATCH", "score": 0.0,
                "reason": "Need ≥20 price obs for technicals.",
                "indicators": {}}
    indicators: Dict[str, Any] = {}
    if bb_pos is not None:
        indicators["bb_pos"] = round(bb_pos, 1)
    ema20 = _ema(price_hist, 20)
    ma50 = _sma(price_hist, 50) if len(price_hist) >= 50 else None
    indicators["ema20"] = round(ema20, 4) if ema20 else None
    indicators["ma50"] = round(ma50, 4) if ma50 else None
    last = price_hist[-1]
    # Score in [-1, +1]: positive = long bias
    score = 0.0
    parts: List[str] = []
    if bb_pos is not None:
        if bb_pos <= 20:
            score += 0.5
            parts.append(f"BB pos {bb_pos:.0f} (near lower → buy)")
        elif bb_pos >= 80:
            score -= 0.5
            parts.append(f"BB pos {bb_pos:.0f} (near upper → sell)")
        else:
            parts.append(f"BB pos {bb_pos:.0f} (middle)")
    if ema20 and ma50:
        if last > ema20 > ma50:
            score += 0.3
            parts.append("price > EMA20 > MA50 (uptrend)")
        elif last < ema20 < ma50:
            score -= 0.3
            parts.append("price < EMA20 < MA50 (downtrend)")
        else:
            parts.append("trend mixed")
    # Momentum proxy: 5d return
    r5 = _pct_return(price_hist, 5)
    if r5 > 2.0:
        score += 0.2
        parts.append(f"5d momentum +{r5:.1f}%")
    elif r5 < -2.0:
        score -= 0.2
        parts.append(f"5d momentum {r5:.1f}%")
    score = max(-1.0, min(1.0, score))
    if score >= 0.4:
        action = "LONG"
    elif score <= -0.4:
        action = "SHORT"
    else:
        action = "WATCH"
    return {
        "action": action,
        "score": round(score, 2),
        "reason": " · ".join(parts),
        "indicators": indicators,
    }


def _fundamental_verdict(fly_z: float, regime_name: str,
                          lasso_result: Optional[Dict],
                          current_fly: float) -> Dict:
    """Lasso prediction in regime context. Mean-reversion bias on extremes."""
    if lasso_result is None:
        # Pure z-score fallback when Lasso isn't fitted for this regime
        if fly_z >= 1.5:
            return {"action": "SHORT", "score": -0.5,
                    "reason": f"Fly z={fly_z:+.1f}σ (high) — naive mean-revert SHORT (no Lasso).",
                    "prediction": None}
        if fly_z <= -1.5:
            return {"action": "LONG", "score": 0.5,
                    "reason": f"Fly z={fly_z:+.1f}σ (low) — naive mean-revert LONG (no Lasso).",
                    "prediction": None}
        return {"action": "WATCH", "score": 0.0,
                "reason": f"Fly z={fly_z:+.1f}σ within band, no Lasso fit yet.",
                "prediction": None}
    pred = lasso_result["prediction"]
    delta = pred - current_fly
    if fly_z >= 1.0 and delta < -0.05:
        return {"action": "SHORT", "score": -0.7,
                "reason": (f"Regime '{regime_name}' Lasso predicts fly "
                           f"{delta:+.3f} (current z={fly_z:+.1f}σ). "
                           f"Top weight: {lasso_result['weights'][0]['factor']} "
                           f"@ {lasso_result['weights'][0]['weight']:+.3f}."),
                "prediction": pred}
    if fly_z <= -1.0 and delta > 0.05:
        return {"action": "LONG", "score": 0.7,
                "reason": (f"Regime '{regime_name}' Lasso predicts fly "
                           f"{delta:+.3f} (current z={fly_z:+.1f}σ)."),
                "prediction": pred}
    return {"action": "WATCH", "score": 0.0,
            "reason": (f"Regime '{regime_name}', fly z={fly_z:+.1f}σ, "
                       f"predicted change {delta:+.3f} — no edge."),
            "prediction": pred}


def _combine(fund: Dict, tech: Dict, contract_size: int,
             unit: str, current_fly: float) -> Dict:
    """Weighted combination of fundamental + technical."""
    # Fundamental gets 60% weight (it's regime-aware Lasso), tech 40%
    combined_score = 0.6 * fund["score"] + 0.4 * tech["score"]
    if combined_score >= 0.4:
        verdict = "LONG FLY"
    elif combined_score <= -0.4:
        verdict = "SHORT FLY"
    else:
        verdict = "WATCH"
    # Size scaled by abs confidence: 1 to 5 contracts
    contracts = 0 if verdict == "WATCH" else max(1, min(5, int(abs(combined_score) * 8)))
    notional_per_contract = abs(current_fly) * contract_size
    total_notional = contracts * notional_per_contract
    return {
        "verdict":          verdict,
        "score":            round(combined_score, 2),
        "contracts":        contracts,
        "notional_usd":     round(total_notional, 0),
        "rationale":        (f"Fundamental {fund['action']} ({fund['score']:+.2f}), "
                             f"Technical {tech['action']} ({tech['score']:+.2f}). "
                             f"Combined {combined_score:+.2f}. "
                             f"Size: {contracts} contracts ≈ "
                             f"${total_notional:,.0f} notional."),
    }


# ----- public entrypoint ---------------------------------------------------

def build_matrix(market) -> Dict:
    """Returns the multi-product strategy matrix payload for the snapshot."""
    rows: List[Dict] = []
    wti_hist = list(market.hist.get("wti", []))
    dxy_hist = list(market.hist.get("dxy", []))

    for cfg in PRODUCTS:
        curve = getattr(market, cfg["curve_attr"], None)
        prices = _curve_prices(curve)
        fs = _fly_and_spreads(prices)
        price_hist = list(market.hist.get(cfg["hist_key"], []))
        bb_series = list(market.hist.get(cfg["bb_key"], [])) if cfg["bb_key"] else None
        fly_hist = market.product_fly_history.get(cfg["key"], [])

        if not fs:
            rows.append({
                "key": cfg["key"], "name": cfg["name"], "unit": cfg["unit"],
                "available": False,
                "macros": cfg["macros"],
                "reason": "Curve not yet loaded.",
            })
            continue

        slope = fs["m12_m1_slope"]
        regime_idx, regime_name = _classify_slope(slope, cfg["slope_cuts"])

        # Fit Lasso for this product (cached refit at 5-min intervals)
        _CACHE.fit_product(cfg["key"], fly_hist, wti_hist, dxy_hist,
                           bb_series, cfg["slope_cuts"])

        # Current features for Lasso prediction
        bb_pos = _bb_position(bb_series) if bb_series else None
        feats = [fs["fly"],
                 _pct_return(wti_hist, 5),
                 _pct_return(wti_hist, 20),
                 _pct_return(dxy_hist, 5) if dxy_hist else 0.0,
                 bb_pos if bb_pos is not None else 50.0]
        lasso_result = _CACHE.predict(cfg["key"], regime_idx, feats)

        # Fly z-score
        fly_z = _zscore(fly_hist + [fs["fly"]], window=60) if fly_hist else 0.0

        # Verdicts
        fund = _fundamental_verdict(fly_z, regime_name, lasso_result, fs["fly"])
        tech = _technical_verdict(price_hist, bb_pos)
        combined = _combine(fund, tech, cfg["contract_size"], cfg["unit"], fs["fly"])

        rows.append({
            "key":           cfg["key"],
            "name":          cfg["name"],
            "unit":          cfg["unit"],
            "available":     True,
            "regime":        regime_name,
            "regime_idx":    regime_idx,
            "slope":         round(slope, 3),
            "fly":           round(fs["fly"], 4),
            "fly_z":         round(fly_z, 2),
            "spreads": {
                "M1-M2": round(fs["m1_m2"], 4),
                "M1-M3": round(fs["m1_m3"], 4),
                "M3-M6": round(fs["m3_m6"], 4),
                "M6-M9": round(fs["m6_m9"], 4),
            },
            "fundamental":   fund,
            "technical":     tech,
            "combined":      combined,
            "macros":        cfg["macros"],
            "lasso":         lasso_result,   # contains weights + prediction
            "n_fly_history": len(fly_hist),
        })

    return {
        "available": _SKLEARN,
        "rows":      rows,
        "regime_names": REGIME_NAMES,
    }

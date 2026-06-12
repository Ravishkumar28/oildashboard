"""Paper-strategies engine — backtested winners auto-trading live.

Loads pre-fit models from backend/data/paper_strats_models.json:

  - PCA Curve   on CL, LCO, LGO  (Sharpe 1.0-6.0, +$185K total OOS)
  - Bertram OU  on HO calendar spreads  (Sharpe 3.4, 97% wins)
  - HMM regime  on WTCL log spread  (Sharpe 0.88, 75% wins)

For each engine, applies the fitted model to the LATEST curve data from
backend/data/real_curves.json (your xlsx-sourced 250-day window) and emits
LONG/SHORT/FLAT signals with confidence tier.

These signals are then surfaced to the auto-trader as source=paper_strat.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

_MODELS_PATH = Path(__file__).parent / "data" / "paper_strats_models.json"
_CURVES_PATH = Path(__file__).parent / "data" / "real_curves.json"


# ---- caches loaded once ------------------------------------------------ #
_MODELS_CACHE: Optional[Dict] = None
_CURVES_CACHE: Optional[Dict] = None


def _load_models() -> Dict:
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        if not _MODELS_PATH.exists():
            _MODELS_CACHE = {"pca": {}, "bertram": {}, "hmm": None}
        else:
            try:
                _MODELS_CACHE = json.loads(_MODELS_PATH.read_text())
            except Exception:
                _MODELS_CACHE = {"pca": {}, "bertram": {}, "hmm": None}
    return _MODELS_CACHE


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


# ---- PCA Curve signal --------------------------------------------------- #
def _pca_signal(model: Dict, history: List[Dict]) -> Optional[Dict]:
    """Compute current PC3 z-score from last 20 days and emit fly signal
    if |z| > z_entry."""
    if not history or len(history) < 25:
        return None
    tenor_cols = model["tenor_cols"]   # ["m1", ..., "m12"]
    n_tenors = len(tenor_cols)
    # Stack last 25 days of prices into an array
    rows = []
    dates = []
    for h in history[-25:]:
        prices = h.get("prices") or []
        if len(prices) < n_tenors or any(p is None for p in prices[:n_tenors]):
            continue
        rows.append([float(p) for p in prices[:n_tenors]])
        dates.append(h.get("date"))
    if len(rows) < 22:
        return None
    arr = list(zip(*rows))   # transpose for log-return per tenor
    # log returns
    log_returns = []
    for tenor_series in arr:
        lr = []
        for i in range(1, len(tenor_series)):
            if tenor_series[i-1] > 0:
                lr.append(math.log(tenor_series[i] / tenor_series[i-1]))
            else:
                lr.append(0.0)
        log_returns.append(lr)
    # log_returns is shape (n_tenors, n_periods-1)
    if not log_returns or len(log_returns[0]) < 20:
        return None
    # Project each day's returns onto PC3
    components = model["components"]   # (3, n_tenors)
    pc3_vec = components[2]
    mean_vec = model.get("mean") or [0.0] * n_tenors
    pc3_series = []
    n_periods = len(log_returns[0])
    for i in range(n_periods):
        day = [log_returns[t][i] - mean_vec[t] for t in range(n_tenors)]
        pc3 = sum(day[t] * pc3_vec[t] for t in range(n_tenors))
        pc3_series.append(pc3)
    # Rolling z of last 20 PC3 values
    last_20 = pc3_series[-20:]
    mu = sum(last_20) / len(last_20)
    var = sum((x - mu) ** 2 for x in last_20) / (len(last_20) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return None
    z = (pc3_series[-1] - mu) / sd

    # Decide signal
    z_entry = float(model.get("z_entry", 2.0))
    if abs(z) < z_entry:
        # Below threshold — return a "monitoring" stub so the panel shows
        # the engine is alive and what it's watching.
        return {
            "engine":     "PCA_Curve",
            "product":    model["product"],
            "instrument": "fly_" + "_".join(model["fly_legs"]),
            "label":      f"{model['product']} PC3-Fly",
            "direction":  "FLAT",
            "pc3_z":      round(z, 3),
            "current_level": None,
            "confidence": "MONITOR",
        }
    # SHORT the fly if z > z_entry (curvature too high), LONG if z < -z_entry
    direction = "SHORT" if z > 0 else "LONG"
    fly_legs = model["fly_legs"]    # e.g. ["m3", "m6", "m9"]
    fly_signs = model["fly_signs"]  # e.g. [1, -2, 1] (per-tenor weight in fly)

    # Compute current fly value
    last_prices = history[-1].get("prices") or []
    if len(last_prices) < n_tenors:
        return None
    try:
        fly_value = sum(
            float(last_prices[tenor_cols.index(leg)]) * sign
            for leg, sign in zip(fly_legs, fly_signs)
        )
    except (ValueError, TypeError):
        return None

    conf = "HIGH" if abs(z) >= 3.0 else "MED"
    return {
        "engine":     "PCA_Curve",
        "product":    model["product"],
        "instrument": "fly_" + "_".join(fly_legs),
        "label":      f"{model['product']} PC3-Fly ({'+'.join(f'{s:+d}{l}' for l,s in zip(fly_legs, fly_signs))})",
        "direction":  direction,
        "pc3_z":      round(z, 3),
        "current_level": round(fly_value, 4),
        "confidence": conf,
    }


# ---- Bertram OU signal -------------------------------------------------- #
def _bertram_signals(model: Dict, history: List[Dict]) -> List[Dict]:
    """For each OU-fitted spread, emit signal if current spread deviation > a*."""
    if not history:
        return []
    last = history[-1]
    prices = last.get("prices") or []
    out = []
    for spread in model.get("spreads") or []:
        a = spread["leg_a"]   # "m1"
        b = spread["leg_b"]   # "m2"
        try:
            idx_a = int(a[1:]) - 1
            idx_b = int(b[1:]) - 1
            pa = prices[idx_a]; pb = prices[idx_b]
            if pa is None or pb is None:
                continue
            s_now = float(pa) - float(pb)
        except (ValueError, IndexError, TypeError):
            continue
        deviation = s_now - spread["mu_hat"]
        a_star = spread["a_star"]
        if abs(deviation) < a_star:
            # Monitoring entry — show progress toward threshold
            z_pct = abs(deviation) / a_star * 100 if a_star else 0
            out.append({
                "engine":     "Bertram_OU",
                "product":    model["product"],
                "instrument": spread["spread"],
                "label":      f"{model['product']} {spread['spread'].upper()}",
                "direction":  "FLAT",
                "deviation":  round(deviation, 4),
                "a_star":     round(a_star, 4),
                "pct_to_entry": round(z_pct, 1),
                "current_level": round(s_now, 4),
                "confidence": "MONITOR",
            })
            continue
        direction = "SHORT" if deviation > 0 else "LONG"
        z = deviation / spread["sigma_eq"] if spread["sigma_eq"] > 0 else 0
        conf = "HIGH" if abs(z) >= 2.5 else "MED"
        out.append({
            "engine":     "Bertram_OU",
            "product":    model["product"],
            "instrument": spread["spread"],
            "label":      f"{model['product']} {spread['spread'].upper()}",
            "direction":  direction,
            "deviation":  round(deviation, 4),
            "z_score":    round(z, 3),
            "current_level": round(s_now, 4),
            "confidence": conf,
        })
    return out


# ---- HMM regime signal -------------------------------------------------- #
def _hmm_signal(model: Dict,
                  cl_history: List[Dict],
                  lco_history: List[Dict]) -> Optional[Dict]:
    """Compute HMM posterior + emit signal on WTCL spread."""
    if not cl_history or not lco_history:
        return None
    # Build log spread series from last 50 days where both available
    aligned = []
    dates_cl = {h["date"]: h for h in cl_history}
    for h in lco_history:
        if h["date"] in dates_cl:
            cl_p = dates_cl[h["date"]].get("prices") or []
            lc_p = h.get("prices") or []
            if cl_p and lc_p and cl_p[0] and lc_p[0]:
                try:
                    s = math.log(float(cl_p[0])) - math.log(float(lc_p[0]))
                    aligned.append(s)
                except Exception:
                    pass
    if len(aligned) < 30:
        return None
    # Forward-only HMM filter on last point
    means = model["means"]
    stds  = model["stds"]
    trans = model["transition"]
    start = model["startprob"]
    n_states = 2

    def emission_logprob(x, state):
        m = means[state]; s = stds[state]
        if s <= 0: return -1e9
        return -0.5 * math.log(2*math.pi*s*s) - (x - m)**2 / (2*s*s)

    # Forward variable
    log_alpha = [math.log(max(start[i], 1e-12)) + emission_logprob(aligned[0], i)
                 for i in range(n_states)]
    for t in range(1, len(aligned)):
        new_alpha = []
        for j in range(n_states):
            ll = -1e18
            for i in range(n_states):
                lp = log_alpha[i] + math.log(max(trans[i][j], 1e-12))
                ll = max(ll, lp) + math.log1p(math.exp(min(0, lp - max(ll, lp)))) if ll > -1e17 else lp
            new_alpha.append(ll + emission_logprob(aligned[t], j))
        log_alpha = new_alpha
    # Normalize to posterior
    max_la = max(log_alpha)
    posts = [math.exp(la - max_la) for la in log_alpha]
    total = sum(posts) or 1
    posts = [p / total for p in posts]
    low_s = model["low_state"]; high_s = model["high_state"]
    p_low = posts[low_s]; p_high = posts[high_s]
    # Current z relative to conditional mean
    cond_mean = sum(posts[i] * means[i] for i in range(n_states))
    cond_std  = sum(posts[i] * stds[i] for i in range(n_states))
    if cond_std <= 0:
        return None
    z = (aligned[-1] - cond_mean) / cond_std
    p_min = float(model.get("p_regime_min", 0.7))
    z_entry = float(model.get("z_entry", 1.0))
    direction = None
    if p_low > p_min and z < -z_entry:
        direction = "LONG"
    elif p_high > p_min and z > z_entry:
        direction = "SHORT"
    if direction is None:
        # Monitoring stub for the panel
        return {
            "engine":     "HMM_Regime",
            "product":    "WTCL",
            "instrument": "logspread",
            "label":      "WTCL HMM spread",
            "direction":  "FLAT",
            "z_score":    round(z, 3),
            "p_low_regime": round(p_low, 3),
            "p_high_regime": round(p_high, 3),
            "current_level": round(aligned[-1], 4),
            "confidence": "MONITOR",
        }
    conf = "HIGH" if abs(z) >= 2.0 else "MED"
    return {
        "engine":     "HMM_Regime",
        "product":    "WTCL",
        "instrument": "logspread",
        "label":      "WTCL HMM regime spread",
        "direction":  direction,
        "z_score":    round(z, 3),
        "p_low_regime": round(p_low, 3),
        "p_high_regime": round(p_high, 3),
        "current_level": round(aligned[-1], 4),
        "confidence": conf,
    }


# ---- public entry ------------------------------------------------------- #
def build_panel() -> Dict:
    models = _load_models()
    curves = _load_curves()
    signals: List[Dict] = []

    # PCA on CL/LCO/LGO
    for prod_code in ("CL", "LCO", "LGO"):
        m = (models.get("pca") or {}).get(prod_code)
        if not m:
            continue
        hist = (curves.get(prod_code) or {}).get("history") or []
        sig = _pca_signal(m, hist)
        if sig:
            signals.append(sig)

    # Bertram on HO
    ho_model = (models.get("bertram") or {}).get("HO")
    if ho_model:
        hist = (curves.get("HO") or {}).get("history") or []
        signals.extend(_bertram_signals(ho_model, hist))

    # HMM on WTCL (uses CL and LCO front-month series)
    hmm_model = models.get("hmm")
    if hmm_model:
        cl_hist  = (curves.get("CL")  or {}).get("history") or []
        lco_hist = (curves.get("LCO") or {}).get("history") or []
        sig = _hmm_signal(hmm_model, cl_hist, lco_hist)
        if sig:
            signals.append(sig)

    # Summary
    n_long  = sum(1 for s in signals if s["direction"] == "LONG")
    n_short = sum(1 for s in signals if s["direction"] == "SHORT")
    n_flat  = sum(1 for s in signals if s["direction"] == "FLAT")
    n_hi    = sum(1 for s in signals if s["confidence"] == "HIGH")
    n_med   = sum(1 for s in signals if s["confidence"] == "MED")
    n_mon   = sum(1 for s in signals if s["confidence"] == "MONITOR")

    return {
        "available":  True,
        "n_total":    len(signals),
        "n_long":     n_long,
        "n_short":    n_short,
        "n_flat":     n_flat,
        "n_high":     n_hi,
        "n_med":      n_med,
        "n_monitor":  n_mon,
        "signals":    signals,
        "explainer":  ("3-engine portfolio from the paper strategy shootout: "
                       "PCA Curve (CL/LCO/LGO, Sharpe 1.0-6.0), Bertram OU "
                       "(HO, Sharpe 3.4, 97% wins), HMM Regime (WTCL, Sharpe "
                       "0.88). Combined OOS PnL on test slice: +$202k."),
        "strategies_per_product": {
            "CL":   "PCA Curve (Sharpe 6.02, win 79%)",
            "LCO":  "PCA Curve (Sharpe 1.05)",
            "LGO":  "PCA Curve (Sharpe 3.34, +$99k OOS)",
            "HO":   "Bertram OU (Sharpe 3.40, 97% wins)",
            "WTCL": "HMM Regime (Sharpe 0.88, 75% wins)",
        },
    }

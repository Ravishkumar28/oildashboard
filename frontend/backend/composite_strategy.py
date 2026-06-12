"""Composite multi-factor strategy.

Fuses signals from every working engine on the dashboard into ONE per-product
verdict, weighted by each engine's track record:

    Real-data 60/20/20 LGBM   25%   highest weight: trained on user's
                                    actual xlsx curves, R^2 validated
    Live LGBM (5yr yfinance)  20%   solid out-of-sample Sharpe history
    Technical signals (P3F)   15%   BB/RSI/EMA/momentum aligned
    News-aware (P5, fixed)    10%   FinBERT + shock keywords
    Regime butterfly          10%   curve-shape conditioning
    Macro factors              10%   DXY direction + vol regime
    Regression engine (P3B)   10%   lowest: mean-reversion engine,
                                    lost money before risk controls

Each engine contributes a vote in {-1, 0, +1} per product. The composite
score = sum(weight * vote). Final verdict:

    LONG   if composite >= +0.30  (med),  >= +0.50  (high conviction)
    SHORT  if composite <= -0.30  (med),  <= -0.50  (high conviction)
    FLAT   otherwise

Products scored:
    wti, brent, wtcl_spread, rbob, heat, natgas

This module is PURELY a fusion layer — it doesn't call any models itself,
it just consumes the precomputed engine outputs that are already in the
snapshot payload before it's serialized.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------- #
# Weights are designed so the strongest engines (real-data + live LGBM,
# both validated on real data) account for nearly half the vote. The
# weakest (P3B mean-reversion) is capped at 10% so it can't drive
# decisions on its own.
WEIGHTS = {
    "real_data":  0.25,
    "live_lgbm":  0.20,
    "technical":  0.15,
    "news":       0.10,
    "regime":     0.10,
    "macro":      0.10,
    "regression": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.00) < 1e-9, "WEIGHTS must sum to 1.0"

# Map composite product key -> (yfinance proxy used by live LGBM,
#                                xlsx product code used by real-data engine,
#                                technical product key, news product key)
PRODUCT_KEYS = {
    "wti":          {"live": "WTI",       "real": "CL",   "tech": "wti",    "news": "wti"},
    "brent":        {"live": "Brent",     "real": "LCO",  "tech": "brent",  "news": "brent"},
    "wtcl_spread":  {"live": "WTI-Brent", "real": "WTCL", "tech": None,     "news": "wtcl"},
    "rbob":         {"live": "RBOB",      "real": None,   "tech": "rbob",   "news": "rbob"},
    "heat":         {"live": "HO",        "real": "HO",   "tech": "heat",   "news": "heat"},
    "natgas":       {"live": "NatGas",    "real": None,   "tech": "natgas", "news": "natgas"},
}
PRODUCT_NAMES = {
    "wti": "WTI Crude", "brent": "Brent Crude",
    "wtcl_spread": "WTI-Brent Spread", "rbob": "RBOB Gasoline",
    "heat": "Heating Oil", "natgas": "Natural Gas",
}


def _signal_to_vote(signal: Optional[str]) -> int:
    """LONG -> +1, SHORT -> -1, FLAT/HOLD/None -> 0."""
    if signal is None:
        return 0
    s = str(signal).upper()
    if s in ("LONG", "BUY"):
        return 1
    if s in ("SHORT", "SELL"):
        return -1
    return 0


# ---- per-engine vote extraction ---------------------------------------- #
def _live_lgbm_vote(panel: Dict, live_key: Optional[str]) -> Dict[str, Any]:
    if not panel or not panel.get("available") or live_key is None:
        return {"vote": 0, "detail": "n/a"}
    for s in panel.get("signals") or []:
        if s.get("label") == live_key:
            sig = s.get("signal")
            conf = s.get("confidence", "")
            vote = _signal_to_vote(sig)
            # Down-weight LOW confidence signals to half a vote.
            if conf == "LOW":
                vote = vote * 0.5
            return {"vote": vote,
                    "detail": f"{sig or '—'} ({conf or '—'})"}
    return {"vote": 0, "detail": "no match"}


def _real_data_vote(panel: Dict, real_code: Optional[str]) -> Dict[str, Any]:
    """For real-data, aggregate the OUTRIGHT M1 signal AND the strongest
    spread/fly signal on the same product (averaged to one vote)."""
    if not panel or not panel.get("available") or real_code is None:
        return {"vote": 0, "detail": "n/a"}
    outright_vote = 0
    spread_votes: List[int] = []
    best_r2 = 0.0
    best_label = "—"
    best_sig = "FLAT"
    for s in panel.get("signals") or []:
        if s.get("product") != real_code:
            continue
        if s.get("confidence") in ("NOISE",):
            continue
        sig = s.get("signal")
        vote = _signal_to_vote(sig)
        if s.get("kind") == "OUTRIGHT" and s.get("tenor") == "m1":
            outright_vote = vote
        elif s.get("kind") in ("SPREAD", "FLY") and vote != 0:
            spread_votes.append(vote)
        r2 = s.get("test_r2") or 0
        if r2 > best_r2 and sig != "FLAT":
            best_r2 = r2
            best_label = s.get("label", "?")
            best_sig = sig
    spread_avg = (sum(spread_votes) / len(spread_votes)) if spread_votes else 0
    composite = 0.5 * outright_vote + 0.5 * spread_avg
    # Clamp to [-1, +1]
    composite = max(-1.0, min(1.0, composite))
    return {"vote": composite,
            "detail": f"top={best_label}:{best_sig} R²={best_r2:.2f}" if best_r2 > 0 else "no strong target"}


def _technical_vote(panel: Dict, tech_key: Optional[str]) -> Dict[str, Any]:
    if not panel or not panel.get("available") or tech_key is None:
        return {"vote": 0, "detail": "n/a"}
    for p in panel.get("products") or []:
        if p.get("product") == tech_key:
            d = p.get("direction") or p.get("verdict")
            raw = p.get("raw_score", 0) or 0
            vote = _signal_to_vote(d)
            return {"vote": vote,
                    "detail": f"{d or '—'} raw={raw:+d}/±4"}
    return {"vote": 0, "detail": "no match"}


def _news_vote(panel: Dict, news_key: Optional[str]) -> Dict[str, Any]:
    if not panel or not panel.get("available") or news_key is None:
        return {"vote": 0, "detail": "n/a"}
    for p in panel.get("products") or []:
        if p.get("product") == news_key:
            d = p.get("direction")
            conv = p.get("conviction", "")
            vote = _signal_to_vote(d)
            if conv == "low":
                vote = vote * 0.5
            return {"vote": vote, "detail": f"{d or '—'} ({conv or '—'})"}
    return {"vote": 0, "detail": "no match"}


def _regime_vote(regime_panel: Dict, key: str) -> Dict[str, Any]:
    """Regime-butterfly only directly applies to WTI fly. For other
    products, we use the curve-slope sign as a coarse regime read:
    backwardation -> bullish front-end (LONG outright bias for crude),
    contango -> bearish (SHORT bias for crude)."""
    if not regime_panel:
        return {"vote": 0, "detail": "n/a"}
    slope = regime_panel.get("slope")
    if slope is None:
        return {"vote": 0, "detail": "no regime data"}
    # For crude/products: backwardation (slope < 0) is bullish for outrights,
    # bearish for spreads (spreads tend to compress).
    if key in ("wti", "brent", "rbob", "heat"):
        if slope <= -2:
            return {"vote": +1, "detail": f"steep back (slope {slope:+.1f})"}
        if slope >= +2:
            return {"vote": -1, "detail": f"contango (slope {slope:+.1f})"}
        return {"vote": 0, "detail": f"flat (slope {slope:+.1f})"}
    if key == "wtcl_spread":
        # WTI-Brent spread tends to converge in stress (steep back) -> LONG spread
        if slope <= -2:
            return {"vote": +1, "detail": f"stress regime -> spread tightens"}
        if slope >= +2:
            return {"vote": -1, "detail": f"calm regime -> spread loose"}
        return {"vote": 0, "detail": f"neutral regime"}
    if key == "natgas":
        # NG is decoupled from crude regime
        return {"vote": 0, "detail": "n/a (decoupled)"}
    return {"vote": 0, "detail": "—"}


def _macro_vote(market, key: str) -> Dict[str, Any]:
    """DXY direction (bullish $ = bearish oil) + vol regime.
       DXY 20d return > +2%  -> -1 vote on crude/products
       DXY 20d return < -2%  -> +1 vote on crude/products
       NG decoupled."""
    if key == "natgas":
        return {"vote": 0, "detail": "n/a"}
    if not market:
        return {"vote": 0, "detail": "no market"}
    dxy_hist = []
    try:
        hist = getattr(market, "hist", {}) or {}
        dxy_hist = hist.get("dxy") or []
    except Exception:
        return {"vote": 0, "detail": "no dxy"}
    if len(dxy_hist) < 22:
        return {"vote": 0, "detail": "dxy<22d"}
    cur = float(dxy_hist[-1])
    past = float(dxy_hist[-21])
    if past <= 0:
        return {"vote": 0, "detail": "—"}
    dxy_ret_20d = (cur - past) / past * 100.0
    if dxy_ret_20d >= 2.0:
        return {"vote": -1, "detail": f"DXY +{dxy_ret_20d:.1f}% 20d -> $ strong, oil bearish"}
    if dxy_ret_20d <= -2.0:
        return {"vote": +1, "detail": f"DXY {dxy_ret_20d:+.1f}% 20d -> $ weak, oil bullish"}
    return {"vote": 0, "detail": f"DXY {dxy_ret_20d:+.1f}% 20d -> neutral"}


def _regression_vote(term_structure: Dict, key: str) -> Dict[str, Any]:
    """Pick the strongest term-structure idea matching this product."""
    if not term_structure or not term_structure.get("available"):
        return {"vote": 0, "detail": "n/a"}
    code = {"wti": "CL", "brent": "LCO3", "rbob": None, "heat": "HO",
            "natgas": None, "wtcl_spread": "WTCL"}.get(key)
    if code is None:
        return {"vote": 0, "detail": "n/a"}
    top = term_structure.get("cross_product_top") or []
    best = None
    for idea in top:
        if idea.get("product") != code:
            continue
        if best is None or abs(idea.get("z", 0)) > abs(best.get("z", 0)):
            best = idea
    if best is None:
        return {"vote": 0, "detail": "no idea"}
    vote = _signal_to_vote(best.get("direction"))
    return {"vote": vote,
            "detail": f"{best.get('label','')} z={best.get('z',0):+.2f}"}


# ---- main fusion ------------------------------------------------------- #
def _verdict_and_conf(score: float) -> tuple:
    if score >= 0.50:   return "LONG",  "HIGH"
    if score >= 0.30:   return "LONG",  "MED"
    if score <= -0.50:  return "SHORT", "HIGH"
    if score <= -0.30:  return "SHORT", "MED"
    return "FLAT", "LOW"


def build_panel(
    market,
    live_signals: Optional[Dict] = None,
    real_data_panel: Optional[Dict] = None,
    technical: Optional[Dict] = None,
    news_signals_panel: Optional[Dict] = None,
    regime_butterfly: Optional[Dict] = None,
    term_structure: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build the composite panel by fusing per-engine votes."""
    products_out: List[Dict[str, Any]] = []
    n_long = n_short = n_flat = 0

    for key, keymap in PRODUCT_KEYS.items():
        factors: Dict[str, Dict[str, Any]] = {}
        factors["real_data"]  = _real_data_vote(real_data_panel,  keymap.get("real"))
        factors["live_lgbm"]  = _live_lgbm_vote(live_signals,     keymap.get("live"))
        factors["technical"]  = _technical_vote(technical,        keymap.get("tech"))
        factors["news"]       = _news_vote(news_signals_panel,    keymap.get("news"))
        factors["regime"]     = _regime_vote(regime_butterfly,    key)
        factors["macro"]      = _macro_vote(market,               key)
        factors["regression"] = _regression_vote(term_structure,  key)

        score = sum(WEIGHTS[k] * factors[k]["vote"] for k in WEIGHTS)
        score = max(-1.0, min(1.0, score))
        verdict, conf = _verdict_and_conf(score)
        if verdict == "LONG":  n_long  += 1
        elif verdict == "SHORT": n_short += 1
        else:                    n_flat  += 1

        products_out.append({
            "product":     key,
            "name":        PRODUCT_NAMES[key],
            "composite_score": round(float(score), 3),
            "verdict":     verdict,
            "conviction":  conf,
            "factors":     [
                {"engine": k, "weight": WEIGHTS[k],
                 "vote":   round(factors[k]["vote"], 2),
                 "detail": factors[k]["detail"]}
                for k in WEIGHTS
            ],
        })

    products_out.sort(key=lambda p: (
        0 if p["verdict"] != "FLAT" else 1,
        {"HIGH": 0, "MED": 1, "LOW": 2}.get(p["conviction"], 9),
        -abs(p["composite_score"]),
    ))

    return {
        "available":   True,
        "weights":     WEIGHTS,
        "products":    products_out,
        "n_long":      n_long,
        "n_short":     n_short,
        "n_flat":      n_flat,
        "explainer":   ("Per-engine vote fused into a composite score in "
                        "[-1, +1]. Verdict = LONG/SHORT/FLAT by score "
                        "threshold; conviction = HIGH if |score| ≥ 0.50, "
                        "MED if ≥ 0.30."),
    }

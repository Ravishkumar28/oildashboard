"""Composite bullish / bearish / neutral signal from all dashboard inputs.

Aggregates real-time signals across three buckets — TECHNICAL (BB, curve
structure, crack z-scores, active strategy signals), FUNDAMENTAL (EIA
stocks/OPEC/refinery, STEO forward balance, COT positioning, FRED PMI),
and NEWS (VADER-aggregated headlines) — into a single composite score
in [-100, +100] with a 5-tier label.

Each bucket contributes scaled points based on extremity. The output
includes the per-driver breakdown so the user can see WHY the score is
where it is, not just the headline number.

This is a heuristic aggregator, not a backtested alpha model. It's a
"dashboard at a glance" gauge — useful for situational awareness, not
for sizing positions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _technical_score(snapshot: Dict) -> Tuple[float, List[Dict]]:
    """Bull/bear contributions from price-action + curve structure."""
    score = 0.0
    drivers: List[Dict] = []

    # Bollinger Bands position — extreme reads = mean-reversion setup
    bb = snapshot.get("bb") or {}
    bb_pos = _safe_float(bb.get("position"))
    if bb_pos is not None:
        if bb_pos < 10:
            score += 12
            drivers.append({"label": "BB position deeply oversold (<10%)",
                            "contribution": +12, "value": f"{bb_pos:.0f}%"})
        elif bb_pos < 25:
            score += 6
            drivers.append({"label": "BB position oversold (<25%)",
                            "contribution": +6, "value": f"{bb_pos:.0f}%"})
        elif bb_pos > 90:
            score -= 12
            drivers.append({"label": "BB position deeply overbought (>90%)",
                            "contribution": -12, "value": f"{bb_pos:.0f}%"})
        elif bb_pos > 75:
            score -= 6
            drivers.append({"label": "BB position overbought (>75%)",
                            "contribution": -6, "value": f"{bb_pos:.0f}%"})

    # Futures curve structure
    futures = snapshot.get("futures") or {}
    structure = futures.get("structure")
    m12 = _safe_float(futures.get("m12_spread"))
    if structure == "Backwardation":
        contrib = +10
        score += contrib
        drivers.append({
            "label": "Curve in backwardation — tight physical market",
            "contribution": contrib,
            "value": f"M12 vs spot: ${m12:+.2f}" if m12 is not None else None,
        })
    elif structure == "Contango":
        contrib = -6
        score += contrib
        drivers.append({
            "label": "Curve in contango — oversupplied",
            "contribution": contrib,
            "value": f"M12 vs spot: ${m12:+.2f}" if m12 is not None else None,
        })

    # 3-2-1 crack z-score — refining margin extremes
    for crack in snapshot.get("cracks", []) or []:
        if "3-2-1" in str(crack.get("name", "")):
            z = _safe_float(crack.get("zscore"))
            if z is not None:
                if z < -1.5:
                    score += 8
                    drivers.append({
                        "label": "3-2-1 crack deeply cheap (mean-revert up)",
                        "contribution": +8,
                        "value": f"{z:+.2f}σ"})
                elif z > 1.5:
                    score -= 8
                    drivers.append({
                        "label": "3-2-1 crack deeply rich (margin compression risk)",
                        "contribution": -8,
                        "value": f"{z:+.2f}σ"})
            break

    # 5-year same-week range — at lows = secular buy signal
    fy = snapshot.get("fiveyear") or {}
    if fy.get("buy_signal"):
        score += 10
        drivers.append({
            "label": "Price at 5-year same-week low — historical buy signal",
            "contribution": +10,
            "value": f"${_safe_float(fy.get('current')) or 0:.2f}"})

    # ACTIVE strategy signals — small weight, direction-summed
    sig_long = sig_short = 0
    for s in snapshot.get("signals", []) or []:
        if s.get("status") == "ACTIVE":
            direction = (s.get("direction") or "").upper()
            if "LONG" in direction:
                sig_long += 1
            elif "SHORT" in direction:
                sig_short += 1
    net_sig = sig_long - sig_short
    if net_sig:
        score += 3 * net_sig
        drivers.append({
            "label": f"Strategy signals net: {sig_long} long − {sig_short} short",
            "contribution": +3 * net_sig,
            "value": f"{net_sig:+d}"})

    return score, drivers


def _fundamental_score(snapshot: Dict) -> Tuple[float, List[Dict]]:
    """Bull/bear contributions from EIA + STEO + COT + PMI."""
    score = 0.0
    drivers: List[Dict] = []

    # EIA fundamentals — direction of weekly inventories
    for f in snapshot.get("fundamentals", []) or []:
        label = f.get("label", "")
        trend = f.get("trend")
        if label == "US Crude Storage":
            if trend == "down":
                score += 10
                drivers.append({"label": "US crude stocks drawing",
                                "contribution": +10, "value": f.get("value")})
            elif trend == "up":
                score -= 10
                drivers.append({"label": "US crude stocks building",
                                "contribution": -10, "value": f.get("value")})
        elif label == "Cushing Inventory":
            if trend == "down":
                score += 5
                drivers.append({"label": "Cushing drawing — WTI delivery hub",
                                "contribution": +5, "value": f.get("value")})
            elif trend == "up":
                score -= 5
                drivers.append({"label": "Cushing building",
                                "contribution": -5, "value": f.get("value")})
        elif label == "Refinery Utilization":
            val = _safe_float(f.get("value"))
            if val is not None:
                if val > 92:
                    score += 6
                    drivers.append({"label": "Refinery utilization high — demand strong",
                                    "contribution": +6, "value": f"{val:.1f}%"})
                elif val < 85:
                    score -= 6
                    drivers.append({"label": "Refinery utilization low — demand soft",
                                    "contribution": -6, "value": f"{val:.1f}%"})
        elif label == "OPEC Production":
            # delta vs quota: negative = under quota (compliance/bullish);
            # positive = over quota (bearish)
            delta_str = str(f.get("delta", ""))
            try:
                delta = float(delta_str.replace("+", ""))
            except ValueError:
                delta = 0.0
            if delta > 0.3:
                score -= 7
                drivers.append({"label": "OPEC over quota",
                                "contribution": -7,
                                "value": f"{delta:+.2f} vs quota"})
            elif delta < -0.3:
                score += 7
                drivers.append({"label": "OPEC under quota — compliance",
                                "contribution": +7,
                                "value": f"{delta:+.2f} vs quota"})

    # STEO forward balance — 12-month deficit/surplus
    steo = snapshot.get("steo") or {}
    fwd12 = _safe_float(steo.get("fwd12_avg_balance"))
    if fwd12 is not None:
        if fwd12 < -1.0:
            score += 15
            drivers.append({
                "label": "STEO 12M forward balance deficit — tightening",
                "contribution": +15, "value": f"{fwd12:+.2f} Mbpd"})
        elif fwd12 < -0.3:
            score += 7
            drivers.append({
                "label": "STEO 12M forward balance slight deficit",
                "contribution": +7, "value": f"{fwd12:+.2f} Mbpd"})
        elif fwd12 > 1.0:
            score -= 15
            drivers.append({
                "label": "STEO 12M forward balance surplus — loosening",
                "contribution": -15, "value": f"{fwd12:+.2f} Mbpd"})
        elif fwd12 > 0.3:
            score -= 7
            drivers.append({
                "label": "STEO 12M forward balance slight surplus",
                "contribution": -7, "value": f"{fwd12:+.2f} Mbpd"})

    # CFTC COT — managed money positioning
    cot = snapshot.get("cot") or {}
    if cot and cot.get("categories"):
        for cat in cot["categories"]:
            if "Managed Money" in str(cat.get("label", "")):
                net = _safe_float(cat.get("net"))
                wow = _safe_float(cat.get("net_change"))
                if net is not None and wow is not None:
                    # Momentum: net positioning growing → trend persists
                    if wow > 20000:
                        score += 6
                        drivers.append({
                            "label": "Managed Money adding longs (momentum)",
                            "contribution": +6,
                            "value": f"WoW +{int(wow/1000)}k"})
                    elif wow < -20000:
                        score -= 6
                        drivers.append({
                            "label": "Managed Money cutting longs (momentum)",
                            "contribution": -6,
                            "value": f"WoW {int(wow/1000)}k"})
                break

    # FRED Manufacturing composite — PMI proxy
    mfg = snapshot.get("manufacturing") or {}
    comp = (mfg.get("composite_proxy") or {})
    state = comp.get("state")
    val = _safe_float(comp.get("value"))
    if state == "expansion" and val is not None:
        contrib = +8 if val >= 55 else +4
        score += contrib
        drivers.append({
            "label": f"US manufacturing expansion (PMI proxy)",
            "contribution": contrib, "value": f"{val:.1f}"})
    elif state == "contraction" and val is not None:
        contrib = -8 if val <= 45 else -4
        score += contrib
        drivers.append({
            "label": f"US manufacturing contraction (PMI proxy)",
            "contribution": contrib, "value": f"{val:.1f}"})

    return score, drivers


def _news_score(snapshot: Dict) -> Tuple[float, List[Dict]]:
    """Bull/bear contribution averaged across per-item FinBERT scores.

    FinBERT (ProsusAI/finbert) is the finance-tuned transformer classifier
    — much better than VADER at "production cut = bullish for crude" type
    semantics. Only items where FinBERT actually classified them (score
    != None) contribute; if the model hasn't finished loading yet, we fall
    back to the VADER-aggregated news_sentiment with an honest label so
    the panel never silently uses the wrong signal.
    """
    score = 0.0
    drivers: List[Dict] = []

    news_items = snapshot.get("news") or []
    finbert_scores = [_safe_float(it.get("finbert_score"))
                      for it in news_items
                      if it.get("finbert_score") is not None]
    finbert_scores = [s for s in finbert_scores if s is not None]

    if finbert_scores:
        n = len(finbert_scores)
        avg = sum(finbert_scores) / n
        if avg >= 0.15:
            label = "bullish"
        elif avg <= -0.15:
            label = "bearish"
        else:
            label = "neutral"
        # Scale to ±25 max contribution — news is meaningful but shouldn't
        # dominate the technical/fundamental signal stack.
        scaled = max(-25.0, min(25.0, avg * 40.0))
        score += scaled
        drivers.append({
            "label": f"FinBERT news mood: {label} ({n} headlines)",
            "contribution": round(scaled, 1),
            "value": f"{avg:+.2f}"})
        return score, drivers

    # FinBERT not loaded / no items scored — fall back to VADER but label it
    ns = snapshot.get("news_sentiment") or {}
    compound = _safe_float(ns.get("compound"))
    count = ns.get("count", 0)
    if compound is not None and count > 0:
        scaled = max(-25.0, min(25.0, compound * 40.0))
        score += scaled
        drivers.append({
            "label": f"VADER fallback (FinBERT not loaded): {ns.get('label','neutral')} ({count} headlines)",
            "contribution": round(scaled, 1),
            "value": f"{compound:+.2f}"})
    return score, drivers


def _label_and_color(score: float) -> Tuple[str, str]:
    if score >= 50:
        return "Strongly Bullish", "strong-bull"
    if score >= 20:
        return "Bullish", "bull"
    if score > -20:
        return "Neutral", "neutral"
    if score > -50:
        return "Bearish", "bear"
    return "Strongly Bearish", "strong-bear"


def compute(snapshot: Dict) -> Dict:
    """Build the composite signal from a full dashboard snapshot.

    Returns:
        {
          score:        float in [-100, +100],
          label:        "Strongly Bullish" | "Bullish" | ...,
          color:        tag for frontend CSS class,
          buckets: {
            technical:    {score, drivers: [...]},
            fundamental:  {score, drivers: [...]},
            news:         {score, drivers: [...]},
          },
        }
    """
    tech, tech_drivers = _technical_score(snapshot)
    fund, fund_drivers = _fundamental_score(snapshot)
    news, news_drivers = _news_score(snapshot)

    total = tech + fund + news
    total = max(-100.0, min(100.0, total))

    label, color = _label_and_color(total)

    return {
        "score": round(total, 1),
        "label": label,
        "color": color,
        "buckets": {
            "technical":   {"score": round(tech, 1), "drivers": tech_drivers},
            "fundamental": {"score": round(fund, 1), "drivers": fund_drivers},
            "news":        {"score": round(news, 1), "drivers": news_drivers},
        },
    }

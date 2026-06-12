"""News-driven trade signal engine.

Reads the live news feed, routes each headline to the products it affects
(WTI / Brent / WTCL spread / RBOB / HO / NatGas) using a keyword classifier,
aggregates FinBERT-preferred sentiment, and fuses that with the dashboard's
existing z-score state (BB position, fly z, slope z, RSI) to emit per-product
buy/sell verdicts.

Verdict logic per product:

    news_vote   = +1 if news_score > +0.15 (bullish), -1 if < -0.15, else 0
    zscore_vote = +1 if BB% < 25 (oversold) OR fly_z < -1, etc.
                  -1 for the inverse, else 0
    combined    = news_vote + zscore_vote in {-2, -1, 0, +1, +2}

    LONG  · high   if combined == +2  -> 5 lots
    LONG  · med    if combined == +1  -> 2 lots
    HOLD           if combined ==  0  -> 0 lots
    SHORT · med    if combined == -1  -> 2 lots
    SHORT · high   if combined == -2  -> 5 lots

Entry / Take Profit / Stop Loss use the same vol-banded formula as P3F
(technical_signals) and P4D (regime signals) for cross-engine comparability:
    σ_$ = |entry| × vol20    TP at +/- 2σ × √5    SL at -/+ 1.5σ    R:R ≈ 2.98:1

The news engine also surfaces:
- Recent high-impact headlines per product (top 3 by |score|)
- Aggregate sentiment counters (bullish / bearish / neutral per product)
- News momentum (z-score of last 6h sentiment vs trailing 7-day baseline)
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence


# --- product routing keywords --------------------------------------------- #
# A headline is matched to a product if it contains ANY of the listed terms.
# Same headline can be routed to multiple products (e.g. "OPEC quota cut"
# hits WTI, Brent and WTCL).

PRODUCT_KEYWORDS = {
    "wti":   ["wti", "us crude", "u.s. crude", "permian", "cushing", "shale",
              "us oil production", "domestic crude", "north dakota", "bakken",
              "midland", "us oil", "american oil"],
    "brent": ["brent", "north sea", "european crude", "saudi", "opec",
              "russian oil", "russia crude", "ural", "espo", "esp opec",
              "middle east oil", "iran", "iraq", "uae crude", "kazakh"],
    "wtcl":  ["wti-brent", "wti brent spread", "transatlantic", "arb",
              "atlantic basin", "us export", "crude export", "arbitrage"],
    "rbob":  ["rbob", "gasoline", "usgc gasoline", "summer driving",
              "gasoline demand", "us gasoline stocks", "refinery turnaround",
              "harbor gasoline", "rfg", "ethanol", "memorial day driving"],
    "heat":  ["heating oil", "ulsd", "distillate", "diesel", "diesel demand",
              "trucking", "winter heating", "northeast heating",
              "distillate stocks", "low sulfur diesel", "gasoil"],
    "natgas": ["natural gas", "lng", "henry hub", "ng futures",
               "european gas", "ttf", "freeport lng", "asian lng",
               "european lng", "ng storage", "weekly gas storage",
               "permian gas", "haynesville"],
}

# Generic / macro keywords route to ALL products (with smaller weight)
MACRO_KEYWORDS = ["fed", "interest rate", "dxy", "dollar", "inflation",
                  "recession", "gdp", "fomc", "central bank", "rate cut",
                  "rate hike", "treasury", "yield", "china growth",
                  "chinese demand", "us demand", "global demand"]

# Geopolitical / supply shock keywords are bullish for crude complex
SUPPLY_SHOCK_KEYWORDS = ["sanctions", "attack", "drone", "missile", "embargo",
                         "blockade", "hormuz", "red sea", "houthi",
                         "pipeline rupture", "outage", "force majeure",
                         "war", "invasion", "strike on", "explosion at",
                         "fire at refinery", "hurricane", "storm shut"]

DEMAND_WEAK_KEYWORDS = ["demand drop", "weak demand", "demand slows",
                       "consumption falls", "lockdown", "recession looms",
                       "manufacturing slumps", "import drop", "covid",
                       "stockpile builds"]


# --- vol-banded entry / TP / SL ------------------------------------------- #
def _daily_vol(prices: Sequence[float]) -> Optional[float]:
    """20-day realized log-return std (daily)."""
    if len(prices) < 22:
        return None
    rets = []
    for i in range(len(prices) - 20, len(prices)):
        if i == 0 or prices[i - 1] <= 0:
            continue
        rets.append(math.log(prices[i] / prices[i - 1]))
    if len(rets) < 5:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / len(rets)
    return var ** 0.5


def _trade_plan(price: float, direction: str,
                sigma_dollar: Optional[float],
                vol20_for_display: Optional[float] = None) -> Dict[str, Any]:
    """Generic vol-banded plan. Caller pre-computes sigma_dollar (1-day $ std).

    For spot prices: sigma_dollar = |price| × vol20  (log-return based)
    For spreads:     sigma_dollar = std of absolute daily change directly
                                    (log-return doesn't work for series that
                                     can be ≈ 0 or negative).
    """
    if direction not in ("LONG", "SHORT") or sigma_dollar is None or price is None:
        return {"entry": price, "tp": None, "sl": None,
                "rr": None, "tp_pct": None, "sl_pct": None}
    tp_dist = 2.0 * sigma_dollar * math.sqrt(5)
    sl_dist = 1.5 * sigma_dollar
    sgn = 1 if direction == "LONG" else -1
    tp = price + sgn * tp_dist
    sl = price - sgn * sl_dist
    abs_price = max(abs(price), 0.01)
    vol_disp = (vol20_for_display if vol20_for_display is not None
                else sigma_dollar / abs_price)
    return {
        "entry":         round(float(price), 4),
        "tp":            round(float(tp), 4),
        "sl":            round(float(sl), 4),
        "tp_pct":        round(tp_dist / abs_price * 100, 2),
        "sl_pct":        round(sl_dist / abs_price * 100, 2),
        "rr":            round(tp_dist / sl_dist, 2),
        "daily_vol_pct": round(vol_disp * 100, 2),
        "horizon_days":  5,
    }


def _spread_dollar_vol(spread_hist: List[float]) -> Optional[float]:
    """Std of absolute daily $ changes of a spread (works for negative values)."""
    if len(spread_hist) < 22:
        return None
    changes = [spread_hist[-i] - spread_hist[-i-1] for i in range(1, 21)]
    if len(changes) < 5:
        return None
    mu = sum(changes) / len(changes)
    var = sum((c - mu) ** 2 for c in changes) / len(changes)
    return var ** 0.5


# --- headline categorisation ---------------------------------------------- #
def _matches_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def _route_headline(item: Dict[str, Any]) -> List[str]:
    """Return list of product keys this headline affects."""
    h = item.get("headline", "")
    products: List[str] = []
    for prod, kws in PRODUCT_KEYWORDS.items():
        if _matches_any(h, kws):
            products.append(prod)
    if _matches_any(h, MACRO_KEYWORDS):
        for prod in ("wti", "brent", "wtcl", "rbob", "heat", "natgas"):
            if prod not in products:
                products.append(prod)
    return products


def _impact_bias(item: Dict[str, Any]) -> int:
    """Supply-shock headlines = +1 (bullish crude). Weak-demand = -1.
    Otherwise 0 — relies on raw sentiment."""
    h = item.get("headline", "")
    if _matches_any(h, SUPPLY_SHOCK_KEYWORDS):
        return +1
    if _matches_any(h, DEMAND_WEAK_KEYWORDS):
        return -1
    return 0


def _item_score(item: Dict[str, Any]) -> float:
    """Best available sentiment score (-1 to +1). Prefer FinBERT
    over VADER."""
    fs = item.get("finbert_score")
    if fs is not None:
        return float(fs)
    return float(item.get("sentiment_score") or 0.0)


# --- per-product aggregation ---------------------------------------------- #
def _per_product_news(news_items: List[Dict[str, Any]],
                      window_hours: float = 24.0) -> Dict[str, Dict[str, Any]]:
    """Group recent news per product and aggregate sentiment."""
    cutoff = time.time() - window_hours * 3600
    by_product: Dict[str, Dict[str, Any]] = {}
    for prod in PRODUCT_KEYWORDS:
        by_product[prod] = {"items": [], "n": 0,
                            "bull": 0, "bear": 0, "neutral": 0,
                            "agg_score": 0.0, "impact_bias": 0,
                            "top_headlines": []}

    for it in news_items:
        if it.get("ts", 0) < cutoff:
            continue
        prods = _route_headline(it)
        if not prods:
            continue
        score = _item_score(it)
        impact = _impact_bias(it)
        for prod in prods:
            slot = by_product[prod]
            slot["items"].append({**it, "_score": score, "_impact": impact})
            slot["n"] += 1
            slot["agg_score"] += score
            slot["impact_bias"] += impact
            if score >= 0.15:
                slot["bull"] += 1
            elif score <= -0.15:
                slot["bear"] += 1
            else:
                slot["neutral"] += 1

    # Average score + pick top 3 headlines by |score|
    for prod, slot in by_product.items():
        n = slot["n"]
        slot["avg_score"] = round(slot["agg_score"] / n, 3) if n else 0.0
        slot["agg_score"] = round(slot["agg_score"], 3)
        items = sorted(slot["items"], key=lambda x: -abs(x.get("_score", 0)))
        slot["top_headlines"] = [
            {"headline": it["headline"][:140],
             "source":   it.get("source", ""),
             "ts":       int(it.get("ts", 0)),
             "score":    round(it["_score"], 3),
             "impact":   it["_impact"]}
            for it in items[:3]
        ]
        del slot["items"]   # don't ship the full set
    return by_product


# --- z-score collection from the live snapshot ---------------------------- #
def _zscore_signals(market,
                    bb_pos: Optional[float],
                    fly_z: Optional[float]) -> Dict[str, Dict[str, Any]]:
    """For each product, extract its OWN BB% and fly z (not the global WTI one).

    Vote:
        +1 (LONG bias)  -> spread/structure looks cheap / oversold
        -1 (SHORT bias) -> rich / overbought
         0              -> mid-band

    The previous version hardcoded WTI's BB% and fly z onto every product,
    producing identical LONG verdicts across the board even when the
    individual products' positions diverged. Now each product computes its
    own 20d BB% from its own price history.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def bb_vote(bb):
        if bb is None: return 0, "—"
        if bb <= 25:   return +1, f"BB% {bb:.0f} oversold"
        if bb >= 75:   return -1, f"BB% {bb:.0f} overbought"
        return 0, f"BB% {bb:.0f} mid-band"

    def fly_vote(z):
        if z is None: return 0, "—"
        if z <= -1.0: return +1, f"fly z={z:+.1f} cheap"
        if z >= +1.0: return -1, f"fly z={z:+.1f} rich"
        return 0, f"fly z={z:+.1f} neutral"

    def per_product_bb(hist_series):
        """20d BB% position from a price history list."""
        if not hist_series or len(hist_series) < 20:
            return None
        tail = list(hist_series[-20:])
        mu = sum(tail) / len(tail)
        sd = (sum((x - mu) ** 2 for x in tail) / len(tail)) ** 0.5
        if sd == 0:
            return None
        last = float(hist_series[-1])
        return max(0.0, min(100.0, (last - (mu - 2 * sd)) / (4 * sd) * 100))

    hist = getattr(market, "hist", {}) or {}

    # WTCL spread: BB% on the (wti - brent) historical series
    wti_h   = hist.get("wti")   or []
    brent_h = hist.get("brent") or []
    wtcl_h: List[float] = []
    if wti_h and brent_h:
        n = min(len(wti_h), len(brent_h))
        wtcl_h = [float(wti_h[-(n - i)]) - float(brent_h[-(n - i)])
                  for i in range(n)]

    # Per-product BB% — each product computes its own.
    # Fly z is WTI-specific (M3-2M6+M9 of the WTI curve), so only WTI inherits
    # the global fly_z; other products get neutral (0) on the fly vote.
    bb_by_product = {
        "wti":    bb_pos if bb_pos is not None else per_product_bb(wti_h),
        "brent":  per_product_bb(brent_h),
        "wtcl":   per_product_bb(wtcl_h) if wtcl_h else None,
        "rbob":   per_product_bb(hist.get("rbob")),
        "heat":   per_product_bb(hist.get("heat") or hist.get("ho")),
        "natgas": per_product_bb(hist.get("natgas")),
    }
    fly_by_product = {
        "wti":    fly_z,
        "brent":  None,   # no per-product fly z available
        "wtcl":   None,
        "rbob":   None,
        "heat":   None,
        "natgas": None,
    }

    for prod in ("wti", "brent", "wtcl", "rbob", "heat", "natgas"):
        bb_v, bb_lbl = bb_vote(bb_by_product[prod])
        fly_v, fly_lbl = fly_vote(fly_by_product[prod])
        combined = max(-1, min(+1, bb_v + fly_v))
        out[prod] = {
            "vote":  combined,
            "label": f"{bb_lbl}" + (f" · {fly_lbl}" if fly_by_product[prod] is not None else ""),
        }
    return out


# --- public entry --------------------------------------------------------- #
PRODUCT_NAMES = {
    "wti":    "WTI Crude",
    "brent":  "Brent Crude",
    "wtcl":   "WTI-Brent Spread",
    "rbob":   "RBOB Gasoline",
    "heat":   "Heating Oil",
    "natgas": "Natural Gas",
}

# Current-price source on the MarketEngine
PRICE_FROM = {
    "wti":    lambda m: getattr(m, "wti", None),
    "brent":  lambda m: getattr(m, "brent", None),
    "wtcl":   lambda m: (getattr(m, "wti", 0.0) - getattr(m, "brent", 0.0))
                        if (m.wti and m.brent) else None,
    "rbob":   lambda m: getattr(m, "rbob", None),
    "heat":   lambda m: getattr(m, "heat", None),
    "natgas": lambda m: (m.hist["natgas"][-1]
                         if m.hist.get("natgas") else None),
}

# Price unit for display
PRICE_UNIT = {
    "wti":    "$/bbl",
    "brent":  "$/bbl",
    "wtcl":   "$/bbl",
    "rbob":   "$/gal",
    "heat":   "$/gal",
    "natgas": "$/MMBtu",
}

HIST_KEYS = {
    "wti":    "wti",
    "brent":  "brent",
    "rbob":   "rbob",
    "heat":   "heat",
    "natgas": "natgas",
}


def build_panel(market, news_items: List[Dict[str, Any]],
                bb_pos: Optional[float] = None,
                fly_z: Optional[float] = None) -> Dict[str, Any]:
    """Produce the news-aware trade signal panel for the snapshot."""
    by_news = _per_product_news(news_items, window_hours=24.0)
    by_z    = _zscore_signals(market, bb_pos, fly_z)

    products_out: List[Dict[str, Any]] = []
    n_long = n_short = n_hold = 0

    for prod in ("wti", "brent", "wtcl", "rbob", "heat", "natgas"):
        news = by_news.get(prod, {})
        zs   = by_z.get(prod, {"vote": 0, "label": "—"})

        # News vote: combine average sentiment + impact bias, with two
        # corrections for the prior "Brent BULLISH on bearish headlines" bug:
        # 1. Per-shock weight reduced from 0.10 to 0.03 (was 5x the sentiment)
        # 2. Impact CANNOT flip clearly directional sentiment — if FinBERT
        #    already read the headline as solidly bearish/bullish, supply-
        #    shock keywords only amplify, never reverse. FinBERT already
        #    accounts for "Hormuz blockade -> prices falling" being bearish;
        #    the keyword bias on top was double-counting.
        avg = news.get("avg_score", 0.0)
        impact = news.get("impact_bias", 0)
        impact_bonus = max(-0.10, min(0.10, 0.03 * impact))   # cap at ±0.10
        if avg <= -0.10 and impact_bonus > 0:
            impact_bonus = 0    # don't let bullish shock flip clear bearish read
        elif avg >= 0.10 and impact_bonus < 0:
            impact_bonus = 0    # don't let bearish bias flip clear bullish read
        news_score = avg + impact_bonus
        if news_score >= 0.15:
            news_vote = +1
            news_lbl = (f"BULLISH avg {avg:+.2f}"
                        + (f", +{impact} shock" if impact > 0 else ""))
        elif news_score <= -0.15:
            news_vote = -1
            news_lbl = (f"BEARISH avg {avg:+.2f}"
                        + (f", {impact} weak-dem" if impact < 0 else ""))
        elif news.get("n", 0) == 0:
            news_vote = 0
            news_lbl = "no recent news"
        else:
            news_vote = 0
            news_lbl = f"NEUTRAL avg {avg:+.2f} (n={news['n']})"

        z_vote = zs["vote"]
        combined = news_vote + z_vote
        if combined >= 2:
            direction, conv, lots = "LONG", "high", 5
        elif combined == 1:
            direction, conv, lots = "LONG", "med", 2
        elif combined == 0:
            direction, conv, lots = "HOLD", "low", 0
        elif combined == -1:
            direction, conv, lots = "SHORT", "med", 2
        else:
            direction, conv, lots = "SHORT", "high", 5

        # Current price for entry
        try:
            price_now = PRICE_FROM[prod](market)
        except Exception:
            price_now = None
        if price_now is not None:
            try:
                price_now = float(price_now)
            except Exception:
                price_now = None

        # Realized vol from product's price history.
        # For spot products use log-return vol; for the WTI-Brent spread we
        # compute the std of absolute daily $ changes (log-returns don't work
        # for series that can be ≈ 0 or negative).
        vol_pct = None         # vol20 ratio for display & lot-sizing gate
        sigma_d = None         # 1-day $ std for entry/TP/SL
        if prod == "wtcl":
            wh = list(market.hist.get("wti", []))
            bh = list(market.hist.get("brent", []))
            n = min(len(wh), len(bh))
            if n >= 22:
                sp_hist = [wh[-n + i] - bh[-n + i] for i in range(n)]
                sigma_d = _spread_dollar_vol(sp_hist)
                if sigma_d is not None and price_now is not None:
                    vol_pct = sigma_d / max(abs(price_now), 0.5)
        else:
            hist_key = HIST_KEYS.get(prod)
            if hist_key:
                hist = list(market.hist.get(hist_key, []))
                vol_pct = _daily_vol(hist)
                if vol_pct is not None and price_now is not None:
                    sigma_d = abs(price_now) * vol_pct

        # Halve lots in high-vol regime (matches P3F / P4D).
        if vol_pct is not None and vol_pct > 0.05 and lots > 0:
            lots = max(1, lots // 2)

        plan = (_trade_plan(price_now, direction, sigma_d, vol_pct)
                if price_now is not None else {})

        if   direction == "LONG":  n_long += 1
        elif direction == "SHORT": n_short += 1
        else:                      n_hold  += 1

        products_out.append({
            "product":     prod,
            "name":        PRODUCT_NAMES[prod],
            "price":       round(price_now, 4) if price_now is not None else None,
            "price_unit":  PRICE_UNIT[prod],
            "news_vote":   news_vote,
            "news_label":  news_lbl,
            "news_n":      news.get("n", 0),
            "news_avg":    avg,
            "news_impact": impact,
            "news_bull":   news.get("bull", 0),
            "news_bear":   news.get("bear", 0),
            "news_neutral": news.get("neutral", 0),
            "top_headlines": news.get("top_headlines", []),
            "z_vote":      z_vote,
            "z_label":     zs.get("label"),
            "combined":    combined,
            "direction":   direction,
            "conviction":  conv,
            "lots":        lots,
            "plan":        plan,
        })

    return {
        "available":  True,
        "products":   products_out,
        "n_long":     n_long,
        "n_short":    n_short,
        "n_hold":     n_hold,
        "explainer": (
            "Per product: routes recent news to that product via headline "
            "keywords, scores sentiment (FinBERT preferred over VADER), "
            "adds ±impact bias for supply-shock / weak-demand headlines, "
            "then votes alongside the existing BB-position + fly z-scores. "
            "Combined vote in {-2,-1,0,+1,+2} -> direction + conviction. "
            "Same vol-banded entry/TP/SL formula as P3F so plans are "
            "directly comparable across the three signal engines."
        ),
    }

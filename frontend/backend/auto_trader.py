"""Auto-trader bridge.

Takes the trade ideas surfaced by the term-structure regression engine and
the technical-analysis engine and packages them as signal dicts that the
existing paper book understands. The paper book then auto-executes them on
a virtual portfolio so the dashboard's equity curve reflects what these
two engines would have made if you had actually traded them.

Two trade families:

  source="term"  -- regression-engine ideas (P3B cross-product top).
                    Static while training data holds; position stays open
                    until the idea drops out of the top-N list, then we
                    emit a non-ACTIVE signal that closes it.

  source="tech"  -- technical-analysis ideas (P3E). Status flips with the
                    live RSI/BB/EMA-cross/momentum score. HOLD verdict
                    auto-closes the corresponding position.

To keep the existing paper book schema (one position per "title", flips on
ACTIVE -> non-ACTIVE), titles are stable strings:

    "TS CL M1-M2 (level)"    -- term-structure
    "TA WTI"                 -- technical

Asset prices for these new instruments are exposed via the same `assets`
dict so the paper book can mark-to-market and compute P&L. Each TS trade
prices off its product's FRONT MONTH price (a simplification of the actual
spread trade — see UI hint). Each TA trade prices off the product's spot.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Term-structure code -> (front asset_key, contract size in bbl-equivalent)
_PRODUCT_FRONT = {
    "CL":   ("cl_front",     1_000),
    "LCO3": ("brent_front",  1_000),
    "HO":   ("ho_front",     1_000),   # treat as bbl-equiv for paper sim
    "LGO":  ("brent_front",  1_000),   # no live LGO feed; proxy with Brent
    "WTCL": ("wtcl_spread",  1_000),
}

# Technical-signals product key -> (asset_key, contract size)
_TA_PRODUCT_FRONT = {
    "wti":    ("cl_front",     1_000),
    "brent":  ("brent_front",  1_000),
    "rbob":   ("rbob_front",   1_000),
    "heat":   ("ho_front",     1_000),
    "natgas": ("natgas_front", 1_000),
}


def _live_asset_prices(market) -> Dict[str, float]:
    """Pull the front-month price for each product we trade in the paper sim."""
    out: Dict[str, float] = {}
    try:
        if getattr(market, "wti", None) is not None:
            out["cl_front"] = float(market.wti)
        if getattr(market, "brent", None) is not None:
            out["brent_front"] = float(market.brent)
        if getattr(market, "rbob", None) is not None:
            out["rbob_front"] = float(market.rbob)
        if getattr(market, "heat", None) is not None:
            out["ho_front"] = float(market.heat)
        # Natural gas: derive from natgas_curve_hist or hist
        ng = None
        hist = getattr(market, "hist", {}) or {}
        ng_hist = hist.get("natgas") or []
        if ng_hist:
            ng = float(ng_hist[-1])
        else:
            ng_curve = getattr(market, "natgas_curve", None)
            if ng_curve:
                try:
                    ng = float(ng_curve[0].get("price"))
                except Exception:
                    ng = None
        if ng is not None:
            out["natgas_front"] = ng
        # WTI-Brent live spread (front month)
        if "cl_front" in out and "brent_front" in out:
            out["wtcl_spread"] = round(out["cl_front"] - out["brent_front"], 4)
    except Exception:
        pass
    return out


# Real-data signal -> (asset_key for live MTM, kind of mark)
# OUTRIGHT M1: trades the front-month directly.
# OUTRIGHT M2-M12: uses front-month as a high-correlation proxy (deep
#   tenors track front-month moves at ~0.7-0.9 beta on daily horizons).
# SPREAD / FLY: pinned to the static entry level — we don't have live
#   deep-curve data so the P&L is "frozen" until the signal regenerates.
#   This is the honest treatment: a STATIC mark that records the trade
#   without pretending we have live prices we don't have.
_REAL_OUTRIGHT_PROXY = {
    "CL":   "cl_front",
    "LCO":  "brent_front",
    "HO":   "ho_front",
    "LGO":  "brent_front",       # no LGO yfinance feed; Brent is the closest proxy
    "WTCL": "wtcl_spread",
}

# Composite-engine product key -> live asset key
_COMPOSITE_ASSET = {
    "wti":          "cl_front",
    "brent":        "brent_front",
    "wtcl_spread":  "wtcl_spread",
    "rbob":         "rbob_front",
    "heat":         "ho_front",
    "natgas":       "natgas_front",
}

# Paper-strategies engine: product code -> live asset key (proxy MTM)
_PSTRAT_PROXY = {
    "CL":   "cl_front",
    "LCO":  "brent_front",
    "LGO":  "brent_front",     # no LGO feed, use Brent as closest correlate
    "HO":   "ho_front",
    "WTCL": "wtcl_spread",
}

# Best-per-product engine: xlsx product code -> live asset key for MTM
_BPP_OUTRIGHT_PROXY = {
    "CL":   "cl_front",
    "LCO":  "brent_front",
    "HO":   "ho_front",
    "LGO":  "brent_front",   # no LGO live feed, Brent is closest proxy
    "WTCL": "wtcl_spread",
}


def _real_asset_key(signal: Dict[str, Any]) -> str:
    """Map a real-data signal to its paper-book asset_key."""
    kind = signal.get("kind", "")
    prod = signal.get("product", "")
    if kind == "OUTRIGHT":
        return _REAL_OUTRIGHT_PROXY.get(prod, "")
    # SPREAD / FLY: static key tied to this exact label
    safe = signal.get("label", "").replace(" ", "_").replace("-", "_")
    return f"real_static_{safe}"


def _bpp_signal_dict(prod_code: str, best_model: str,
                       trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a best-per-product trade entry into a paper-book signal.

    Each product uses ONLY its winning model from the 138-target backtest.
    Sizing scales with confidence tier (HIGH: 3 lots, MED: 2 lots)."""
    direction = trade.get("signal")
    if direction not in ("LONG", "SHORT"):
        return None
    conf = trade.get("confidence", "")
    if conf not in ("HIGH", "MED"):
        return None
    asset_key = _BPP_OUTRIGHT_PROXY.get(prod_code)
    if not asset_key:
        return None
    lots = 3 if conf == "HIGH" else 2
    r2 = trade.get("test_r2") or 0.0
    pred = trade.get("predicted_5d") or 0.0
    label = trade.get("label", "?")
    kind = trade.get("kind", "?")
    return {
        "title":     f"BPP {label} ({kind})",
        "asset_key": asset_key,
        "status":    "ACTIVE",
        "direction": direction,
        "size_bbl":  lots * 1_000,
        "rationale": (f"Best-per-product: {prod_code} -> {best_model} "
                      f"(its specialist model from 138-target backtest). "
                      f"Test R² {r2:+.3f}. Pred 5d {pred:+.3f}. Conf {conf}."),
        "source":    "bestpp",
        "model":     best_model,
        "test_r2":   r2,
        "lots":      lots,
        "conviction": conf,
    }


def _pstrat_signal_dict(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a paper-strategies signal to a paper-book trade.

    Only HIGH/MED-confidence LONG or SHORT signals are emitted (FLAT/MONITOR
    is just for the panel display).
    """
    direction = signal.get("direction", "")
    if direction not in ("LONG", "SHORT"):
        return None
    conf = signal.get("confidence", "")
    if conf not in ("HIGH", "MED"):
        return None
    prod = signal.get("product", "")
    asset_key = _PSTRAT_PROXY.get(prod)
    if not asset_key:
        return None
    engine = signal.get("engine", "?")
    label = signal.get("label", prod)
    lots = 3 if conf == "HIGH" else 2
    z_info = (signal.get("pc3_z") or signal.get("z_score")
               or signal.get("deviation") or 0)
    return {
        "title":     f"PSTRAT {engine} {label}",
        "asset_key": asset_key,
        "status":    "ACTIVE",
        "direction": direction,
        "size_bbl":  lots * 1_000,
        "rationale": (f"Paper-strategy {engine} signal on {prod}. "
                      f"z/deviation = {z_info}. Conviction {conf}. "
                      f"Backtest: PCA Sharpe 1.0-6.0 / Bertram Sharpe 3.4 / "
                      f"HMM Sharpe 0.88."),
        "source":    "pstrat",
        "engine":    engine,
        "lots":      lots,
        "conviction": conf,
    }


def _composite_signal_dict(product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one composite-strategy product entry into a paper-book signal.

    Composite fuses real-data + live LGBM + technical + news + regime + macro
    + regression with weights summing to 1.0. We only trade HIGH or MED
    conviction non-FLAT verdicts; size scales with conviction tier.
    """
    verdict = product.get("verdict", "")
    if verdict not in ("LONG", "SHORT"):
        return None
    conv = product.get("conviction", "")
    if conv not in ("HIGH", "MED"):
        return None
    key = product.get("product", "")
    asset_key = _COMPOSITE_ASSET.get(key)
    if not asset_key:
        return None
    lots = 3 if conv == "HIGH" else 2
    score = product.get("composite_score", 0)
    name = product.get("name", key)
    # Build a one-line summary of WHICH factors drove the verdict
    top_factors = sorted(product.get("factors") or [],
                          key=lambda f: -abs(f.get("vote", 0)))[:3]
    fac_str = ", ".join(f"{f['engine']}={f['vote']:+.1f}" for f in top_factors)
    return {
        "title":     f"COMPOSITE {name}",
        "asset_key": asset_key,
        "status":    "ACTIVE",
        "direction": verdict,
        "size_bbl":  lots * 1_000,
        "rationale": (f"Multi-factor composite score {score:+.2f} ({conv}). "
                      f"Top contributors: {fac_str}."),
        "source":    "composite",
        "composite_score": score,
        "lots":      lots,
        "conviction": conv,
    }


def _real_signal_dict(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a real-data signal into a paper-book signal dict."""
    direction = signal.get("signal", "")
    if direction not in ("LONG", "SHORT"):
        return None
    conf = signal.get("confidence", "")
    if conf not in ("HIGH", "MED"):
        return None
    asset_key = _real_asset_key(signal)
    if not asset_key:
        return None
    # Size: HIGH -> 2 lots, MED -> 1 lot. Small by default; paper.py risk
    # controls (stop-loss + time-stop + take-profit) cap downside.
    lots = 2 if conf == "HIGH" else 1
    label = signal.get("label", "?")
    model = signal.get("winner_model", "?")
    r2 = signal.get("test_r2") or 0.0
    pred = signal.get("predicted_5d") or 0.0
    return {
        "title":     f"REAL {label} ({signal.get('kind','?')})",
        "asset_key": asset_key,
        "status":    "ACTIVE",
        "direction": direction,
        "size_bbl":  lots * 1_000,
        "rationale": (f"Real-curve model {model} trained on user's xlsx data "
                      f"(60/20/20). Test R² {r2:+.3f}. Predicted 5d move {pred:+.3f}. "
                      f"Confidence {conf}."),
        "source":    "real",
        "model":     model,
        "test_r2":   r2,
        "predicted_5d": pred,
        "lots":      lots,
        "kind":      signal.get("kind"),
    }


def _ts_signal_dict(idea: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one term-structure cross-top idea into a paper-book signal."""
    product = idea.get("product", "")
    front = _PRODUCT_FRONT.get(product)
    if not front:
        return None
    asset_key, _ = front
    direction = idea.get("direction", "")
    if direction not in ("LONG", "SHORT"):
        return None
    lots = int(idea.get("lots") or 0)
    if lots <= 0:
        return None
    title = f"TS {product} {idea.get('label', '')} ({idea.get('kind', '')})"
    rationale = (
        f"Regression {idea.get('model','?')} R²={idea.get('r2',0)} · "
        f"actual {idea.get('actual')} vs pred {idea.get('predicted')} → "
        f"z={idea.get('z', 0)}σ. Spread mean-reverts toward model."
    )
    return {
        "title":     title,
        "asset_key": asset_key,
        "status":    "ACTIVE",
        "direction": direction,
        "size_bbl":  lots * 1_000,
        "rationale": rationale,
        "source":    "term",
        # extras the UI can render:
        "model":     idea.get("model"),
        "r2":        idea.get("r2"),
        "z":         idea.get("z"),
        "conf":      idea.get("conf"),
        "lots":      lots,
    }


def _ta_signal_dict(prod: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one technical-analysis product entry into a paper-book signal."""
    key = prod.get("product", "")
    front = _TA_PRODUCT_FRONT.get(key)
    if not front:
        return None
    asset_key, _ = front
    direction = prod.get("direction", "")
    lots = int(prod.get("lots") or 0)
    title = f"TA {prod.get('name', key)}"

    # If the verdict is HOLD we still emit a signal — status is non-ACTIVE
    # so the paper book closes any open position with this title.
    if direction not in ("LONG", "SHORT") or lots <= 0:
        return {
            "title":     title,
            "asset_key": asset_key,
            "status":    "HOLD",         # not ACTIVE -> triggers close
            "direction": "",
            "size_bbl":  0,
            "rationale": "TA signals mixed (|raw| < 2). Hold or close.",
            "source":    "tech",
            "raw_score": prod.get("raw_score"),
            "conf_pct":  prod.get("conf_pct"),
        }

    rationale = (
        f"TA raw {prod.get('raw_score', 0):+d}/±4 (RSI / BB / EMA-cross / "
        f"momentum aligned {prod.get('conf_pct', 0)}%). {direction} {lots} lot"
        f"{'s' if lots != 1 else ''}."
    )
    return {
        "title":     title,
        "asset_key": asset_key,
        "status":    "ACTIVE",
        "direction": direction,
        "size_bbl":  lots * 1_000,
        "rationale": rationale,
        "source":    "tech",
        "raw_score": prod.get("raw_score"),
        "conf_pct":  prod.get("conf_pct"),
        "lots":      lots,
    }


def build(market,
          term_structure: Optional[Dict[str, Any]],
          technical: Optional[Dict[str, Any]],
          real_data_panel: Optional[Dict[str, Any]] = None,
          composite_panel: Optional[Dict[str, Any]] = None,
          best_pp_panel: Optional[Dict[str, Any]] = None,
          pstrat_panel: Optional[Dict[str, Any]] = None,
          top_n: int = 10,
          real_top_n: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Produce the auto-trader signal list + the extended assets dict.

    Returns:
        signals: List of paper-book-compatible signal dicts. Includes both
                 ACTIVE (open trades) and HOLD (close-if-open) entries.
        assets:  Dict of asset_key -> live front-month price for each new
                 instrument. Caller must merge with the existing assets dict
                 before passing to paper_book.update().
    """
    signals: List[Dict[str, Any]] = []
    extra_assets = _live_asset_prices(market)

    # Term-structure: top-N cross-product ideas (active trades).
    if term_structure and term_structure.get("available"):
        top = (term_structure.get("cross_product_top") or [])[:top_n]
        for idea in top:
            sig = _ts_signal_dict(idea)
            if sig:
                signals.append(sig)

    # Technical: ALL products (so HOLD verdicts can close existing positions).
    if technical and technical.get("available"):
        for prod in technical.get("products") or []:
            sig = _ta_signal_dict(prod)
            if sig:
                signals.append(sig)

    # REAL-data (xlsx-trained 60/20/20 winners): top-N HIGH+MED signals.
    # Spreads/flies pin to a static price set here in extra_assets so the
    # paper book has SOMETHING to look up — without a live deep-curve feed
    # those positions don't move (entry == mark), and the time-stop in
    # paper.py force-closes them after 72h regardless.
    if real_data_panel and real_data_panel.get("available"):
        real_sigs = real_data_panel.get("signals") or []
        added = 0
        for s in real_sigs:
            if added >= real_top_n:
                break
            sig = _real_signal_dict(s)
            if not sig:
                continue
            ak = sig["asset_key"]
            # For spread/fly static keys, pin the price to the signal level
            # so the paper book has something to mark against.
            if ak.startswith("real_static_") and ak not in extra_assets:
                lvl = s.get("current_level")
                if lvl is not None:
                    extra_assets[ak] = float(lvl)
            # Skip emitting if we have no price for the asset_key.
            if ak not in extra_assets:
                continue
            signals.append(sig)
            added += 1

    # COMPOSITE strategy: HIGH/MED-conviction non-FLAT verdicts ONLY.
    # Each product produces at most one signal (whichever side the weighted
    # vote landed on). Uses live yfinance asset keys (cl_front, brent_front,
    # etc.) so MTM is real, not pinned.
    if composite_panel and composite_panel.get("available"):
        for prod in composite_panel.get("products") or []:
            sig = _composite_signal_dict(prod)
            if not sig:
                continue
            if sig["asset_key"] not in extra_assets:
                continue   # need a live price to mark
            signals.append(sig)

    # BEST-PER-PRODUCT strategy: each xlsx product code uses ONLY its
    # winning model's signals. Top-3 highest-R² trades per product max.
    if best_pp_panel and best_pp_panel.get("available"):
        for prod_code, prod_data in (best_pp_panel.get("per_product") or {}).items():
            best_model = prod_data.get("best_model", "?")
            for trade in prod_data.get("trades") or []:
                sig = _bpp_signal_dict(prod_code, best_model, trade)
                if not sig:
                    continue
                if sig["asset_key"] not in extra_assets:
                    continue
                signals.append(sig)

    # PAPER-STRATEGIES engine: PCA (CL/LCO/LGO) + Bertram (HO) + HMM (WTCL)
    # Only LONG/SHORT with HIGH/MED conviction; FLAT/MONITOR are display-only.
    if pstrat_panel and pstrat_panel.get("available"):
        for s in pstrat_panel.get("signals") or []:
            sig = _pstrat_signal_dict(s)
            if not sig:
                continue
            if sig["asset_key"] not in extra_assets:
                continue
            signals.append(sig)

    return signals, extra_assets

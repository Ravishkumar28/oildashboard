"""Term-structure trade engine.

Consumes the user-supplied regression results (`backend/data/term_structure_summary.py`)
and generates regime-aware trade signals for every calendar spread and butterfly
across five products: WTI (CL), Brent 3-yr (LCO3), Heating Oil (HO),
ICE Gasoil (LGO), and the WTI-Brent spread (WTCL).

For each (product, structure, kind) entry the latest residual
    r = actual_last - predicted_last
is compared to the model's RMSE.  z = r / rmse.  Strong positive z means the
spread/fly is rich (above what the model expects) -> SHORT it.  Strong negative
z -> cheap -> LONG it.

Only structures where the best model's out-of-sample R^2 > 0.30 generate
actionable signals; everything else is downgraded to WATCH.

Position sizing:
    lots = round(clip(|z| * R^2 * 2.5, 1, 5))

Each product is also tagged with a curve regime (steep_backwardation /
backwardation / flat / contango / steep_contango) derived from its M12-M1 slope
in the live market snapshot (positive = contango, negative = backwardation -
the standard convention).  Trades are grouped by regime so the user can see,
at a glance, "in the current contango regime, here is what to do across
spreads and flies."
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from .data.term_structure_summary import SUMMARY, LAST_DATE
except ImportError:
    from data.term_structure_summary import SUMMARY, LAST_DATE


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Contract sizing - bbl / gal / MMBtu * USD per unit ~ $/contract per $1 move.
# Used to convert "lots" into dollar P&L impact in the UI.
CONTRACT_SIZE_PER_DOLLAR = {
    "CL":   1_000.0,      # 1,000 bbl per CL contract
    "LCO3": 1_000.0,      # 1,000 bbl per ICE Brent
    "LGO":  100.0,        # 100 metric tonnes Gasoil ($/ton -> $/contract)
    "HO":   42_000.0,     # 42,000 gal per HO ($/gal -> $/contract)
    "WTCL": 1_000.0,      # WTI-Brent paired as 1k bbl
}

PRODUCT_NAMES = {
    "CL":   "WTI (CL)",
    "LCO3": "Brent (LCO)",
    "HO":   "Heating Oil (HO)",
    "LGO":  "Gasoil (LGO)",
    "WTCL": "WTI-Brent",
}

# Curve-slope cuts ($/bbl on the M1-M12 leg, scaled per product).
# Matches the existing regime_fly.py cuts so we stay consistent.
SLOPE_CUTS = {
    "CL":   (-4.0, -1.0, 1.0, 4.0),   # $/bbl
    "LCO3": (-4.0, -1.0, 1.0, 4.0),
    "HO":   (-0.10, -0.03, 0.03, 0.10),  # $/gal
    "LGO":  (-40.0, -10.0, 10.0, 40.0),  # $/ton
    "WTCL": (-2.0, -0.5, 0.5, 2.0),
}

REGIME_LABELS = (
    "steep_backwardation",
    "backwardation",
    "flat",
    "contango",
    "steep_contango",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _classify_regime(slope: float, cuts: Tuple[float, float, float, float]) -> str:
    """Map curve slope -> 5-bucket regime label.

    Slope convention: M12 - M1 (standard).  Positive = contango (forward
    premium), negative = backwardation (prompt premium).
    """
    a, b, c, d = cuts
    if slope <= a:
        return "steep_backwardation"
    if slope <= b:
        return "backwardation"
    if slope <= c:
        return "flat"
    if slope <= d:
        return "contango"
    return "steep_contango"


def _curve_slope(curve: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """M12 - M1 from a curve list of {"price": ...}.

    Standard contango/backwardation sign convention:
      positive slope  = forward premium (CONTANGO)
      negative slope  = prompt premium  (BACKWARDATION)
    Falls back to M_last - M1 if curve is shorter than 12 months.
    """
    if not curve:
        return None
    try:
        prices = [float(row.get("price", 0.0)) for row in curve]
        prices = [p for p in prices if p > 0]
        if len(prices) < 2:
            return None
        if len(prices) >= 12:
            return prices[11] - prices[0]
        return prices[-1] - prices[0]
    except Exception:
        return None


def _signal_lots(z: float, r2: float) -> int:
    """Lots from z-score and confidence (R^2). Range 1-3.

    Capped lower than before because the paper book was running 9 unhedged
    mean-reversion positions on a trending market. Smaller default size
    plus the new stop-loss in paper.py limits per-trade damage.
    """
    raw = abs(z) * max(0.0, r2) * 1.5
    return int(max(1, min(3, round(raw))))


def _format_struct(struct: str) -> str:
    """SPR_1_2 -> 'M1-M2',  FLY_3_4_5 -> 'M3-2M4+M5'."""
    if struct.startswith("SPR"):
        _, a, b = struct.split("_")
        return f"M{a}-M{b}"
    parts = struct.split("_")
    return f"M{parts[1]}-2M{parts[2]}+M{parts[3]}"


# --------------------------------------------------------------------------- #
# Per-entry signal
# --------------------------------------------------------------------------- #

def _entry_signal(product: str, struct: str, kind: str, row: tuple) -> Dict[str, Any]:
    model, r2, rmse, act, pred, n_test = row
    resid = float(act) - float(pred)
    z = resid / rmse if rmse > 1e-9 else 0.0

    # Confidence gate: only trade STRONG dislocations (|z| >= 1.5σ) on
    # confident models (R² >= 0.45). The previous 0.75σ / 0.30 R² gate was
    # opening trades on weak signals that trending markets blew through.
    confident = r2 >= 0.45
    if not confident or abs(z) < 1.5:
        direction = "WATCH"
        lots = 0
    elif z >= 1.5:
        direction = "SHORT"  # spread is rich vs model -> short it (expect revert down)
        lots = _signal_lots(z, r2)
    else:
        direction = "LONG"
        lots = _signal_lots(z, r2)

    # Confidence bucket for color-coding in the UI.
    if r2 >= 0.70:
        conf = "high"
    elif r2 >= 0.50:
        conf = "med"
    elif r2 >= 0.30:
        conf = "low"
    else:
        conf = "noise"

    return {
        "product":   product,
        "structure": struct,
        "label":     _format_struct(struct),
        "kind":      kind,
        "model":     model,
        "r2":        round(float(r2), 3),
        "rmse":      round(float(rmse), 3),
        "actual":    round(float(act), 3),
        "predicted": round(float(pred), 3),
        "residual":  round(float(resid), 3),
        "z":         round(float(z), 2),
        "direction": direction,
        "lots":      lots,
        "conf":      conf,
        "n_test":    int(n_test),
    }


# --------------------------------------------------------------------------- #
# Per-product signal sets + regime grouping
# --------------------------------------------------------------------------- #

def _product_signals(product: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (spread_signals, fly_signals) for one product."""
    spreads, flies = [], []
    for (p, struct, kind), row in SUMMARY.items():
        if p != product:
            continue
        sig = _entry_signal(p, struct, kind, row)
        if struct.startswith("SPR"):
            spreads.append(sig)
        else:
            flies.append(sig)
    # Keep level signals primary, then change signals for timing.
    def _sort_key(s):
        parts = s["structure"].split("_")[1:]
        return (s["kind"] != "level", tuple(int(x) for x in parts))
    spreads.sort(key=_sort_key)
    flies.sort(key=_sort_key)
    return spreads, flies


def _best_trade_ideas(signals: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Top-N actionable signals ranked by |z| * R^2."""
    actionable = [s for s in signals if s["direction"] in ("LONG", "SHORT")]
    actionable.sort(key=lambda s: abs(s["z"]) * max(0.0, s["r2"]), reverse=True)
    return actionable[:top_n]


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #

def build_panel(m, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the term-structure panel for the snapshot.

    Args:
        m: MarketEngine - used for current curve data (real_curve, brent_curve, etc.)
        snapshot: optional partial snapshot dict (unused now, future hook).

    Returns:
        Dict with:
            available: bool
            products: per-product breakdown (signals, regime, best ideas)
            cross_product_top: top 10 ideas across all products
            generated_at: ISO timestamp (from market clock)
    """
    products_out: Dict[str, Any] = {}
    all_signals: List[Dict[str, Any]] = []

    # Map each product code to the live curve from MarketEngine.
    curve_for = {
        "CL":   getattr(m, "real_curve", None),
        "LCO3": getattr(m, "brent_curve", None),
        "HO":   getattr(m, "heat_curve", None),
        "LGO":  getattr(m, "gasoil_curve", None) or getattr(m, "brent_curve", None),
        "WTCL": None,  # WTI-Brent has its own regime - we use spread directly
    }

    # WTI-Brent live spread (front-month) for WTCL regime.
    wti_front = None
    brent_front = None
    try:
        if getattr(m, "real_curve", None):
            wti_front = float(m.real_curve[0]["price"])
        if getattr(m, "brent_curve", None):
            brent_front = float(m.brent_curve[0]["price"])
    except Exception:
        pass

    for product in PRODUCT_NAMES:
        spreads, flies = _product_signals(product)
        if not spreads and not flies:
            continue

        # ----- regime classification ----- #
        if product == "WTCL" and wti_front is not None and brent_front is not None:
            slope = wti_front - brent_front  # negative when Brent premium dominates
        else:
            slope = _curve_slope(curve_for.get(product))

        regime = _classify_regime(slope, SLOPE_CUTS[product]) if slope is not None else "unknown"

        best_ideas = _best_trade_ideas(spreads + flies, top_n=5)

        products_out[product] = {
            "name":         PRODUCT_NAMES[product],
            "regime":       regime,
            "slope":        round(float(slope), 3) if slope is not None else None,
            "spreads":      spreads,
            "flies":        flies,
            "best_ideas":   best_ideas,
            "contract_per_dollar": CONTRACT_SIZE_PER_DOLLAR.get(product, 1000.0),
            "last_test_date": LAST_DATE.get(product),
            "n_spreads":    len(spreads),
            "n_flies":      len(flies),
            "n_actionable": sum(1 for s in spreads + flies if s["direction"] in ("LONG", "SHORT")),
        }
        all_signals.extend(spreads + flies)

    # Cross-product top ideas ranked by |z| * R^2.
    cross_top = _best_trade_ideas(all_signals, top_n=10)

    # Per-regime aggregation across all products.
    by_regime: Dict[str, List[Dict[str, Any]]] = {r: [] for r in REGIME_LABELS}
    for product, info in products_out.items():
        regime = info["regime"]
        if regime not in by_regime:
            continue
        for sig in info["best_ideas"]:
            by_regime[regime].append({**sig, "product_name": info["name"]})

    # Quality summary: how many high/med/low confidence signals.
    quality = {"high": 0, "med": 0, "low": 0, "noise": 0}
    for sig in all_signals:
        if sig["conf"] in quality:
            quality[sig["conf"]] += 1

    return {
        "available":         True,
        "products":          products_out,
        "cross_product_top": cross_top,
        "by_regime":         by_regime,
        "quality_counts":    quality,
        "n_total_signals":   len(all_signals),
        "n_actionable":      sum(1 for s in all_signals if s["direction"] in ("LONG", "SHORT")),
        "data_source":       "User-supplied regression run; 10 model families per target.",
        "explainer": (
            "z = (actual_last - predicted_last) / RMSE.  Strong +z = rich vs model -> SHORT. "
            "Strong -z = cheap -> LONG.  Position size = round(clip(|z| * R^2 * 2.5, 1, 5)) lots. "
            "Only structures with R^2 >= 0.30 generate actionable trades."
        ),
    }

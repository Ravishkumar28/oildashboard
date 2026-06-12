"""Opportunity ranking engine.

Phase 2 deliverable: combine the regime classifier + historical regime DB
+ live market state to produce a RANKED list of "what's mispriced right now."

For every (product, structure) pair:
  product   ∈ {WTI, Brent, RBOB, HO, NatGas}
  structure ∈ {fly, M1-M2, M3-M6, M6-M9, slope}

1. Compute current value from the live curve
2. Look up the regime-conditional historical distribution
   (we have per-DIMENSION historical stats from regime_history.py)
3. Compute the deviation: z = (actual − expected_mean) / expected_std
4. Compute a confidence score: higher when more historical samples back
   the comparison, lower when the regime is sparsely populated
5. Compute a robustness score: how consistent the deviation is across
   different dimensions of the regime fingerprint — a fly that's
   2σ rich vs the Inventory dimension AND 1.8σ rich vs the Volatility
   dimension AND 2.2σ rich vs the Curve dimension is a much more robust
   signal than one that's only off vs one slice.

Final rank score = |z_avg| × confidence × robustness

Each opportunity carries a full natural-language rationale so the operator
sees exactly WHY a row is on the list.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import regime_history as rh
import regime_classifier as rc


PRODUCTS_FOR_RANK = [
    {"key": "wti",    "name": "WTI Crude",      "curve_attr": "real_curve",
     "hist_attr": "curve_hist"},
    {"key": "brent",  "name": "Brent Crude",    "curve_attr": "brent_curve",
     "hist_attr": "brent_curve_hist"},
    {"key": "rbob",   "name": "RBOB Gasoline",  "curve_attr": "rbob_curve",
     "hist_attr": "rbob_curve_hist"},
    {"key": "heat",   "name": "Heating Oil",    "curve_attr": "heat_curve",
     "hist_attr": "heat_curve_hist"},
    {"key": "natgas", "name": "Natural Gas",    "curve_attr": "natgas_curve",
     "hist_attr": "natgas_curve_hist"},
]

# Per-product curve-slope cuts for the simple regime classifier used on
# non-WTI products. Slope = M1 - M_last. Buckets in slope-ascending order:
# steep_backwardation, backwardation, flat, contango, steep_contango.
_PRODUCT_SLOPE_CUTS = {
    "wti":    (-4.0, -1.0, 1.0, 4.0),
    "brent":  (-4.0, -1.0, 1.0, 4.0),
    "rbob":   (-0.10, -0.03, 0.03, 0.10),
    "heat":   (-0.10, -0.03, 0.03, 0.10),
    "natgas": (-0.50, -0.10, 0.10, 0.50),
}
_REGIME_LABELS = ("steep_backwardation", "backwardation", "flat",
                  "contango", "steep_contango")


def _bucket_for_slope(slope: float, cuts) -> int:
    a, b, c, d = cuts
    if slope <= a: return 0
    if slope <= b: return 1
    if slope <= c: return 2
    if slope <= d: return 3
    return 4


def _per_product_db(market, product_key: str) -> Optional[Dict]:
    """Build a per-product regime-conditional spread/fly database.

    Uses ONLY the product's own curve_hist (no WTI fingerprint mixing). The
    regime is single-dimensional — curve-slope bucket — which is the most
    relevant axis for spread / fly trades anyway.

    Returns:
        {
            "buckets": {
                bucket_idx: {
                    "label": "backwardation",
                    "fly":   {"n":N, "mean":..., "std":..., "p10":...},
                    "m1_m2": {...}, "m3_m6": {...}, "m6_m9": {...},
                    "slope": {...}
                }, ...
            },
            "current_bucket": int,  # current regime bucket from latest curve
            "current_label":  str,
            "n_total": int,
        }
        or None if no history available.
    """
    cfg = next((p for p in PRODUCTS_FOR_RANK if p["key"] == product_key), None)
    if not cfg:
        return None
    hist = list(getattr(market, cfg["hist_attr"], []) or [])
    if len(hist) < 20:
        return None

    cuts = _PRODUCT_SLOPE_CUTS.get(product_key, _PRODUCT_SLOPE_CUTS["wti"])

    # Walk the history; per bucket, accumulate samples per structure.
    by_bucket: Dict[int, Dict[str, List[float]]] = {}
    for curve in hist:
        if len(curve) < 9:
            continue
        slope = curve[-1] - curve[0]
        bk = _bucket_for_slope(slope, cuts)
        stats = {
            "fly":   curve[2] - 2 * curve[5] + curve[8],
            "m1_m2": curve[1] - curve[0],
            "m3_m6": curve[5] - curve[2],
            "m6_m9": curve[8] - curve[5],
            "slope": slope,
        }
        bucket = by_bucket.setdefault(bk, {s: [] for s in stats})
        for s, v in stats.items():
            bucket[s].append(v)

    buckets_out: Dict[int, Dict] = {}
    for bk, samples in by_bucket.items():
        buckets_out[bk] = {
            "label": _REGIME_LABELS[bk],
            **{s: rh._stats(v) for s, v in samples.items()},
        }

    # Classify current state from the latest curve.
    cur_curve = hist[-1]
    cur_slope = cur_curve[-1] - cur_curve[0] if len(cur_curve) >= 2 else 0.0
    cur_bk = _bucket_for_slope(cur_slope, cuts)

    return {
        "buckets":        buckets_out,
        "current_bucket": cur_bk,
        "current_label":  _REGIME_LABELS[cur_bk],
        "n_total":        sum(len(v["fly"]) for v in by_bucket.values()),
    }

STRUCTURES = ["fly", "m1_m2", "m3_m6", "m6_m9", "slope"]

STRUCTURE_LABELS = {
    "fly":   "M3-M6-M9 Fly",
    "m1_m2": "M1-M2 Spread",
    "m3_m6": "M3-M6 Spread",
    "m6_m9": "M6-M9 Spread",
    "slope": "M12-M1 Slope",
}


def _curve_prices(curve) -> List[float]:
    if not curve:
        return []
    try:
        return [float(row.get("price", 0.0)) for row in curve]
    except Exception:
        return []


def _struct_values(prices: List[float]) -> Optional[Dict[str, float]]:
    if len(prices) < 9:
        return None
    return {
        "fly":   prices[2] - 2 * prices[5] + prices[8],
        "m1_m2": prices[1] - prices[0],
        "m1_m3": prices[2] - prices[0],
        "m3_m6": prices[5] - prices[2],
        "m6_m9": prices[8] - prices[5],
        "slope": prices[-1] - prices[0],
    }


def _z_vs_stats(actual: float, stats: Optional[Dict]) -> Optional[float]:
    if not stats or stats.get("mean") is None or stats.get("std") is None:
        return None
    if stats["std"] == 0:
        return None
    return (actual - stats["mean"]) / stats["std"]


def _direction(z: float) -> str:
    if z >= 0.5:  return "SHORT"   # value is rich → expect mean-revert down
    if z <= -0.5: return "LONG"    # value is cheap → expect mean-revert up
    return "WATCH"


def build_opportunities(market, db: Optional[Dict] = None,
                        regime: Optional[Dict] = None) -> Dict:
    """Compute ranked opportunities across all products × structures.

    `db` and `regime` are passed in to avoid recomputation when the caller
    already has them — both are optional and rebuilt if missing."""
    if db is None:
        db = rh.build_database(market)
    if not db.get("available"):
        return {"available": False,
                "reason":    db.get("reason", "regime DB unavailable")}

    # Current regime fingerprint from market state — uses WTI as anchor
    if regime is None:
        wti_hist = list(market.hist.get("wti", []))
        dxy_hist = list(market.hist.get("dxy", []))
        inv_hist = list(market.hist.get("crude_inventory", []))
        wti_curve = _curve_prices(getattr(market, "real_curve", None))
        regime = rc.classify(wti_hist, wti_curve, inv_hist, dxy_hist)

    cur_buckets = [d["bucket"] for d in regime["dimensions"]]
    dim_names = [d["name"] for d in regime["dimensions"]]

    # Per-dimension historical stats from `db`
    per_dim = db.get("per_dim") or []

    opportunities: List[Dict] = []

    # Only WTI structures have historical regime stats (since regime_history
    # builds from curve_hist which is WTI-only). For Brent/RBOB/HO/NatGas we
    # report the current value but flag historical comparison unavailable.
    for cfg in PRODUCTS_FOR_RANK:
        curve = getattr(market, cfg["curve_attr"], None)
        prices = _curve_prices(curve)
        vals = _struct_values(prices)
        if not vals:
            continue

        for struct in STRUCTURES:
            actual = vals[struct]

            if cfg["key"] != "wti":
                # Other products: now have their own curve_hist seeded from
                # Yahoo at boot. Compute regime-conditional z-score using
                # the product's own historical curve distribution, bucketed
                # by curve slope (single dimension — most relevant for
                # spread / fly trades).
                pdb = _per_product_db(market, cfg["key"])
                if not pdb or struct not in ("fly", "m1_m2", "m3_m6", "m6_m9", "slope"):
                    opportunities.append({
                        "product":      cfg["name"],
                        "product_key":  cfg["key"],
                        "structure":    struct,
                        "structure_label": STRUCTURE_LABELS[struct],
                        "actual":       round(actual, 4),
                        "expected":     None,
                        "z_avg":        None,
                        "confidence":   0.0,
                        "robustness":   0.0,
                        "score":        0.0,
                        "direction":    "WATCH",
                        "rationale":    (f"Per-product history not yet seeded. "
                                         f"Current {STRUCTURE_LABELS[struct]} "
                                         f"= {actual:.3f}."),
                        "per_dim_z":    [],
                    })
                    continue
                cur_bucket = pdb["buckets"].get(pdb["current_bucket"], {})
                stats = cur_bucket.get(struct)
                z = _z_vs_stats(actual, stats)
                if z is None or not stats or stats.get("n", 0) < 5:
                    opportunities.append({
                        "product":      cfg["name"],
                        "product_key":  cfg["key"],
                        "structure":    struct,
                        "structure_label": STRUCTURE_LABELS[struct],
                        "actual":       round(actual, 4),
                        "expected":     stats.get("mean") if stats else None,
                        "z_avg":        None,
                        "confidence":   0.0,
                        "robustness":   0.0,
                        "score":        0.0,
                        "direction":    "WATCH",
                        "rationale":    (f"Insufficient samples in current "
                                         f"{pdb['current_label']} regime "
                                         f"(n={stats.get('n', 0) if stats else 0}). "
                                         f"Current {STRUCTURE_LABELS[struct]} "
                                         f"= {actual:.3f}."),
                        "per_dim_z":    [],
                    })
                    continue
                import math
                confidence = min(1.0, math.log10(max(2, stats["n"])) / 2.5)
                robustness = 0.6  # single-dim, fixed prior
                score = abs(z) * confidence * robustness
                direction = _direction(z)
                rationale = (
                    f"Curve regime: {pdb['current_label']} (n={stats['n']} "
                    f"historical observations). Regime-mean "
                    f"{STRUCTURE_LABELS[struct]} = {stats['mean']:.3f}, "
                    f"actual = {actual:.3f}, σ = {stats['std']:.3f}. "
                    f"z = {z:+.2f}σ → {direction}."
                )
                opportunities.append({
                    "product":      cfg["name"],
                    "product_key":  cfg["key"],
                    "structure":    struct,
                    "structure_label": STRUCTURE_LABELS[struct],
                    "actual":       round(actual, 4),
                    "expected":     round(stats["mean"], 4),
                    "z_avg":        round(z, 2),
                    "confidence":   round(confidence, 3),
                    "robustness":   round(robustness, 3),
                    "score":        round(score, 3),
                    "direction":    direction,
                    "rationale":    rationale,
                    "per_dim_z":    [{
                        "dimension":     "Curve slope",
                        "bucket_label":  pdb["current_label"],
                        "expected_mean": stats["mean"],
                        "expected_std":  stats["std"],
                        "n":             stats["n"],
                        "z":             round(z, 2),
                    }],
                })
                continue

            # For WTI: z vs each dimension's bucket-conditional distribution
            per_dim_z: List[Dict] = []
            expected_means: List[float] = []
            expected_ns: List[int] = []
            for d_idx, dim in enumerate(per_dim):
                bk_idx = cur_buckets[d_idx]
                stats = None
                for b in dim["buckets"]:
                    if b["bucket_idx"] == bk_idx:
                        stats = b.get(struct)
                        break
                z = _z_vs_stats(actual, stats)
                if z is not None and stats and stats.get("n", 0) >= 5:
                    per_dim_z.append({
                        "dimension":   dim["dimension"],
                        "bucket_label": next(
                            (b["label"] for b in dim["buckets"]
                             if b["bucket_idx"] == bk_idx),
                            "?"),
                        "expected_mean": stats["mean"],
                        "expected_std":  stats["std"],
                        "n":            stats["n"],
                        "z":            round(z, 2),
                    })
                    expected_means.append(stats["mean"])
                    expected_ns.append(stats["n"])

            if not per_dim_z:
                opportunities.append({
                    "product":         cfg["name"],
                    "product_key":     cfg["key"],
                    "structure":       struct,
                    "structure_label": STRUCTURE_LABELS[struct],
                    "actual":          round(actual, 4),
                    "expected":        None,
                    "z_avg":           None,
                    "confidence":      0.0,
                    "robustness":      0.0,
                    "score":           0.0,
                    "direction":       "WATCH",
                    "rationale":       (f"No historical samples in current regime "
                                        f"buckets for this structure yet."),
                    "per_dim_z":       [],
                })
                continue

            # Aggregate
            z_avg = sum(d["z"] for d in per_dim_z) / len(per_dim_z)
            exp_mean = sum(expected_means) / len(expected_means)
            # Confidence: log-scale of total samples across dimensions
            n_total = sum(expected_ns)
            import math
            confidence = min(1.0, math.log10(max(2, n_total)) / 3.0)
            # Robustness: 1 - (std of per-dim z's) / max(|avg|, 1)
            z_values = [d["z"] for d in per_dim_z]
            zsd = (sum((z - z_avg) ** 2 for z in z_values) / len(z_values)) ** 0.5
            robustness = max(0.0, 1.0 - zsd / max(abs(z_avg), 1.0))

            score = abs(z_avg) * confidence * robustness
            direction = _direction(z_avg)

            # Build rationale
            top_signals = sorted(per_dim_z, key=lambda d: -abs(d["z"]))[:3]
            sig_str = "; ".join(
                f"{d['dimension']}({d['bucket_label']}): z={d['z']:+.1f} "
                f"(n={d['n']})"
                for d in top_signals
            )
            if abs(z_avg) >= 1.5:
                bias = ("RICH" if z_avg > 0 else "CHEAP")
                rationale = (f"{STRUCTURE_LABELS[struct]} is {bias} vs the current "
                             f"regime. Avg z = {z_avg:+.2f} across "
                             f"{len(per_dim_z)} dimensions. Strongest: {sig_str}. "
                             f"Expected ≈ {exp_mean:.3f}, actual = {actual:.3f}.")
            else:
                rationale = (f"In line with regime — avg z = {z_avg:+.2f}, "
                             f"expected ≈ {exp_mean:.3f}, actual = {actual:.3f}.")

            opportunities.append({
                "product":         cfg["name"],
                "product_key":     cfg["key"],
                "structure":       struct,
                "structure_label": STRUCTURE_LABELS[struct],
                "actual":          round(actual, 4),
                "expected":        round(exp_mean, 4),
                "z_avg":           round(z_avg, 2),
                "confidence":      round(confidence, 2),
                "robustness":      round(robustness, 2),
                "score":           round(score, 3),
                "direction":       direction,
                "rationale":       rationale,
                "per_dim_z":       per_dim_z,
            })

    # Sort by descending score (highest signal first)
    opportunities.sort(key=lambda o: -(o.get("score") or 0))

    return {
        "available": True,
        "opportunities": opportunities,
        "ranked_top":    opportunities[:8],
        "current_regime_label": regime.get("fingerprint_label"),
    }

"""Historical regime database.

Walks the WTI price + curve history, classifies each historical point with
the 5-dimension fingerprint from regime_classifier, and bins distributional
statistics per regime. Result: a lookup table that answers
    "what's a typical fly value when we're in (Low Inv, High Vol,
     Summer Driving, Steep Backwardation, Strong Dollar)?"

Used by the opportunity engine to compute z-score deviations of CURRENT
fly / spread values vs their regime-conditional historical distribution.

Two granularities:
  * Per-DIMENSION stats (single dimension at a time) — much higher sample
    count per bucket, useful when the 5-tuple regime is too rare in history
  * Per-FINGERPRINT stats (full 5-tuple) — most regime-specific but only
    populated for the most common fingerprints

`build_database()` is called once at refresh tick — caches result for
`REFRESH_EVERY_SEC` seconds. Database is O(MB) so safe to keep in memory.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import regime_classifier as rc


REFRESH_EVERY_SEC = 300


def _stats(values: List[float]) -> Dict:
    """Distributional summary for a sample. Returns mean, std, n, quantiles."""
    n = len(values)
    if n < 3:
        return {"n": n, "mean": None, "std": None, "p10": None,
                "p50": None, "p90": None}
    s = sorted(values)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    std = var ** 0.5
    def q(p):
        i = max(0, min(n - 1, int(p * (n - 1))))
        return s[i]
    return {"n": n, "mean": round(mean, 4), "std": round(std, 4),
            "p10": round(q(0.10), 4), "p50": round(q(0.50), 4),
            "p90": round(q(0.90), 4)}


def _spreads_and_fly_from_curve(curve: List[float]) -> Optional[Dict]:
    if len(curve) < 9:
        return None
    return {
        "fly":   curve[2] - 2 * curve[5] + curve[8],
        "m1_m2": curve[1] - curve[0],
        "m1_m3": curve[2] - curve[0],
        "m3_m6": curve[5] - curve[2],
        "m6_m9": curve[8] - curve[5],
        "slope": curve[-1] - curve[0],
    }


class _DatabaseCache:
    def __init__(self) -> None:
        self.last_built: float = 0.0
        self.payload: Dict = {}

    def build(self, market) -> Dict:
        now = time.time()
        if self.payload and now - self.last_built < REFRESH_EVERY_SEC:
            return self.payload

        wti_hist = list(market.hist.get("wti", []))
        dxy_hist = list(market.hist.get("dxy", []))
        inv_hist = list(market.hist.get("crude_inventory", []))
        curves = list(market.curve_hist)
        n_wti = len(wti_hist)
        n_curves = len(curves)
        if n_wti < 60 or n_curves < 20:
            self.payload = {
                "available": False,
                "reason": (f"Insufficient history (WTI={n_wti}, "
                            f"curves={n_curves}) — need ≥60 / ≥20.")
            }
            self.last_built = now
            return self.payload

        # Per-dimension samples (each fingerprint dimension separately)
        # dim_samples[dim_idx][bucket_idx] = list of fly values
        dim_samples_fly: Dict[int, Dict[int, List[float]]] = {
            d: {} for d in range(5)
        }
        dim_samples_m1m2: Dict[int, Dict[int, List[float]]] = {
            d: {} for d in range(5)
        }
        dim_samples_m3m6: Dict[int, Dict[int, List[float]]] = {
            d: {} for d in range(5)
        }
        dim_samples_m6m9: Dict[int, Dict[int, List[float]]] = {
            d: {} for d in range(5)
        }
        dim_samples_slope: Dict[int, Dict[int, List[float]]] = {
            d: {} for d in range(5)
        }
        # Full-fingerprint samples
        fp_samples_fly: Dict[Tuple, List[float]] = {}
        # Per-dimension bucket labels (lookup for the UI)
        dim_labels: Dict[int, Dict[int, str]] = {d: {} for d in range(5)}

        # Iterate over curve snapshots. Each curve corresponds to a tick.
        for i in range(n_curves - 1):
            curve = curves[i]
            stats = _spreads_and_fly_from_curve(curve)
            if not stats:
                continue
            # Align curve index back to WTI index. Curves grow in lock-step
            # with WTI, so the last curve = last WTI tick.
            wti_idx = n_wti - (n_curves - i) - 1
            if wti_idx < 20 or wti_idx >= n_wti:
                continue

            fp = rc.classify(
                price_hist=wti_hist[:wti_idx + 1],
                curve_prices=list(curve),
                inv_hist=inv_hist[:wti_idx + 1],
                dxy_hist=dxy_hist[:wti_idx + 1],
            )
            for d_idx, dim in enumerate(fp["dimensions"]):
                bk = dim["bucket"]
                dim_labels[d_idx][bk] = dim["label"]
                dim_samples_fly[d_idx].setdefault(bk, []).append(stats["fly"])
                dim_samples_m1m2[d_idx].setdefault(bk, []).append(stats["m1_m2"])
                dim_samples_m3m6[d_idx].setdefault(bk, []).append(stats["m3_m6"])
                dim_samples_m6m9[d_idx].setdefault(bk, []).append(stats["m6_m9"])
                dim_samples_slope[d_idx].setdefault(bk, []).append(stats["slope"])
            fp_samples_fly.setdefault(fp["fingerprint"], []).append(stats["fly"])

        # Build payload — per-dimension stats
        dim_names = ["Inventory", "Volatility", "Seasonality", "Curve", "Macro"]
        dim_stats: List[Dict] = []
        for d_idx, name in enumerate(dim_names):
            buckets = []
            for bk, samples in sorted(dim_samples_fly[d_idx].items()):
                buckets.append({
                    "bucket_idx":  bk,
                    "label":       dim_labels[d_idx].get(bk, "?"),
                    "fly":         _stats(samples),
                    "m1_m2":       _stats(dim_samples_m1m2[d_idx].get(bk, [])),
                    "m3_m6":       _stats(dim_samples_m3m6[d_idx].get(bk, [])),
                    "m6_m9":       _stats(dim_samples_m6m9[d_idx].get(bk, [])),
                    "slope":       _stats(dim_samples_slope[d_idx].get(bk, [])),
                })
            dim_stats.append({"dimension": name, "buckets": buckets})

        # Full-fingerprint stats (top 8 by sample size)
        fp_stats = []
        for fp, samples in fp_samples_fly.items():
            fp_stats.append({
                "fingerprint": list(fp),
                "key": rc.regime_key(fp),
                "fly": _stats(samples),
            })
        fp_stats.sort(key=lambda x: -x["fly"]["n"])
        fp_stats = fp_stats[:8]

        self.payload = {
            "available":  True,
            "n_total":    n_curves,
            "per_dim":    dim_stats,
            "top_fingerprints": fp_stats,
            "dim_labels_for_idx": {d: dict(dim_labels[d]) for d in range(5)},
        }
        self.last_built = now
        return self.payload


_DB = _DatabaseCache()


def build_database(market) -> Dict:
    """Public entrypoint — returns cached snapshot of the per-regime DB."""
    return _DB.build(market)


def lookup_expected(market, dimension_idx: int, bucket_idx: int,
                     measure: str) -> Optional[Dict]:
    """For a single dimension/bucket, return the stats dict for `measure`."""
    db = _DB.payload
    if not db or not db.get("available"):
        return None
    per_dim = db["per_dim"]
    if dimension_idx >= len(per_dim):
        return None
    for b in per_dim[dimension_idx]["buckets"]:
        if b["bucket_idx"] == bucket_idx:
            return b.get(measure)
    return None

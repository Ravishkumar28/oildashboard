"""Compare TWO regime classification schemes on the SAME data, per product.

Scheme A — HARD CUTOFFS (current production rule, backend/regime_classifier.py):
    slope <= -4   ->  "Steep Backwardation"
    -4 <  slope <= -1 -> "Backwardation"
    -1 <  slope <=  1 -> "Flat"
     1 <  slope <=  4 -> "Contango"
     4 <  slope        -> "Steep Contango"

Scheme B — PERCENTILES PER PRODUCT:
    Each product's M12-M1 slope distribution (over the full xlsx history)
    is bucketed into 5 quintiles. The quintile cutoffs are PRODUCT-SPECIFIC
    (so each product gets a well-balanced 20-20-20-20-20 split) and FROZEN
    at the cutoffs derived from the full series. Labels are:
        Q1 (deepest backwardation)
        Q2
        Q3 (middle)
        Q4
        Q5 (deepest contango)

NO prediction. NO ML. This is a labeling/bucketing comparison.

For each product:
  - print the percentile cutoffs (the actual slope numbers at the 20/40/60/80 marks)
  - compare per-day distribution under both schemes
  - measure agreement rate (how often do the two schemes give the same regime ordinal)
  - report today's label under each scheme
  - measure "stickiness" (mean run-length of consecutive identical labels)

Output saved to:
  backend/data/percentile_regime_cutoffs.json   - the per-product cutoffs
  tools/results/regime_label_comparison.json    - the comparison metrics
"""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_FILES = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
    "WTCL": ROOT / "wtcl_lco_outrights_1min_trimmed_daily_close.xlsx",
}

OUT_CUTOFFS    = ROOT / "backend" / "data" / "percentile_regime_cutoffs.json"
OUT_COMPARISON = ROOT / "tools" / "results" / "regime_label_comparison.json"


HARD_CUTS = (-4.0, -1.0, 1.0, 4.0)
HARD_NAMES = ["Steep Backwardation", "Backwardation", "Flat",
               "Contango", "Steep Contango"]
PCT_NAMES = ["Q1 (deepest back)", "Q2", "Q3 (middle)", "Q4", "Q5 (deepest contango)"]


def hard_label(slope: float) -> str:
    if slope <= HARD_CUTS[0]: return HARD_NAMES[0]
    if slope <= HARD_CUTS[1]: return HARD_NAMES[1]
    if slope <= HARD_CUTS[2]: return HARD_NAMES[2]
    if slope <= HARD_CUTS[3]: return HARD_NAMES[3]
    return HARD_NAMES[4]


def hard_index(slope: float) -> int:
    for i, c in enumerate(HARD_CUTS):
        if slope <= c: return i
    return 4


def pct_label(slope: float, cuts: Tuple[float, float, float, float]) -> str:
    if slope <= cuts[0]: return PCT_NAMES[0]
    if slope <= cuts[1]: return PCT_NAMES[1]
    if slope <= cuts[2]: return PCT_NAMES[2]
    if slope <= cuts[3]: return PCT_NAMES[3]
    return PCT_NAMES[4]


def pct_index(slope: float, cuts: Tuple[float, float, float, float]) -> int:
    for i, c in enumerate(cuts):
        if slope <= c: return i
    return 4


def load_slope_series(path: Path) -> pd.Series:
    """M12-M1 slope per day."""
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    if "c1||weighted_mid" not in df.columns or "c12||weighted_mid" not in df.columns:
        return pd.Series(dtype=float)
    m1  = pd.to_numeric(df["c1||weighted_mid"], errors="coerce")
    m12 = pd.to_numeric(df["c12||weighted_mid"], errors="coerce")
    return (m12 - m1).dropna()


def stickiness(labels: List[str]) -> float:
    """Average number of consecutive identical labels (regime run length)."""
    if not labels:
        return 0.0
    runs = []
    cur, n = labels[0], 1
    for L in labels[1:]:
        if L == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = L, 1
    runs.append(n)
    return sum(runs) / len(runs)


def main():
    cutoffs_out = {}
    comparison_out = {}

    print("=" * 90)
    print("REGIME LABELING COMPARISON — Hard cutoffs (current) vs Per-Product Percentiles")
    print("=" * 90)

    for prod, path in PRODUCT_FILES.items():
        if not path.exists():
            continue
        slope = load_slope_series(path)
        if len(slope) < 100:
            continue

        # Compute per-product percentile cutoffs from FULL history
        p20, p40, p60, p80 = np.percentile(slope.values, [20, 40, 60, 80])
        pct_cuts = (float(p20), float(p40), float(p60), float(p80))
        cutoffs_out[prod] = {
            "p20": round(pct_cuts[0], 4),
            "p40": round(pct_cuts[1], 4),
            "p60": round(pct_cuts[2], 4),
            "p80": round(pct_cuts[3], 4),
            "n_days_history": int(len(slope)),
            "fit_range": f"{slope.index[0].date()} -> {slope.index[-1].date()}",
        }

        hard_labels = [hard_label(s) for s in slope.values]
        pct_labels  = [pct_label(s, pct_cuts) for s in slope.values]

        hard_dist = Counter(hard_labels)
        pct_dist  = Counter(pct_labels)

        # Agreement: same ordinal class index under both schemes
        hard_idx = [hard_index(s) for s in slope.values]
        pct_idx  = [pct_index(s, pct_cuts) for s in slope.values]
        agree_n = sum(1 for h, p in zip(hard_idx, pct_idx) if h == p)
        agree_pct = agree_n / len(slope) * 100

        # Today's labels
        today_slope = float(slope.iloc[-1])
        today_hard = hard_label(today_slope)
        today_pct  = pct_label(today_slope, pct_cuts)

        # Stickiness — how often does each label persist
        hard_stickiness = stickiness(hard_labels)
        pct_stickiness  = stickiness(pct_labels)

        comparison_out[prod] = {
            "today_slope":  round(today_slope, 3),
            "today_hard":   today_hard,
            "today_pct":    today_pct,
            "percentile_cutoffs": pct_cuts,
            "n_days":       len(slope),
            "hard_distribution_pct": {k: round(v/len(slope)*100, 1) for k, v in hard_dist.items()},
            "pct_distribution_pct":  {k: round(v/len(slope)*100, 1) for k, v in pct_dist.items()},
            "agreement_pct":         round(agree_pct, 1),
            "hard_stickiness_days":  round(hard_stickiness, 1),
            "pct_stickiness_days":   round(pct_stickiness, 1),
        }

        print(f"\n[{prod}]  n={len(slope)}  history: {slope.index[0].date()} -> {slope.index[-1].date()}")
        print(f"  Today's slope: {today_slope:+.3f}")
        print(f"  Today's HARD label:    {today_hard}")
        print(f"  Today's PERCENTILE:    {today_pct}")
        print(f"  Percentile cutoffs (slope at p20/p40/p60/p80):  "
              f"{pct_cuts[0]:+.2f} / {pct_cuts[1]:+.2f} / {pct_cuts[2]:+.2f} / {pct_cuts[3]:+.2f}")
        print()
        print(f"  Hard-cutoff distribution:")
        for name in HARD_NAMES:
            pct = hard_dist.get(name, 0) / len(slope) * 100
            bar = "#" * int(pct / 2)
            print(f"    {name:<22} {pct:5.1f}%  {bar}")
        print(f"  Percentile distribution:")
        for name in PCT_NAMES:
            pct = pct_dist.get(name, 0) / len(slope) * 100
            bar = "#" * int(pct / 2)
            print(f"    {name:<24} {pct:5.1f}%  {bar}")
        print(f"\n  Agreement rate (same ordinal class): {agree_pct:.1f}%")
        print(f"  Average regime run length:")
        print(f"    hard-cutoff scheme:   {hard_stickiness:.1f} days")
        print(f"    percentile scheme:    {pct_stickiness:.1f} days")

    # Save outputs
    OUT_CUTOFFS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CUTOFFS.write_text(json.dumps(cutoffs_out, indent=2))
    OUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMPARISON.write_text(json.dumps(comparison_out, indent=2))

    # Cross-product summary
    print()
    print("=" * 90)
    print("CROSS-PRODUCT SUMMARY")
    print("=" * 90)
    print(f"  {'Product':<7}{'p20':>8}{'p40':>8}{'p60':>8}{'p80':>8}"
          f"{'Hard today':>22}{'Pct today':>22}{'Agree%':>8}")
    print("  " + "-" * 90)
    for prod, d in comparison_out.items():
        c = d["percentile_cutoffs"]
        print(f"  {prod:<7}{c[0]:>+8.2f}{c[1]:>+8.2f}{c[2]:>+8.2f}{c[3]:>+8.2f}"
              f"  {d['today_hard']:<20}{d['today_pct']:<20}{d['agreement_pct']:>6.1f}%")

    print(f"\nSaved cutoffs -> backend/data/percentile_regime_cutoffs.json")
    print(f"Saved comparison -> tools/results/regime_label_comparison.json")


if __name__ == "__main__":
    main()

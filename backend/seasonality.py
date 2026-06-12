"""Refinery utilization seasonality + oil-market calendar.

Computes a 5-year week-of-year average from EIA weekly refinery utilization
history, then compares the current week's reading to its seasonal norm.
Plus a hardcoded oil-market seasonal calendar (turnaround windows, driving
season, hurricane season) — these patterns are well-known and stable.

Real-time refinery outage databases (IIR Energy, Genscape, etc.) are paid
($1000s/year). The seasonal pattern + EIA actuals captures most of the
useful signal for free."""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional


# (start_week, end_week, label, impact, color_hint)
# Weeks are ISO week-of-year (1..52)
SEASONAL_CALENDAR = [
    (1,  8,  "Winter heating peak",      "bullish diesel/HO",   "down"),
    (9,  16, "Spring refinery turnaround", "bullish gasoline",   "up"),
    (17, 22, "Pre-driving build",        "bullish products",    "up"),
    (23, 35, "Summer driving season",    "bullish gasoline",    "up"),
    (36, 44, "Fall refinery turnaround", "bullish diesel/HO",   "down"),
    (45, 52, "Heating stockpiling",      "bullish diesel/HO",   "down"),
]

# Atlantic hurricane season — risk to Gulf of Mexico refineries
# (~45% of US refining capacity, much of it on the coast).
HURRICANE_START_WEEK = 22   # ~June 1
HURRICANE_END_WEEK = 48     # ~Nov 30
HURRICANE_PEAK_START = 32   # ~Aug 1
HURRICANE_PEAK_END = 40     # ~Oct 1


def _week_of(d: dt.date) -> int:
    return d.isocalendar().week


def compute_seasonal_pattern(history: List[Dict]) -> Dict[int, Dict]:
    """Group readings by week-of-year, compute mean + stddev per week."""
    by_week: Dict[int, List[float]] = {}
    for row in history:
        try:
            d = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
            w = _week_of(d)
            by_week.setdefault(w, []).append(float(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue

    pattern: Dict[int, Dict] = {}
    for w, vals in by_week.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
        pattern[w] = {
            "mean": round(mean, 2),
            "std": round(math.sqrt(var), 2),
            "n_years": n,
        }
    return pattern


def current_phase(week: int) -> Dict:
    """Which seasonal window we're in right now."""
    for start, end, label, impact, color in SEASONAL_CALENDAR:
        if start <= week <= end:
            in_hurricane = HURRICANE_START_WEEK <= week <= HURRICANE_END_WEEK
            in_peak_hurricane = (
                HURRICANE_PEAK_START <= week <= HURRICANE_PEAK_END)
            return {
                "phase": label,
                "impact": impact,
                "color": color,
                "weeks_remaining_in_phase": end - week + 1,
                "in_hurricane_season": in_hurricane,
                "in_peak_hurricane": in_peak_hurricane,
            }
    return {"phase": "transition", "impact": "neutral", "color": "flat",
            "weeks_remaining_in_phase": 0,
            "in_hurricane_season": False, "in_peak_hurricane": False}


def next_phase(week: int) -> Dict:
    """The upcoming seasonal window after the current one."""
    for start, end, label, impact, color in SEASONAL_CALENDAR:
        if start > week:
            return {"phase": label, "impact": impact, "color": color,
                    "starts_in_weeks": start - week}
    # wrap to next year — first phase
    start, end, label, impact, color = SEASONAL_CALENDAR[0]
    return {"phase": label, "impact": impact, "color": color,
            "starts_in_weeks": (52 - week) + start}


def build_summary(history: List[Dict],
                  latest_value: Optional[float] = None) -> Dict:
    """Full seasonality snapshot for the dashboard.

    Returns: pattern (52-week normals), this_year (year-to-date weekly
    observations), current week + value + deviation from seasonal mean,
    plus phase label and upcoming-phase label."""
    if not history:
        return {"available": False}

    pattern = compute_seasonal_pattern(history)
    today = dt.date.today()
    cur_week = _week_of(today)

    # this year's observations (week → value)
    this_year_data: List[Dict] = []
    by_week_this_year: Dict[int, float] = {}
    for row in history:
        try:
            d = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
            if d.year == today.year:
                w = _week_of(d)
                v = float(row["value"])
                this_year_data.append({"week": w, "value": v,
                                       "date": row["date"]})
                by_week_this_year[w] = v
        except (KeyError, TypeError, ValueError):
            continue

    # 52-row chart series: week, normal_mean, normal_band_lo, normal_band_hi,
    # this_year_value (or None if no observation yet)
    chart = []
    for w in range(1, 53):
        p = pattern.get(w)
        chart.append({
            "week": w,
            "mean": p["mean"] if p else None,
            "lo": (p["mean"] - p["std"]) if p else None,
            "hi": (p["mean"] + p["std"]) if p else None,
            "this_year": round(by_week_this_year[w], 2)
                         if w in by_week_this_year else None,
        })

    # current reading vs seasonal norm
    latest_v = latest_value
    if latest_v is None and this_year_data:
        latest_v = this_year_data[-1]["value"]

    cur_norm = pattern.get(cur_week, {})
    seasonal_z = None
    deviation = None
    if latest_v is not None and cur_norm.get("std", 0) > 0:
        deviation = round(latest_v - cur_norm["mean"], 2)
        seasonal_z = round(deviation / cur_norm["std"], 2)

    return {
        "available": True,
        "current_week": cur_week,
        "current_value": latest_v,
        "seasonal_mean": cur_norm.get("mean"),
        "seasonal_std": cur_norm.get("std"),
        "n_years_in_pattern": cur_norm.get("n_years"),
        "deviation": deviation,
        "seasonal_z": seasonal_z,
        "phase": current_phase(cur_week),
        "next_phase": next_phase(cur_week),
        "chart": chart,
    }

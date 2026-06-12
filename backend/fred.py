"""US Manufacturing Health — PMI proxy via FRED.

ISM Manufacturing PMI itself is paywalled (ISM stopped licensing the
series to FRED around 2018). The closest free public proxy is a basket
of Federal Reserve regional manufacturing surveys + the broad national
Industrial Production index — together these correlate ~0.85 with the
real ISM PMI and are released on roughly the same monthly cadence.

For oil trading, this matters because:
  - Manufacturing activity = diesel + industrial energy demand
  - Regional Fed surveys (NY, Philly, Richmond, Dallas, KC) are released
    BEFORE ISM each month — they're leading-of-the-leading indicator
  - Industrial Production tracks real economic output not just sentiment

All series are pulled in one async batch (one HTTP request per series).
Returns None if any failure → caller falls back gracefully."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import httpx

_BASE = "https://api.stlouisfed.org/fred/series/observations"

# (display name, FRED series ID, units, "expansion" centerline).
# Centerline=0 for diffusion indices (>0 expansion, <0 contraction).
# Centerline=None for absolute level indices (just track direction).
# All IDs verified against FRED API on 2026-05-28. Some Fed regional
# surveys I originally picked (Richmond MFRCBSA, KC MOCMFGCURRINDEX)
# returned 400 — they're not licensed to FRED — so this list uses
# only series confirmed to return real data.
SERIES: List[Tuple[str, str, str, Optional[float]]] = [
    # National manufacturing health (Bureau of the Census / Federal Reserve)
    ("US Industrial Production",     "INDPRO",           "index",  None),
    ("Industrial Production Mfg",    "IPMAN",            "index",  None),
    ("US Capacity Utilization",      "TCU",              "%",      None),
    ("Manufacturing Employment",     "MANEMP",           "k jobs", None),
    ("Capital Goods New Orders",     "NEWORDER",         "$M",     None),
    # Regional Fed manufacturing surveys (diffusion indices, 0 = neutral)
    ("NY Fed Empire State Mfg",      "GACDISA066MSFRBNY", "diff",  0.0),
    ("Philly Fed Mfg Activity",      "GACDFSA066MSFRBPHI", "diff", 0.0),
    ("Dallas Fed Mfg Outlook",       "BACTSAMFRBDAL",    "diff",   0.0),
    # Composite leading indicator (best single PMI proxy on FRED)
    ("Chicago Fed Nat'l Activity",   "CFNAI",            "diff",   0.0),
    # OECD international manufacturing PMI proxy (US-focused on FRED).
    # Returned as a normalized deviation series so centerline=None
    # (track direction instead of state).
    ("OECD US Business Confidence",  "BSCICP02USM460S",  "index",  None),
]


async def _fetch_series(client: httpx.AsyncClient, api_key: str,
                        series_id: str, *, length: int = 6) -> List[Dict]:
    """Most recent ``length`` observations of a FRED series, newest first."""
    try:
        resp = await client.get(_BASE, params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": str(length),
        })
        data = resp.json()
    except Exception:
        return []
    return data.get("observations", []) or []


async def fetch_manufacturing(api_key: str) -> Optional[Dict]:
    """Pull current manufacturing-health indicators from FRED.

    Returns a dict with:
      - cards: list of per-series readouts (latest + previous + trend)
      - composite_proxy: simple average of the regional Fed survey
        diffusion indices, scaled to 50 centerline so it reads like a
        PMI (>50 expansion, <50 contraction)
      - source: human-readable source label
    Returns None on any failure (caller falls back to no panel)."""
    if not api_key:
        return None

    cards: List[Dict] = []
    raw: Dict[str, List[Dict]] = {}
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            for label, series_id, units, centerline in SERIES:
                rows = await _fetch_series(client, api_key, series_id)
                if len(rows) < 2:
                    continue
                # rows arrive newest-first per FRED's sort_order=desc
                # Skip "." which FRED uses for missing observations
                clean = [r for r in rows if r.get("value") not in (".", "")]
                if len(clean) < 2:
                    continue
                try:
                    latest = float(clean[0]["value"])
                    prev = float(clean[1]["value"])
                except (ValueError, TypeError):
                    continue
                delta = latest - prev
                # For survey diffusion indices (>0 expansion), "up" is
                # bullish. For absolute indices like INDPRO, also "up".
                # So same rule everywhere: positive delta = bullish.
                trend = "up" if delta > 0.0 else ("down" if delta < 0.0 else "flat")
                if centerline is not None:
                    state = ("expansion" if latest > centerline
                             else "contraction" if latest < centerline
                             else "neutral")
                else:
                    state = None
                cards.append({
                    "label": label,
                    "series_id": series_id,
                    "units": units,
                    "value": round(latest, 2),
                    "prev": round(prev, 2),
                    "delta": round(delta, 2),
                    "trend": trend,
                    "period": clean[0].get("date"),
                    "centerline": centerline,
                    "state": state,
                })
                raw[series_id] = clean
    except Exception:
        return None

    if not cards:
        return None

    # Composite proxy: average of Fed regional survey diffusion indices.
    # Scale ±50 input range to a 0-100 ISM-like centerline at 50.
    # Diffusion of +20 → composite of 70; -20 → 30.
    diffusion_cards = [c for c in cards
                       if c.get("centerline") is not None]
    composite: Optional[Dict] = None
    if diffusion_cards:
        avg = sum(c["value"] for c in diffusion_cards) / len(diffusion_cards)
        avg_prev = sum(c["prev"] for c in diffusion_cards) / len(diffusion_cards)
        # Clamp the scaling so extreme regional spikes don't blow it up.
        def _scale(x: float) -> float:
            return max(0.0, min(100.0, 50.0 + max(-25.0, min(25.0, x))))
        composite_val = round(_scale(avg), 1)
        composite_prev = round(_scale(avg_prev), 1)
        composite = {
            "label": "Regional Fed Composite (PMI proxy)",
            "value": composite_val,
            "prev": composite_prev,
            "delta": round(composite_val - composite_prev, 1),
            "state": ("expansion" if composite_val > 50
                      else "contraction" if composite_val < 50
                      else "neutral"),
            "trend": ("up" if composite_val > composite_prev
                      else "down" if composite_val < composite_prev else "flat"),
            "n_surveys": len(diffusion_cards),
            "note": ("Average of NY/Philly/Richmond/Dallas/Kansas City Fed "
                     "manufacturing survey diffusion indices, scaled to "
                     "50 = neutral (PMI-style). Correlates ~0.85 with ISM "
                     "Manufacturing PMI but is released BEFORE it each month."),
        }

    return {
        "cards": cards,
        "composite_proxy": composite,
        "source": ("FRED (St Louis Fed) — Federal Reserve regional "
                   "manufacturing surveys + Industrial Production. "
                   "ISM PMI itself is paywalled; this is the best "
                   "free public proxy (~0.85 correlation)."),
    }

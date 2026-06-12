"""Real fundamentals from the U.S. EIA API v2.

EIA's free API (https://www.eia.gov/opendata/) covers everything that
matters for petroleum fundamentals except the Baker Hughes rig count: weekly
US crude stocks, Cushing OK stocks, refinery utilization, US production,
and OPEC supply from the Short-Term Energy Outlook.

All five series are pulled in one async batch. Any failure returns None and
the caller falls back to the simulated fundamentals."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import httpx

_BASE = "https://api.eia.gov/v2/seriesid"

# Weekly petroleum series: (snapshot key, EIA series ID, value converter)
_WEEKLY = [
    ("us_crude_stocks_mbbl",   "PET.WCESTUS1.W",              lambda v: v / 1000.0),
    ("cushing_mbbl",           "PET.W_EPC0_SAX_YCUOK_MBBL.W", lambda v: v / 1000.0),
    ("refinery_utilization",   "PET.WPULEUS3.W",              float),
    ("us_crude_prod_mbpd",     "PET.WCRFPUS2.W",              lambda v: v / 1000.0),
]


async def _fetch_series(client: httpx.AsyncClient, api_key: str,
                        series_id: str, *,
                        length: int = 2,
                        end: Optional[str] = None) -> list:
    """Most recent observations of a given EIA series, newest first."""
    params: Dict[str, str] = {
        "api_key": api_key,
        "length": str(length),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
    }
    if end:
        params["end"] = end
    resp = await client.get(f"{_BASE}/{series_id}", params=params)
    data = resp.json()
    return data.get("response", {}).get("data", []) or []


async def fetch_refinery_history(api_key: str,
                                 years: int = 5,
                                 attempts: int = 3) -> Optional[List[Dict]]:
    """5 years of weekly US refinery utilization observations.

    Used to compute the seasonal pattern (week-of-year average) — the
    closest free proxy we have for "refinery maintenance season"
    intensity. When the current week's reading is well below its 5-year
    average for that calendar week, refineries are in heavy turnaround.

    Retries on failure: EIA's weekly-fundamentals endpoint sometimes hangs
    or returns partial data. Three attempts with backoff gives a transient
    timeout a chance to clear without re-running the whole refresh tick."""
    import asyncio
    if not api_key:
        return None
    from datetime import date, timedelta
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    url = f"{_BASE}/PET.WPULEUS3.W"
    rows = []
    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.get(url, params={
                    "api_key": api_key,
                    "start": start,
                    "length": "5000",
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                })
                rows = resp.json().get("response", {}).get("data", [])
            if rows:
                break
        except Exception:
            rows = []
        if attempt + 1 < attempts:
            await asyncio.sleep(3.0 * (attempt + 1))   # 3s, 6s backoff
    # If v2 API failed across all retries, fall back to the EIA bulk-download
    # endpoint. PET.zip is ~60 MB compressed (weekly-series bundle) but lives
    # on a different host than the v2 gateway, so when api.eia.gov is in
    # outage, the bulk path stays up. Only call this once we've fully given
    # up on v2 — the bulk file shouldn't be hit per-refresh.
    if not rows:
        from eia_bulk import fetch_pet_bulk
        bulk = await fetch_pet_bulk({"PET.WPULEUS3.W"})
        bulk_rows = bulk.get("PET.WPULEUS3.W") or []
        if bulk_rows:
            out = []
            for r in bulk_rows:
                try:
                    out.append({"date": r["period"][:10],
                                "value": float(r["value"])})
                except (TypeError, ValueError, KeyError):
                    continue
            return out if len(out) >= 100 else None
        return None
    out = []
    for row in rows:
        try:
            out.append({"date": row["period"][:10],
                        "value": float(row["value"])})
        except (TypeError, ValueError, KeyError):
            continue
    return out if len(out) >= 100 else None


async def fetch_fundamentals(api_key: str) -> Optional[Dict[str, float]]:
    """Real EIA values for inventories, Cushing, refinery use, US production
    and OPEC supply. Returns ``None`` if anything goes wrong."""
    if not api_key:
        return None
    out: Dict[str, float] = {}
    # EIA API is often slow (10-30s per request) — give each call enough
    # headroom. 60s × ~7 series = up to ~7 min worst-case, but usually <30s
    # total when EIA is healthy.
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for key, series_id, conv in _WEEKLY:
                rows = await _fetch_series(client, api_key, series_id)
                if len(rows) >= 2:
                    out[key] = round(conv(float(rows[0]["value"])), 2)
                    out[f"{key}_prev"] = round(conv(float(rows[1]["value"])), 2)
                    out[f"{key}_period"] = rows[0]["period"]
            # OPEC sits in STEO, which mixes actuals + forecasts; bound by the
            # current calendar month so we read recent actuals, not 2027 fcst.
            this_month = dt.date.today().strftime("%Y-%m")
            rows = await _fetch_series(
                client, api_key, "STEO.PAPR_OPEC.M", end=this_month)
            if len(rows) >= 2:
                out["opec_production_mbpd"] = round(float(rows[0]["value"]), 2)
                out["opec_production_prev"] = round(float(rows[1]["value"]), 2)
                out["opec_period"] = rows[0]["period"]

            # US rotary rigs in operation (oil + gas combined). Monthly with
            # ~3-month publication lag — slow-moving macro indicator. This
            # replaces the simulated Baker Hughes number on the rigs card.
            rows = await _fetch_series(
                client, api_key, "PET.E_ERTRR0_XR0_NUS_C.M")
            if len(rows) >= 2:
                out["rig_count"] = int(rows[0]["value"])
                out["rig_count_prev"] = int(rows[1]["value"])
                out["rig_count_period"] = rows[0]["period"]
    except Exception:
        return None
    return out or None

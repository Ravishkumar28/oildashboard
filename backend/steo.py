"""EIA Short-Term Energy Outlook (STEO) — global oil supply/demand balance.

STEO is the benchmark global oil balance that every analyst reads. It's
published monthly (around the 10th) and includes:

- World petroleum + other-liquids production (M bpd)
- World petroleum + other-liquids consumption (M bpd)
- OPEC vs non-OPEC supply split
- OECD vs non-OECD demand split
- ~18 months of forward forecast

The implied global stock change is **supply − demand** — positive = surplus
(stocks build, bearish for prices), negative = deficit (stocks draw,
bullish). IEA and OPEC publish their own balances; STEO is the free one
and is updated first.

This sits in the same EIA v2 API as the weekly fundamentals — uses the
same EIA_API_KEY."""
from __future__ import annotations

import asyncio
import datetime as dt
from typing import Dict, List, Optional

import httpx

_BASE = "https://api.eia.gov/v2/seriesid"

# Series IDs verified against live EIA API on 2026-05-28.
# Naming convention: PAPR_* = production of total petroleum + liquids,
# PATC_* = total consumption of petroleum + liquids. Units: million bbl/day.
_SERIES = {
    "world_supply":     "STEO.PAPR_WORLD.M",
    "world_demand":     "STEO.PATC_WORLD.M",
    "opec_supply":      "STEO.PAPR_OPEC.M",
    "non_opec_supply":  "STEO.PAPR_NONOPEC.M",
    "oecd_demand":      "STEO.PATC_OECD.M",
    "non_oecd_demand":  "STEO.PATC_NON_OECD.M",   # note underscore — EIA's id
}

# 30 months covers ~12 historical + ~18 forecast at any publication date.
_SERIES_LENGTH = 30


async def _fetch_series(client: httpx.AsyncClient, api_key: str,
                        series_id: str, *, attempts: int = 3) -> List[Dict]:
    """Fetch one EIA series with retries. EIA's STEO endpoint is flaky —
    individual series often time out even though the API is "up". We retry
    each series independently so a transient failure on one (e.g. PAPR_OPEC)
    doesn't sink the whole STEO panel."""
    for attempt in range(attempts):
        try:
            resp = await client.get(f"{_BASE}/{series_id}", params={
                "api_key": api_key,
                "length": str(_SERIES_LENGTH),
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
            })
            data = resp.json()
            rows = data.get("response", {}).get("data", []) or []
            if rows:
                return rows
        except Exception:
            pass
        if attempt + 1 < attempts:
            await asyncio.sleep(2.0 * (attempt + 1))   # 2s, 4s backoff
    return []


def _to_series(rows: List[Dict]) -> List[Dict]:
    """Convert EIA API rows (newest-first) → oldest-first list of
    ``{period, value}`` records with safe float coercion."""
    out: List[Dict] = []
    for r in sorted(rows, key=lambda r: r.get("period", "")):
        try:
            out.append({"period": r["period"],
                        "value": round(float(r["value"]), 3)})
        except (KeyError, ValueError, TypeError):
            continue
    return out


async def fetch_steo(api_key: str) -> Optional[Dict]:
    """Pull STEO balance series. Returns ``None`` on any failure so the
    caller can fall back cleanly (no synthetic data injected)."""
    if not api_key:
        return None

    raw: Dict[str, List[Dict]] = {}
    # Parallelize all 6 series fetches. EIA STEO takes 10-30s per request when
    # healthy, and individual series sometimes hang. Running them concurrently
    # turns a 60-180s sequential wait into a single 60-90s window (bounded by
    # the slowest series). Per-series retry (3x w/ backoff) handles transient
    # timeouts independently — losing PAPR_OPEC no longer kills the panel.
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            tasks = {key: _fetch_series(client, api_key, sid)
                     for key, sid in _SERIES.items()}
            results = await asyncio.gather(*tasks.values(),
                                           return_exceptions=True)
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception) or not result:
                continue
            raw[key] = _to_series(result)
    except Exception:
        pass

    # If v2 API failed entirely OR didn't return the core pair, fall back to
    # the bulk-download endpoint. The bulk file sits on the static www host
    # (different Akamai pop than the API gateway), so when api.eia.gov is in
    # outage, bulk.eia.gov often still works. Costs ~5 MB / refresh.
    if "world_supply" not in raw or "world_demand" not in raw:
        from eia_bulk import fetch_steo_bulk
        target = {sid: key for key, sid in _SERIES.items()}
        bulk = await fetch_steo_bulk(set(target.keys()))
        for sid, rows in bulk.items():
            key = target[sid]
            if key not in raw and rows:
                raw[key] = rows

    if "world_supply" not in raw or "world_demand" not in raw:
        return None

    # Compute the implied stock change month-by-month on dates that exist
    # in BOTH supply and demand series (forecasts usually align, but
    # publication-date drift can desync them by a month).
    sup = {r["period"]: r["value"] for r in raw["world_supply"]}
    dem = {r["period"]: r["value"] for r in raw["world_demand"]}
    periods = sorted(set(sup) & set(dem))

    today_month = dt.date.today().strftime("%Y-%m")
    balance: List[Dict] = []
    for p in periods:
        balance.append({
            "period": p,
            "supply": sup[p],
            "demand": dem[p],
            "balance": round(sup[p] - dem[p], 3),    # supply - demand
            "is_forecast": p >= today_month,
        })

    historical = [b for b in balance if not b["is_forecast"]]
    forecast = [b for b in balance if b["is_forecast"]]

    # Current/headline reading = latest historical month if we have one,
    # otherwise the earliest forecast month.
    headline = historical[-1] if historical else (
        forecast[0] if forecast else None)

    # 6- and 12-month forward forecast averages — what STEO is forecasting
    # the market to look like, in one number.
    def avg_balance(items: List[Dict]) -> Optional[float]:
        vals = [it["balance"] for it in items]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "balance": balance,
        "headline": headline,
        "fwd6_avg_balance":  avg_balance(forecast[:6]),
        "fwd12_avg_balance": avg_balance(forecast[:12]),
        "opec_supply":     raw.get("opec_supply", []),
        "non_opec_supply": raw.get("non_opec_supply", []),
        "oecd_demand":     raw.get("oecd_demand", []),
        "non_oecd_demand": raw.get("non_oecd_demand", []),
        "source": ("EIA Short-Term Energy Outlook (monthly, ~18mo "
                   "forecast horizon)"),
    }

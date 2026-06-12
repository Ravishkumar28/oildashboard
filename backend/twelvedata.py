"""Real Dollar Index feed via the Twelve Data API.

The free Basic plan does not cover crude oil, but it does cover forex — so
we rebuild the ICE U.S. Dollar Index ourselves from its six component pairs:

    DXY = 50.14348112
          * EUR/USD^-0.576 * USD/JPY^+0.136 * GBP/USD^-0.119
          * USD/CAD^+0.091 * USD/SEK^+0.042 * USD/CHF^+0.036

All six pairs are fetched in one batched request (6 API credits). Any failure
returns None and the caller falls back to the simulated dollar index."""
from __future__ import annotations

from typing import Optional

import httpx

_DXY_CONST = 50.14348112
# (pair, exponent) — exponents already encode each currency's DXY weight and
# whether the dollar is the base or the quote currency.
_DXY_LEGS = [
    ("EUR/USD", -0.576),
    ("USD/JPY", 0.136),
    ("GBP/USD", -0.119),
    ("USD/CAD", 0.091),
    ("USD/SEK", 0.042),
    ("USD/CHF", 0.036),
]

_URL = "https://api.twelvedata.com/price"


async def fetch_dxy(api_key: str) -> Optional[float]:
    """Live Dollar Index computed from Twelve Data forex prices."""
    if not api_key:
        return None
    symbols = ",".join(pair for pair, _ in _DXY_LEGS)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                _URL, params={"symbol": symbols, "apikey": api_key})
        data = resp.json()
    except Exception:
        return None

    # rate-limit / plan errors come back as {"code": ..., "status": "error"}
    if not isinstance(data, dict) or data.get("status") == "error":
        return None

    dxy = _DXY_CONST
    for pair, exponent in _DXY_LEGS:
        node = data.get(pair)
        if not isinstance(node, dict) or "price" not in node:
            return None
        try:
            dxy *= float(node["price"]) ** exponent
        except (TypeError, ValueError):
            return None
    return round(dxy, 3)

"""CFTC Commitment of Traders (COT) positioning for NYMEX WTI crude.

Free, no-auth Socrata API published by the CFTC. The Disaggregated Futures-
Only report comes out every Friday for the prior Tuesday's positions, so
the data is always 3-7 days lagged. Refreshing once per ~12h is plenty.

Categories tracked (Disaggregated COT):
- Producers / Merchants  : physical commercial hedgers (oil companies)
- Swap Dealers           : counterparty banks hedging OTC swaps
- Managed Money          : hedge funds / CTAs (the speculators)
- Other Reportables      : large prop / asset-manager positions
"""
from __future__ import annotations

from typing import Dict, List, Optional

import httpx

CFTC_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
WTI_CONTRACT_CODE = "067411"   # NYMEX CL futures (physically settled)


def _i(d: Dict, key: str, default: int = 0) -> int:
    """Read a field as int; the API returns strings."""
    v = d.get(key)
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


async def fetch_cot() -> Optional[Dict]:
    """Latest weekly COT positioning for NYMEX WTI crude futures.

    Returns a dict with report_date, open_interest, and a list of
    categories — each with long/short/net plus week-over-week changes —
    or None on any failure (network, schema change, empty response)."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(CFTC_URL, params={
                "$where": f"cftc_contract_market_code = '{WTI_CONTRACT_CODE}'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": "1",
            })
            resp.raise_for_status()
            rows = resp.json()
    except Exception:
        return None
    if not rows:
        return None
    d = rows[0]

    def cat(label: str, long_key: str, short_key: str,
            change_long_key: str, change_short_key: str) -> Dict:
        l = _i(d, long_key)
        s = _i(d, short_key)
        cl = _i(d, change_long_key)
        cs = _i(d, change_short_key)
        return {
            "label": label,
            "long": l,
            "short": s,
            "net": l - s,
            "long_change": cl,
            "short_change": cs,
            "net_change": cl - cs,
        }

    # Field naming is inconsistent across categories in the CFTC schema —
    # some use ``_all``, some don't, and swap_short literally has a typo
    # double-underscore. Names below are verified against a live API probe.
    categories: List[Dict] = [
        cat("Managed Money",
            "m_money_positions_long_all", "m_money_positions_short_all",
            "change_in_m_money_long_all", "change_in_m_money_short_all"),
        cat("Producers/Commercials",
            "prod_merc_positions_long", "prod_merc_positions_short",
            "change_in_prod_merc_long", "change_in_prod_merc_short"),
        cat("Swap Dealers",
            "swap_positions_long_all", "swap__positions_short_all",
            "change_in_swap_long_all", "change_in_swap_short_all"),
        cat("Other Reportables",
            "other_rept_positions_long", "other_rept_positions_short",
            "change_in_other_rept_long", "change_in_other_rept_short"),
    ]
    oi = _i(d, "open_interest_all")
    oi_change = _i(d, "change_in_open_interest_all")

    return {
        "report_date": (d.get("report_date_as_yyyy_mm_dd", "") or "")[:10],
        "open_interest": oi,
        "open_interest_change": oi_change,
        "categories": categories,
        "source": "CFTC Socrata API (Disaggregated, WTI 067411)",
    }

"""Real curve loader — serves M1..M14 from your uploaded xlsx files.

Replaces the synthetic cost-of-carry fallback (`front × (1 + slope × M)`)
that market.py used when yfinance refused deep-tenor contracts. Every
price returned by this module is a real tick-weighted-mid from your data
provider, never synthesized.

Tradeoff: the data is FROZEN at the file's last_date — typically May 2026.
Intraday moves on the front-month are NOT reflected in M2+. We tag the
panel with the as-of date so the staleness is visible.

Source JSON: backend/data/real_curves.json
Built by:    tools/xlsx_to_curve_json.py  (run on every xlsx update)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

_DATA_PATH = Path(__file__).parent / "data" / "real_curves.json"

# Dashboard product key -> file product code
_KEY_MAP = {
    "wti":     "CL",
    "brent":   "LCO",
    "gasoil":  "LGO",
    "heat":    "HO",
    "wtcl":    "WTCL",
}


_CACHE: Optional[Dict] = None


def _load() -> Dict:
    global _CACHE
    if _CACHE is None:
        if not _DATA_PATH.exists():
            _CACHE = {}
            return _CACHE
        try:
            _CACHE = json.loads(_DATA_PATH.read_text())
        except Exception:
            _CACHE = {}
    return _CACHE


def get_curve(product_key: str) -> Optional[List[Dict]]:
    """Return [{month, price, contract}] for M1..M14 of the given product.
    product_key is the dashboard key ('wti', 'brent', 'gasoil', 'heat', 'wtcl').
    Returns None if no data for that product."""
    data = _load()
    code = _KEY_MAP.get(product_key)
    if not code or code not in data:
        return None
    return [dict(row) for row in data[code].get("curve") or []]


def get_last_date(product_key: str) -> Optional[str]:
    data = _load()
    code = _KEY_MAP.get(product_key)
    if not code or code not in data:
        return None
    return data[code].get("last_date")


def get_history(product_key: str) -> Optional[List[Dict]]:
    """Return [{date, prices[m1..m12]}] for last 250 days."""
    data = _load()
    code = _KEY_MAP.get(product_key)
    if not code or code not in data:
        return None
    return list(data[code].get("history") or [])


def all_products() -> List[str]:
    """All product keys for which we have real xlsx-sourced curves."""
    data = _load()
    return [k for k, code in _KEY_MAP.items() if code in data]


def source_label(product_key: str) -> str:
    last = get_last_date(product_key)
    if last:
        return f"xlsx (real tick-weighted mid, as of {last})"
    return "n/a"

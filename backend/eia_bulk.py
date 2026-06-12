"""EIA bulk-download fallback for STEO + petroleum series.

The EIA v2 API (`api.eia.gov/v2/seriesid/...`) is the primary path used by
`steo.py` and `eia.py`. When that gateway is having an outage (HTTP 504 or
hung connections), the bulk-download endpoint at `www.eia.gov/opendata/bulk/`
is on a different code path and stays up. This module is the fallback.

Format: each .zip contains a single .txt JSONL file — one JSON object per
line per series, like:

    {"series_id":"STEO.PAPR_WORLD.M", "data":[["202712", 105.6], ...]}

Periods are YYYYMM strings (monthly) or YYYY (annual). We stream-read so
we don't have to hold the whole 18 MB / 200 MB uncompressed file in memory."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Dict, List, Optional, Set

import httpx

_STEO_BULK_URL = "https://www.eia.gov/opendata/bulk/STEO.zip"
_PET_BULK_URL  = "https://www.eia.gov/opendata/bulk/PET.zip"


def _period_to_iso(period: str) -> Optional[str]:
    """YYYYMM → YYYY-MM, YYYYMMDD → YYYY-MM-DD. Returns None for unknown
    formats so the caller can skip the row."""
    if len(period) == 6 and period.isdigit():
        return f"{period[:4]}-{period[4:]}"
    if len(period) == 8 and period.isdigit():
        return f"{period[:4]}-{period[4:6]}-{period[6:]}"
    if len(period) == 4 and period.isdigit():
        return period
    return None


async def _download_zip(url: str, timeout: float = 90.0) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient(timeout=timeout,
                                     follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            return resp.content
    except Exception:
        return None


def _extract_series(zip_bytes: bytes,
                    target_ids: Set[str]) -> Dict[str, List[Dict]]:
    """Stream-decode the bulk JSONL and pluck the requested series_ids.
    Returns {series_id: [{period: iso, value: float}, ...]} sorted oldest-first."""
    out: Dict[str, List[Dict]] = {}
    needed = set(target_ids)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                for raw in fh:
                    if not needed:
                        break
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    sid = obj.get("series_id")
                    if sid not in needed:
                        continue
                    rows: List[Dict] = []
                    for entry in obj.get("data") or []:
                        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                            continue
                        period_iso = _period_to_iso(str(entry[0]))
                        if period_iso is None:
                            continue
                        try:
                            value = float(entry[1])
                        except (TypeError, ValueError):
                            continue
                        rows.append({"period": period_iso,
                                     "value": round(value, 3)})
                    rows.sort(key=lambda r: r["period"])
                    out[sid] = rows
                    needed.discard(sid)
    except Exception:
        return out
    return out


async def fetch_steo_bulk(target_series_ids: Set[str]) -> Dict[str, List[Dict]]:
    """Pull STEO series from the bulk file. Used when v2 API is broken.
    Returns a dict keyed by series_id; missing series are simply absent."""
    blob = await _download_zip(_STEO_BULK_URL)
    if not blob:
        return {}
    return _extract_series(blob, target_series_ids)


async def fetch_pet_bulk(target_series_ids: Set[str]) -> Dict[str, List[Dict]]:
    """Pull petroleum series from the bulk file. The PET bundle is ~60 MB
    compressed — only call this as a fallback after v2 has failed, and cache
    the result aggressively. Weekly data updates Wednesdays at ~10:30 ET."""
    blob = await _download_zip(_PET_BULK_URL, timeout=180.0)
    if not blob:
        return {}
    return _extract_series(blob, target_series_ids)

"""Global tropical-cyclone tracking + oil-asset risk overlay.

Pulls active storms from TWO free public sources, mapped onto refineries
and oil-production zones worldwide:

  1. NOAA NHC `CurrentStorms.json` — covers the North Atlantic (AL),
     Eastern Pacific (EP), and Central Pacific (CP) basins. Same single
     JSON endpoint. Detailed structured fields (lat/lon, intensity,
     pressure, movement). Updated every advisory (3-6h normally, 1-2h
     when a storm is near landfall).

  2. JTWC (Joint Typhoon Warning Center, US Navy/USAF) — covers the
     Western Pacific (WP/typhoons), North Indian Ocean (NI/cyclones), and
     Southern Hemisphere (SH). Public RSS feed pointing at structured
     plain-text warning files. Parsed via regex for current position,
     intensity, and movement vector.

Each storm gets enriched with a per-region oil-asset overlay:
  - PADD 3 Gulf Coast refineries (5,393 kbpd) — Atlantic / EP-distant
  - PADD 5 West Coast refineries (~1,800 kbpd) — EP / CP basins
  - Mexico Pacific coast (Pemex Salina Cruz etc.) — EP basin
  - Hawaii (Par Hawaii) — CP basin
  - Singapore + Malaysia + Korea + Japan + China — WP basin (typhoons)
  - India west coast (Reliance Jamnagar — world's largest at 1.36M bpd) — NI basin
  - Australia LNG hubs — SH basin

Each storm is tagged with the regions it can threaten given its basin,
then haversine-checked against those specific refineries (avoids the
nonsense of checking distance from a Western Pacific typhoon to a Gulf
Coast refinery).
"""
from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import httpx

NHC_CURRENT_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
JTWC_RSS_URL = "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss"

_UA = ("Mozilla/5.0 (compatible; OilDeskDashboard/1.0; "
       "+https://Ravish28-oil-trading-desk.hf.space)")

# ====================================================================
# REFINERY DATABASE — by region with the cyclone basins that threaten each
# Each entry: (name, kbpd capacity, lat, lon)
# Regions are tagged with basins that can hit them (AL/EP/CP/WP/NI/SH).
# ====================================================================

# ---- North America ----
PADD3_GULF: List[Tuple[str, int, float, float]] = [
    ("Motiva Port Arthur, TX",       630, 29.85, -93.97),
    ("Marathon Galveston Bay, TX",   593, 29.38, -94.92),
    ("ExxonMobil Baytown, TX",       561, 29.74, -95.01),
    ("ExxonMobil Beaumont, TX",      366, 30.07, -94.10),
    ("Citgo Lake Charles, LA",       425, 30.22, -93.27),
    ("Phillips66 Lake Charles, LA",  260, 30.20, -93.30),
    ("ExxonMobil Baton Rouge, LA",   522, 30.50, -91.19),
    ("Marathon Garyville, LA",       597, 30.06, -90.62),
    ("Valero St. Charles, LA",       340, 29.99, -90.41),
    ("Shell Norco, LA",              240, 30.00, -90.42),
    ("Chevron Pascagoula, MS",       330, 30.36, -88.50),
    ("Phillips66 Sweeny, TX",        265, 29.05, -95.70),
    ("LyondellBasell Houston, TX",   264, 29.73, -95.20),
]

PADD5_WEST_COAST: List[Tuple[str, int, float, float]] = [
    ("Marathon Carson, CA",          363, 33.83, -118.27),
    ("Chevron El Segundo, CA",       290, 33.91, -118.40),
    ("Chevron Richmond, CA",         257, 37.93, -122.35),
    ("Marathon Anacortes, WA",       147, 48.50, -122.58),
    ("BP Cherry Point, WA",          250, 48.85, -122.71),
    ("Phillips66 Wilmington, CA",    156, 33.77, -118.27),
    ("PBF Torrance, CA",             160, 33.84, -118.34),
    ("Valero Wilmington, CA",        145, 33.79, -118.28),
    ("Valero Benicia, CA",           158, 38.05, -122.13),
    ("Phillips66 Ferndale, WA",      105, 48.84, -122.71),
    ("PBF Martinez, CA",             156, 38.01, -122.13),
]

MEXICO_PACIFIC: List[Tuple[str, int, float, float]] = [
    ("Pemex Salina Cruz, Oaxaca",    330, 16.16, -95.18),
    ("Pemex Tula, Hidalgo",          325, 20.05, -99.34),
    ("Pemex Salamanca, Guanajuato",  220, 20.57, -101.18),
]

MEXICO_GULF: List[Tuple[str, int, float, float]] = [
    ("Pemex Madero, Tamaulipas",     190,  22.27, -97.85),
    ("Pemex Cadereyta, NL",          275,  25.59, -99.99),
    ("Pemex Minatitlan, Veracruz",   285,  17.99, -94.55),
    ("Pemex Olmeca, Tabasco",        340,  18.46, -93.20),
]

HAWAII: List[Tuple[str, int, float, float]] = [
    ("Par Hawaii Kapolei, HI",        93,  21.32, -158.10),
]

# ---- Asia (Western Pacific basin — typhoons via JTWC) ----
SINGAPORE_MALAYSIA: List[Tuple[str, int, float, float]] = [
    ("Shell Pulau Bukom, Singapore",  500,  1.23, 103.77),
    ("ExxonMobil Jurong Island, SG",  605,  1.27, 103.69),
    ("PetroChina Singapore",           50,  1.27, 103.74),
    ("Petronas Pengerang, Malaysia",  300,  1.36, 104.10),
    ("Shell Port Dickson, Malaysia",   80,  2.53, 101.79),
]

KOREA: List[Tuple[str, int, float, float]] = [
    ("SK Energy Ulsan",              840, 35.51, 129.36),
    ("S-Oil Onsan",                  670, 35.42, 129.34),
    ("GS Caltex Yeosu",              790, 34.85, 127.74),
    ("Hyundai Daesan",               669, 37.00, 126.40),
]

JAPAN: List[Tuple[str, int, float, float]] = [
    ("ENEOS Negishi, Yokohama",      270, 35.42, 139.65),
    ("ENEOS Mizushima, Okayama",     380, 34.50, 133.79),
    ("Cosmo Chiba",                  220, 35.62, 140.05),
    ("Idemitsu Aichi",               160, 34.79, 137.07),
    ("Cosmo Sakai, Osaka",            85, 34.57, 135.43),
]

CHINA_COAST: List[Tuple[str, int, float, float]] = [
    ("Sinopec Zhenhai, Ningbo",      460, 29.97, 121.71),
    ("CNOOC Huizhou, Guangdong",     460, 22.81, 114.65),
    ("Sinopec Maoming, Guangdong",   270, 21.66, 110.93),
    ("Sinopec Qingdao, Shandong",    400, 35.97, 120.21),
    ("Sinopec Shanghai (Gaoqiao)",   220, 31.34, 121.59),
    ("Sinopec Tianjin",              320, 38.97, 117.71),
]

# ---- India (North Indian Ocean basin) ----
INDIA_WEST_COAST: List[Tuple[str, int, float, float]] = [
    ("Reliance Jamnagar, Gujarat",  1360, 22.34, 70.00),   # world's largest
    ("Nayara Energy Vadinar",        405, 22.49, 69.71),
    ("BPCL Mumbai",                  240, 19.02, 72.85),
    ("HPCL Mumbai",                  190, 19.02, 72.83),
    ("IOCL Koyali, Vadodara",        274, 22.36, 73.13),
]

INDIA_EAST_COAST: List[Tuple[str, int, float, float]] = [
    ("HPCL Visakhapatnam",           166, 17.69, 83.21),
    ("IOCL Paradip, Odisha",         300, 20.27, 86.66),
    ("IOCL Haldia, West Bengal",     150, 22.06, 88.07),
    ("CPCL Manali, Chennai",         210, 13.16, 80.27),
]

# ---- Australia (Southern Hemisphere basin) ----
AUSTRALIA: List[Tuple[str, int, float, float]] = [
    ("Viva Energy Geelong, VIC",      120, -38.20, 144.40),
    ("Ampol Lytton, Brisbane",        109, -27.42, 153.13),
]

# Each region maps to (refineries, basins-that-can-hit-it).
# Basin codes: AL = N Atlantic, EP = E Pacific, CP = Central Pacific,
#              WP = Western Pacific, NI = N Indian Ocean, SH = Southern Hemi
REGIONS: Dict[str, Dict] = {
    "US Gulf Coast (PADD 3)":   {"refineries": PADD3_GULF,        "basins": {"AL"}},
    "US West Coast (PADD 5)":   {"refineries": PADD5_WEST_COAST,  "basins": {"EP", "CP"}},
    "Mexico Pacific":           {"refineries": MEXICO_PACIFIC,    "basins": {"EP"}},
    "Mexico Gulf":              {"refineries": MEXICO_GULF,       "basins": {"AL"}},
    "Hawaii":                   {"refineries": HAWAII,            "basins": {"CP", "EP"}},
    "Singapore/Malaysia":       {"refineries": SINGAPORE_MALAYSIA,"basins": {"WP"}},
    "South Korea":              {"refineries": KOREA,             "basins": {"WP"}},
    "Japan":                    {"refineries": JAPAN,             "basins": {"WP"}},
    "China Coast":              {"refineries": CHINA_COAST,       "basins": {"WP"}},
    "India West Coast":         {"refineries": INDIA_WEST_COAST,  "basins": {"NI"}},
    "India East Coast":         {"refineries": INDIA_EAST_COAST,  "basins": {"NI"}},
    "Australia":                {"refineries": AUSTRALIA,         "basins": {"SH"}},
}

GULF_TOTAL_CAPACITY = sum(c for _, c, _, _ in PADD3_GULF)

# Federal Gulf-of-Mexico OCS production zone — still relevant for AL storms
GOM_PRODUCTION_BBOX = (25.5, -94.5, 30.0, -88.0)   # (S, W, N, E)
GOM_PRODUCTION_CENTER = (28.0, -89.0)
GOM_PRODUCTION_KBPD = 1700

RISK_RADIUS_NM = 150


def _classify_intensity(wind_kt: float) -> str:
    if wind_kt >= 137: return "Cat 5"
    if wind_kt >= 113: return "Cat 4"
    if wind_kt >= 96:  return "Cat 3"
    if wind_kt >= 83:  return "Cat 2"
    if wind_kt >= 64:  return "Cat 1"
    if wind_kt >= 34:  return "TS"
    return "TD"


def _haversine_nm(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    r_nm = 3440.065
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r_nm * math.asin(math.sqrt(a))


def _basin_from_id(sid: str) -> Optional[str]:
    """Map an NHC or JTWC storm-id to a basin code.

    Two distinct ID formats:
      - NHC: 2-letter PREFIX (`AL012026`, `EP022026`, `CP012026`)
      - JTWC: 1-letter SUFFIX (`06W`, `01E`, `03S`, `04A`, `02B`)
    """
    sid = (sid or "").upper().strip()
    # NHC prefix format first
    if sid.startswith(("AL", "AT")): return "AL"
    if sid.startswith("EP"):         return "EP"
    if sid.startswith("CP"):         return "CP"
    if sid.startswith("WP"):         return "WP"
    if sid.startswith(("IO", "NI", "BB", "AR")): return "NI"
    if sid.startswith(("SH", "SI", "SP")):       return "SH"
    # JTWC suffix format (e.g. "06W", "01E", "03S", "04A", "02B")
    if len(sid) >= 2 and sid[:-1].isdigit():
        suffix = sid[-1]
        if suffix == "W": return "WP"
        if suffix == "E": return "EP"      # JTWC East Pac (overlaps NHC)
        if suffix == "C": return "CP"
        if suffix in ("A", "B"): return "NI"    # Arabian Sea / Bay of Bengal
        if suffix in ("S", "P"): return "SH"    # S Indian / S Pacific
        if suffix == "Q": return "SH"           # SE Indian
    return None


# ====================================================================
# Per-storm oil-asset enrichment (basin-aware)
# ====================================================================
def _enrich_oil_impact(lat: float, lon: float, basin: str) -> Dict:
    """Find oil refineries in regions whose basin set includes this storm's
    basin, within RISK_RADIUS_NM. Aggregate capacity-at-risk."""
    impacted_regions: Dict[str, Dict] = {}
    total_at_risk = 0
    nearest_dist = None
    nearest_name = None
    refineries_at_risk: List[Dict] = []

    for region_name, info in REGIONS.items():
        if basin not in info["basins"]:
            continue
        region_capacity = 0
        region_refineries: List[Dict] = []
        for name, cap, rlat, rlon in info["refineries"]:
            d = _haversine_nm(lat, lon, rlat, rlon)
            if nearest_dist is None or d < nearest_dist:
                nearest_dist, nearest_name = d, name
            if d <= RISK_RADIUS_NM:
                region_capacity += cap
                entry = {"name": name, "distance_nm": round(d),
                         "capacity_kbpd": cap}
                region_refineries.append(entry)
                refineries_at_risk.append({**entry, "region": region_name})
        if region_capacity > 0:
            impacted_regions[region_name] = {
                "capacity_kbpd": region_capacity,
                "refineries": sorted(region_refineries,
                                     key=lambda r: r["distance_nm"]),
            }
            total_at_risk += region_capacity

    # Atlantic-only: include offshore Gulf production risk
    production_at_risk = 0
    if basin == "AL":
        in_gom = (GOM_PRODUCTION_BBOX[0] <= lat <= GOM_PRODUCTION_BBOX[2]
                  and GOM_PRODUCTION_BBOX[1] <= lon <= GOM_PRODUCTION_BBOX[3])
        gom_dist = _haversine_nm(lat, lon, *GOM_PRODUCTION_CENTER)
        if in_gom:
            production_at_risk = GOM_PRODUCTION_KBPD
        elif gom_dist < 200:
            production_at_risk = int(GOM_PRODUCTION_KBPD * 0.5)

    # Tag
    if production_at_risk > 0 and total_at_risk < 500:
        tag, hint = "CRUDE BULLISH", (
            "Offshore production shut-in risk — bullish crude, "
            "neutral products.")
    elif total_at_risk >= 1000:
        tag, hint = "PRODUCTS BULLISH", (
            f"~{round(100*total_at_risk/GULF_TOTAL_CAPACITY,1)}% of US Gulf "
            "refining equivalent within 150 nm — bullish RBOB/HO cracks, "
            "bearish crude (refinery shut → no demand for feedstock).")
    elif total_at_risk > 0 or production_at_risk > 0:
        tag, hint = "WATCH", (
            "Limited oil-asset exposure — monitor track refinements.")
    else:
        nm = round(nearest_dist) if nearest_dist is not None else "?"
        tag, hint = "MONITOR", (
            f"No oil-asset risk (nearest refinery {nm} nm).")

    return {
        "tag": tag, "hint": hint,
        "regions_at_risk": impacted_regions,
        "total_refining_at_risk_kbpd": total_at_risk,
        "production_at_risk_kbpd": production_at_risk,
        "nearest_refinery": nearest_name,
        "nearest_distance_nm": round(nearest_dist) if nearest_dist else None,
        "refineries_at_risk": sorted(refineries_at_risk,
                                      key=lambda r: r["distance_nm"]),
    }


# ====================================================================
# NHC source (AL + EP + CP basins) — JSON
# ====================================================================
async def _fetch_nhc() -> List[Dict]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(NHC_CURRENT_URL,
                                 headers={"User-Agent": _UA})
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []
    return data.get("activeStorms", []) or []


def _enrich_nhc_storm(s: Dict) -> Optional[Dict]:
    try:
        lat = float(s.get("latitudeNumeric"))
        lon = float(s.get("longitudeNumeric"))
    except (TypeError, ValueError):
        return None
    try:
        wind = float(s.get("intensity", 0) or 0)
    except (TypeError, ValueError):
        wind = 0.0
    try:
        pressure = float(s.get("pressure")) if s.get("pressure") else None
    except (TypeError, ValueError):
        pressure = None
    sid = str(s.get("id") or s.get("binNumber") or "")
    basin = _basin_from_id(sid) or "AL"
    basin_name = {"AL": "North Atlantic", "EP": "Eastern Pacific",
                  "CP": "Central Pacific"}.get(basin, basin)
    return {
        "id": sid, "source": "NOAA NHC",
        "name": s.get("name") or "Unnamed",
        "basin": basin, "basin_name": basin_name,
        "category": _classify_intensity(wind),
        "lat": round(lat, 2), "lon": round(lon, 2),
        "intensity_kt": wind,
        "pressure_mb": pressure,
        "movement_dir_deg": s.get("movementDir"),
        "movement_speed_kt": s.get("movementSpeed"),
        "last_update": s.get("lastUpdate"),
        "public_advisory_url": (s.get("publicAdvisory") or {}).get("url"),
        "oil_impact": _enrich_oil_impact(lat, lon, basin),
    }


# ====================================================================
# JTWC source (WP + NI + SH basins) — RSS index + per-storm text bulletins
# ====================================================================
# Patterns to pull current position + intensity from JTWC warning text.
_JTWC_NAME_RE  = re.compile(
    r"SUBJ/(?:TROPICAL\s+(?:DEPRESSION|STORM)|TYPHOON|SUPER\s+TYPHOON|HURRICANE|CYCLONE)\s+"
    r"(\d{2}[A-Z])\s*\(([^)]+)\)",
    re.IGNORECASE)
_JTWC_POS_RE   = re.compile(
    r"WARNING\s+POSITION:\s*\n\s*\d+Z\s+---\s+NEAR\s+([\d.]+)([NS])\s+([\d.]+)([EW])",
    re.IGNORECASE)
_JTWC_WIND_RE  = re.compile(
    r"MAX\s+SUSTAINED\s+WINDS\s*-\s*(\d+)\s*KT",
    re.IGNORECASE)
_JTWC_MOVE_RE  = re.compile(
    r"MOVEMENT\s+PAST\s+SIX\s+HOURS\s*-\s*(\d+)\s+DEGREES\s+AT\s+(\d+)\s*KTS?",
    re.IGNORECASE)
_JTWC_LINK_RE  = re.compile(
    r"https?://www\.metoc\.navy\.mil/jtwc/products/[a-z0-9]+web\.txt",
    re.IGNORECASE)


async def _fetch_jtwc_index(client: httpx.AsyncClient) -> List[str]:
    """Return URLs of currently-active JTWC TC warning text files."""
    try:
        r = await client.get(JTWC_RSS_URL,
                             headers={"User-Agent": _UA}, timeout=12)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    urls: List[str] = []
    for desc in root.iter("description"):
        if not desc.text:
            continue
        urls.extend(_JTWC_LINK_RE.findall(desc.text))
    # de-dup, preserve order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


async def _fetch_jtwc_warning(client: httpx.AsyncClient,
                               url: str) -> Optional[Dict]:
    try:
        r = await client.get(url, headers={"User-Agent": _UA}, timeout=12)
        r.raise_for_status()
        text = r.text
    except Exception:
        return None

    # Storm name + designator
    sid_match = _JTWC_NAME_RE.search(text)
    if not sid_match:
        return None
    designator = sid_match.group(1)         # e.g. "06W"
    name = sid_match.group(2).strip().title()  # e.g. "Jangmi"
    basin = _basin_from_id(designator)
    if not basin:
        return None

    # Position
    pos_match = _JTWC_POS_RE.search(text)
    if not pos_match:
        return None
    lat = float(pos_match.group(1))
    if pos_match.group(2).upper() == "S":
        lat = -lat
    lon = float(pos_match.group(3))
    if pos_match.group(4).upper() == "W":
        lon = -lon

    # Wind
    wind_match = _JTWC_WIND_RE.search(text)
    wind = float(wind_match.group(1)) if wind_match else 0.0

    # Movement
    mv_match = _JTWC_MOVE_RE.search(text)
    move_dir = int(mv_match.group(1)) if mv_match else None
    move_spd = int(mv_match.group(2)) if mv_match else None

    basin_name = {"WP": "Western Pacific", "NI": "North Indian Ocean",
                  "SH": "Southern Hemisphere"}.get(basin, basin)

    return {
        "id": designator, "source": "JTWC",
        "name": name,
        "basin": basin, "basin_name": basin_name,
        "category": _classify_intensity(wind),
        "lat": round(lat, 2), "lon": round(lon, 2),
        "intensity_kt": wind,
        "pressure_mb": None,            # JTWC text doesn't reliably print pressure
        "movement_dir_deg": move_dir,
        "movement_speed_kt": move_spd,
        "last_update": None,
        "public_advisory_url": url,
        "oil_impact": _enrich_oil_impact(lat, lon, basin),
    }


async def _fetch_jtwc() -> List[Dict]:
    """Pull all active JTWC storms with per-warning detail parsing."""
    results: List[Dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            urls = await _fetch_jtwc_index(client)
            for u in urls[:8]:           # cap to avoid runaway requests
                w = await _fetch_jtwc_warning(client, u)
                if w:
                    results.append(w)
    except Exception:
        return results
    return results


# ====================================================================
# StormTracker — combined NHC + JTWC across all basins
# ====================================================================
class StormTracker:
    def __init__(self) -> None:
        self.storms: List[Dict] = []
        self.last_fetch: Optional[float] = None
        self.source = "pending first fetch"
        self.error: Optional[str] = None

    async def refresh(self) -> int:
        # Fetch both feeds — independent so NHC outage doesn't kill JTWC etc.
        nhc_raw = await _fetch_nhc()
        jtwc = []
        try:
            jtwc = await _fetch_jtwc()
        except Exception:
            pass

        enriched: List[Dict] = []
        nhc_basins_covered: set = set()  # NHC is authoritative for AL/EP/CP
        for s in nhc_raw:
            e = _enrich_nhc_storm(s)
            if e is not None:
                enriched.append(e)
                nhc_basins_covered.add(e["basin"])
        # JTWC also reports EP/CP for Navy ops — duplicates of NHC. Keep
        # JTWC only for basins NHC doesn't cover (WP, NI, SH primarily).
        for e in jtwc:
            if e is None:
                continue
            if e["basin"] in {"AL", "EP", "CP"}:
                # already covered by NHC's authoritative feed — skip dup
                continue
            enriched.append(e)

        self.storms = enriched
        self.last_fetch = time.time()
        self.error = None
        self.source = (
            "NOAA NHC (AL + EP + CP basins) + JTWC RSS (WP + NI + SH basins)")
        return len(self.storms)

    def snapshot(self) -> Dict:
        if not self.storms:
            overall_tag = "CLEAR"
            overall_status = "no active tropical cyclones globally"
        else:
            tags = [s.get("oil_impact", {}).get("tag", "MONITOR")
                    for s in self.storms]
            if "PRODUCTS BULLISH" in tags or "CRUDE BULLISH" in tags:
                overall_tag = "OIL_ASSET_THREAT"
                overall_status = "active threat to oil refining or production"
            elif "WATCH" in tags:
                overall_tag = "WATCH"
                overall_status = (f"{len(self.storms)} active cyclone(s), "
                                  "monitoring asset distances")
            else:
                overall_tag = "DISTANT"
                overall_status = (
                    f"{len(self.storms)} active cyclone(s), all distant from "
                    "tracked oil assets")
        # Group storms by basin for the panel
        by_basin: Dict[str, List[Dict]] = {}
        for s in self.storms:
            by_basin.setdefault(s["basin"], []).append(s)

        return {
            "storms": self.storms,
            "by_basin": by_basin,
            "count": len(self.storms),
            "overall_tag": overall_tag,
            "overall_status": overall_status,
            "last_fetch": self.last_fetch,
            "source": self.source,
            "error": self.error,
            "n_regions_tracked": len(REGIONS),
            "n_refineries_tracked": sum(
                len(r["refineries"]) for r in REGIONS.values()),
            "total_capacity_kbpd": sum(
                cap for r in REGIONS.values()
                for _, cap, _, _ in r["refineries"]),
            "gulf_total_capacity_kbpd": GULF_TOTAL_CAPACITY,
            "gulf_production_baseline_kbpd": GOM_PRODUCTION_KBPD,
            "risk_radius_nm": RISK_RADIUS_NM,
        }

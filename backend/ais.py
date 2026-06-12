"""Live AIS tanker tracking via aisstream.io WebSocket.

Subscribes to PositionReport + ShipStaticData messages inside bounding boxes
around the world's major oil-shipping hubs. Counts vessels currently in each
zone, split between anchored vs underway — the anchored count at hubs like
Singapore / Fujairah is a real-world proxy for floating storage / port
congestion that physical oil traders watch.

Free aisstream.io tier covers terrestrial receivers (~40 mi from shore).
That's enough for port congestion but won't see open-ocean floating storage
or mid-Atlantic transit traffic — that needs paid Kpler / Vortexa.

AIS ship type codes (ITU-R M.1371):
  70-79  cargo
  80-89  tanker (oil / chemical / gas / hazmat carriers)
  90-99  other"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except Exception:                              # pragma: no cover
    websockets = None
    ConnectionClosed = Exception   # type: ignore

AIS_WS_URL = "wss://stream.aisstream.io/v0/stream"

TANKER_TYPE_LO = 80
TANKER_TYPE_HI = 89

# AIS navigational status codes considered "stationary" — anchored, moored,
# or aground. A tanker sitting at one of these for hours at a major hub is
# either waiting to discharge (port congestion) or being used as floating
# storage. Both are bullish-supply-side signals.
ANCHORED_STATUSES = {1, 5, 6}

# Drop vessel records older than this. Tankers move slowly, so a 90-min
# stale window catches everything currently parked in a zone without
# showing ghosts from old position reports.
STALE_AFTER_SECONDS = 90 * 60

# Major oil-shipping zones — (south_lat, west_lon, north_lat, east_lon).
# First 7 = primary hubs (loading/discharge/floating-storage).
# Last 3 = known dark-fleet STS zones for sanctioned crude (Russian/Iranian).
ZONES: Dict[str, Tuple[float, float, float, float]] = {
    "Houston/Galveston": (29.00, -95.10, 29.85, -94.50),
    "LOOP (US Gulf)":    (28.80, -90.10, 28.95, -89.95),
    "Rotterdam/ARA":     (51.85, 3.50, 52.30, 4.50),
    "Singapore/Malacca": (1.10, 103.55, 1.45, 104.25),
    "Fujairah":          (25.05, 56.35, 25.40, 56.75),
    "Caribbean (Vz/Cu)": (11.70, -70.40, 12.20, -68.75),
    "Saldanha Bay":      (-33.10, 17.85, -32.95, 18.10),
    # Lakonikos Gulf (S. Peloponnese, Greece) — primary STS zone for Russian
    # Urals/ESPO crude after the EU price-cap regime began. Heavily used by
    # dark-fleet tankers transferring cargo from sanctioned to "clean" hulls.
    "Lakonikos Gulf":    (36.50, 22.50, 36.90, 23.10),
    # Ceuta (Spanish exclave, Strait of Gibraltar) — major bunkering + STS
    # corridor for Russian crude moving between Baltic exports and final
    # buyers in Asia/India.
    "Ceuta (Gibraltar)": (35.85, -5.45, 35.97, -5.15),
    # Sohar (Oman) — STS zone for Iranian crude transferring to "clean"
    # vessels before delivery to Chinese teapot refiners.
    "Sohar (Oman)":      (24.40, 56.55, 24.55, 56.80),
}

# Strategic global oil choke points — narrow waterways where the
# world's seaborne crude is forced through. Boxes are sized to cover
# the actual transit corridor, not the broader sea area. Coordinates
# verified from EIA chokepoint reports + USCG navigation charts.
# Each choke handles the listed daily flow on average.
CHOKE_POINTS: Dict[str, Dict] = {
    "Strait of Hormuz": {
        "bbox":  (25.50, 55.40, 26.80, 57.20),
        "flow_mbpd": 17.0,
        "context": "~20% of global oil. Iran-UAE-Oman gateway. Iran tension flashpoint.",
    },
    "Strait of Malacca": {
        "bbox":  (0.50, 102.50, 3.50, 104.50),
        "flow_mbpd": 16.0,
        "context": "Asia imports — China/Japan/Korea reliance. Piracy zone.",
    },
    "Bab el-Mandeb": {
        "bbox":  (12.00, 42.90, 13.50, 44.30),
        "flow_mbpd": 6.0,
        "context": "Red Sea ↔ Gulf of Aden. Houthi missile / drone attack zone.",
    },
    "Suez Canal (north)": {
        "bbox":  (30.40, 32.00, 31.60, 33.00),
        "flow_mbpd": 5.0,
        "context": "Mediterranean ↔ Red Sea. Closure forces ~7-day Cape detour.",
    },
    "Turkish Straits": {
        "bbox":  (40.95, 28.80, 41.30, 29.30),
        "flow_mbpd": 3.0,
        "context": "Russian Black Sea exports — Novorossiysk to Mediterranean.",
    },
    "Danish Straits": {
        "bbox":  (54.80, 10.50, 56.20, 13.50),
        "flow_mbpd": 3.0,
        "context": "Russian Baltic exports — Primorsk + Ust-Luga (sanctioned routes).",
    },
    "Panama Canal (Atl)": {
        "bbox":  (9.20, -80.10, 9.60, -79.60),
        "flow_mbpd": 0.5,
        "context": "Pacific ↔ Atlantic. Drought delays force tanker reroutes.",
    },
}

# Per-class tanker capacity in barrels — used to weight throughput +
# at-risk-capacity estimates. Derived from typical DWT × bbl-per-tonne (7.33).
SIZE_CAPACITY_BBL: Dict[str, int] = {
    "VLCC":          2_000_000,   # 320k+ DWT, 320-330m × 58-60m
    "Suezmax":       1_000_000,   # 150-200k DWT, 270-285m × 48-52m
    "Aframax":         750_000,   # 80-120k DWT, 240-250m × 42-44m
    "Panamax":         500_000,   # 60-80k DWT, 200-230m × 32-38m
    "Product":         300_000,   # 30-50k DWT, 180m × 32m
    "Small/Unknown":   200_000,
}

# Fallback used when we don't yet know vessel size — mix-weighted average
# across the global tanker fleet.
AVG_TANKER_CAPACITY_BBL = 800_000

# Floating-storage threshold: tanker continuously anchored at an oil hub
# for this many days is flagged as a floating-storage candidate.
# 7d is the rigorous threshold (matches Kpler/Vortexa methodology), but it
# only fires after we've observed a vessel for that long — meaningless on a
# fresh container boot. 2d catches "currently parked" candidates: shorter
# than true floating-storage but still meaningfully bullish on supply when
# the count rises (vessels not turning around).
FLOATING_STORAGE_THRESHOLD_DAYS = 2

# STS rendezvous detection: two confirmed tankers within this many metres
# of each other, both slow (< 1.5 knots) and roughly co-stationary, get
# flagged as a possible ship-to-ship transfer.
STS_DISTANCE_METERS = 400
STS_MAX_SPEED_KN = 1.5
# Locations where vessels routinely berth alongside each other for normal
# port operations — close proximity isn't STS, it's just docking. Excluded
# from pairwise scanning to avoid 100s of false positives at Rotterdam.
# We still scan Fujairah/Singapore (real STS hubs) and ALL choke points,
# where two tankers parked nose-to-tail genuinely warrants attention.
STS_EXCLUDE_LOCATIONS = {
    "Rotterdam/ARA", "Houston/Galveston", "LOOP (US Gulf)",
    "Saldanha Bay", "Caribbean (Vz/Cu)",
    "Suez Canal (north)", "Danish Straits", "Turkish Straits",
}

# Rolling window for the daily-throughput estimate at choke points.
TRANSIT_WINDOW_SECONDS = 24 * 3600


def _classify_vessel_size(length_m: float, width_m: float) -> str:
    """Map AIS dimensions to a tanker size class. Boundaries reflect IMO
    naming conventions (VLCC/Suezmax/Aframax/Panamax/Product)."""
    if length_m >= 320 and width_m >= 55:
        return "VLCC"
    if length_m >= 260 and width_m >= 45:
        return "Suezmax"
    if length_m >= 230 and width_m >= 38:
        return "Aframax"
    if length_m >= 195 and width_m >= 30:
        return "Panamax"
    if length_m >= 130:
        return "Product"
    return "Small/Unknown"


def _haversine_meters(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _zone_of(lat: float, lon: float) -> Optional[str]:
    """Return the hub-zone name containing (lat, lon), or None."""
    for name, (s, w, n, e) in ZONES.items():
        if s <= lat <= n and w <= lon <= e:
            return name
    return None


def _choke_of(lat: float, lon: float) -> Optional[str]:
    """Return the choke-point name containing (lat, lon), or None.
    Checked independently from hub zones — a vessel can be in neither,
    a hub, OR a choke point (but not both at once because the boxes
    don't overlap)."""
    for name, info in CHOKE_POINTS.items():
        s, w, n, e = info["bbox"]
        if s <= lat <= n and w <= lon <= e:
            return name
    return None


class TankerTracker:
    """Long-running WebSocket consumer + rolling vessel buffer."""

    def __init__(self) -> None:
        self.vessels: Dict[int, Dict] = {}     # MMSI -> latest position record
        self.types: Dict[int, int] = {}        # MMSI -> ship_type (tankers only)
        # Per-tanker static data extracted from ShipStaticData broadcasts.
        # Drives the size classification + capacity estimates.
        self.dimensions: Dict[int, Tuple[float, float]] = {}  # MMSI -> (L, W)
        self.draught: Dict[int, float] = {}     # MMSI -> latest reported draught (m)
        self.size_class: Dict[int, str] = {}    # MMSI -> "VLCC" / "Aframax" / ...
        # Transit tracking: per-MMSI which choke they're currently inside
        # (start time recorded so we can attribute a completed transit to
        # the choke when they exit). choke_transits is a rolling buffer of
        # completed transits per choke, used for 24h throughput estimate.
        self.choke_entry: Dict[int, Tuple[str, float]] = {}   # MMSI -> (choke, entry_ts)
        self.choke_transits: Dict[str, Deque[Dict]] = {
            name: deque(maxlen=2000) for name in CHOKE_POINTS
        }
        # Floating storage: per-MMSI when they were first observed anchored
        # at their current hub. Resets when they go underway.
        self.anchored_since: Dict[int, Tuple[str, float]] = {}
        self.status = "pending first connection"
        self.connected_since: Optional[float] = None
        self.messages_seen = 0
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    # ----- network loop ---------------------------------------------------
    async def run(self, api_key: str) -> None:
        """Keep a WS connection open, reconnecting with exponential backoff."""
        if not api_key:
            self.status = "disabled (no AIS_API_KEY set)"
            return
        if websockets is None:
            self.status = "disabled (websockets lib not installed)"
            return

        # Subscribe to BOTH hub zones AND strategic choke points so
        # aisstream forwards us every PositionReport / ShipStaticData
        # message inside either set of boxes — 7 hubs + 7 chokepoints = 14
        # bounding boxes total, well within aisstream free-tier limits.
        all_boxes: List[List[List[float]]] = [
            [[s, w], [n, e]] for (s, w, n, e) in ZONES.values()
        ]
        all_boxes += [
            [[info["bbox"][0], info["bbox"][1]],
             [info["bbox"][2], info["bbox"][3]]]
            for info in CHOKE_POINTS.values()
        ]
        sub = {
            "APIKey": api_key,
            "BoundingBoxes": all_boxes,
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }
        backoff = 4
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                        AIS_WS_URL, ping_interval=30,
                        close_timeout=5) as ws:
                    self.connected_since = time.time()
                    self.status = "live (aisstream.io)"
                    await ws.send(json.dumps(sub))
                    backoff = 4
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        with contextlib.suppress(Exception):
                            self._handle(json.loads(raw))
            except ConnectionClosed:
                self.status = "reconnecting (ws closed)"
            except Exception as ex:
                self.status = f"reconnecting ({type(ex).__name__})"
            self.connected_since = None
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                break                          # stop() called during sleep
            except asyncio.TimeoutError:
                pass
            backoff = min(60, backoff * 2)
        self.status = "stopped"

    # ----- message handling ----------------------------------------------
    def _handle(self, msg: dict) -> None:
        self.messages_seen += 1
        mtype = msg.get("MessageType")
        meta = msg.get("MetaData") or {}
        mmsi = meta.get("MMSI")
        if not isinstance(mmsi, int):
            return
        body = msg.get("Message") or {}
        if mtype == "ShipStaticData":
            sd = body.get("ShipStaticData") or {}
            st = sd.get("Type")
            if isinstance(st, int) and TANKER_TYPE_LO <= st <= TANKER_TYPE_HI:
                self.types[mmsi] = st
                # Extract dimensions → size class. aisstream returns
                # Dimension as {A, B, C, D} where length = A+B, width = C+D
                # (measured from GPS antenna to bow/stern/port/starboard).
                dim = sd.get("Dimension") or {}
                try:
                    bow   = float(dim.get("A", 0) or 0)
                    stern = float(dim.get("B", 0) or 0)
                    port  = float(dim.get("C", 0) or 0)
                    sb    = float(dim.get("D", 0) or 0)
                    length = bow + stern
                    width = port + sb
                    if length > 0 and width > 0:
                        self.dimensions[mmsi] = (length, width)
                        self.size_class[mmsi] = _classify_vessel_size(
                            length, width)
                except (TypeError, ValueError):
                    pass
                # Maximum static draught (m) — this is the LADEN draught the
                # vessel reports for the current voyage, NOT the design max.
                # Loaded tankers have higher values than ballasting ones.
                try:
                    d = float(sd.get("MaximumStaticDraught", 0) or 0)
                    if d > 0:
                        self.draught[mmsi] = d
                except (TypeError, ValueError):
                    pass
            return
        if mtype != "PositionReport":
            return
        pr = body.get("PositionReport") or {}
        lat = pr.get("Latitude")
        lon = pr.get("Longitude")
        if lat is None:
            lat = meta.get("latitude")
        if lon is None:
            lon = meta.get("longitude")
        if lat is None or lon is None:
            return
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            return
        zone = _zone_of(lat_f, lon_f)
        choke = _choke_of(lat_f, lon_f) if zone is None else None
        if zone is None and choke is None:
            return
        nav = pr.get("NavigationalStatus")
        anchored = nav in ANCHORED_STATUSES
        now = time.time()
        sog_raw = pr.get("Sog")
        try:
            sog_val = float(sog_raw) if sog_raw is not None else None
        except (TypeError, ValueError):
            sog_val = None

        # ---------- Floating-storage tracker ----------
        # If vessel is anchored at a hub zone, start (or continue) timer.
        # If now underway or in a choke (transiting), clear it.
        if zone and anchored:
            prev = self.anchored_since.get(mmsi)
            if not prev or prev[0] != zone:
                self.anchored_since[mmsi] = (zone, now)
        else:
            self.anchored_since.pop(mmsi, None)

        # ---------- Choke transit tracker ----------
        # Vessel inside a choke right now → remember entry time so we can
        # record a completed transit when they leave.
        if choke and mmsi in self.types:        # only count confirmed tankers
            prev = self.choke_entry.get(mmsi)
            if not prev:
                self.choke_entry[mmsi] = (choke, now)
            elif prev[0] != choke:
                # Transitioned between chokes — close out the old one
                cap = SIZE_CAPACITY_BBL.get(
                    self.size_class.get(mmsi, ""), AVG_TANKER_CAPACITY_BBL)
                self.choke_transits[prev[0]].append({
                    "exit_ts": now,
                    "duration_min": round((now - prev[1]) / 60, 1),
                    "capacity_bbl": cap,
                    "size_class": self.size_class.get(mmsi, "Small/Unknown"),
                })
                self.choke_entry[mmsi] = (choke, now)
        elif not choke and mmsi in self.choke_entry:
            # Vessel exited a choke into open water (or a hub) → record transit
            old_choke, entry_ts = self.choke_entry.pop(mmsi)
            cap = SIZE_CAPACITY_BBL.get(
                self.size_class.get(mmsi, ""), AVG_TANKER_CAPACITY_BBL)
            self.choke_transits[old_choke].append({
                "exit_ts": now,
                "duration_min": round((now - entry_ts) / 60, 1),
                "capacity_bbl": cap,
                "size_class": self.size_class.get(mmsi, "Small/Unknown"),
            })

        self.vessels[mmsi] = {
            "zone": zone,
            "choke": choke,
            "lat": lat_f,
            "lon": lon_f,
            "anchored": anchored,
            "nav": nav,
            "name": (meta.get("ShipName") or "").strip(),
            "sog": sog_val,
            "cog": pr.get("Cog"),
            "ts": now,
        }

    # ----- snapshot for the dashboard ------------------------------------
    def _purge_stale(self) -> None:
        cutoff = time.time() - STALE_AFTER_SECONDS
        for mmsi, v in list(self.vessels.items()):
            if v["ts"] < cutoff:
                del self.vessels[mmsi]

    def snapshot(self) -> Dict:
        """Per-zone + per-chokepoint tanker tallies, size mix, transit
        throughput, floating storage, STS rendezvous candidates."""
        self._purge_stale()
        now = time.time()

        def _empty_bucket() -> Dict:
            return {"total": 0, "anchored": 0, "underway": 0,
                    "confirmed_tankers": 0, "unknown_type": 0,
                    "size_mix": {k: 0 for k in SIZE_CAPACITY_BBL},
                    "capacity_bbl": 0,
                    "laden_count": 0, "ballast_count": 0,
                    "samples": []}

        zones: Dict[str, Dict] = {n: _empty_bucket() for n in ZONES}
        chokes: Dict[str, Dict] = {}
        for n, info in CHOKE_POINTS.items():
            b = _empty_bucket()
            b["baseline_flow_mbpd"] = info["flow_mbpd"]
            b["context"] = info["context"]
            chokes[n] = b

        confirmed_tanker_positions: List[Tuple[int, float, float, str]] = []

        for mmsi, v in self.vessels.items():
            if v.get("zone"):
                bucket = zones[v["zone"]]
                location_key = v["zone"]
            elif v.get("choke"):
                bucket = chokes[v["choke"]]
                location_key = v["choke"]
            else:
                continue
            bucket["total"] += 1
            if v["anchored"]:
                bucket["anchored"] += 1
            else:
                bucket["underway"] += 1
            if mmsi in self.types:
                bucket["confirmed_tankers"] += 1
                size = self.size_class.get(mmsi, "Small/Unknown")
                bucket["size_mix"][size] = bucket["size_mix"].get(size, 0) + 1
                bucket["capacity_bbl"] += SIZE_CAPACITY_BBL.get(
                    size, AVG_TANKER_CAPACITY_BBL)
                # Laden / ballast inference from reported draught
                d = self.draught.get(mmsi)
                if d is not None:
                    if d >= 11.0:
                        bucket["laden_count"] += 1
                    elif d > 0:
                        bucket["ballast_count"] += 1
                confirmed_tanker_positions.append(
                    (mmsi, v["lat"], v["lon"], location_key))
            else:
                bucket["unknown_type"] += 1
            if len(bucket["samples"]) < 5:
                bucket["samples"].append({
                    "name": v["name"] or f"MMSI {mmsi}",
                    "anchored": v["anchored"],
                    "sog": (round(float(v["sog"]), 1)
                            if v.get("sog") is not None else None),
                    "cog": (round(float(v["cog"]), 0)
                            if v.get("cog") is not None else None),
                    "size_class": self.size_class.get(mmsi, "Small/Unknown"),
                    "draught_m": self.draught.get(mmsi),
                    "lat": round(v["lat"], 3),
                    "lon": round(v["lon"], 3),
                    "ts": round(v["ts"]),
                })

        for b in list(zones.values()) + list(chokes.values()):
            b["est_capacity_mbbl"] = round(b["capacity_bbl"] / 1_000_000, 1)

        # ---------- Choke 24h throughput ----------
        cutoff = now - TRANSIT_WINDOW_SECONDS
        for name, q in self.choke_transits.items():
            recent = [t for t in q if t["exit_ts"] >= cutoff]
            barrels = sum(t["capacity_bbl"] for t in recent)
            chokes[name]["transits_24h"] = len(recent)
            chokes[name]["throughput_mbpd_24h"] = round(barrels / 1_000_000, 2)
            chokes[name]["throughput_pct_baseline"] = (
                round(100 * (barrels / 1_000_000) /
                      chokes[name]["baseline_flow_mbpd"], 1)
                if chokes[name]["baseline_flow_mbpd"] else None)

        # ---------- Floating storage ----------
        floating_storage: List[Dict] = []
        for mmsi, (zone, since_ts) in self.anchored_since.items():
            age_days = (now - since_ts) / 86400
            if age_days >= FLOATING_STORAGE_THRESHOLD_DAYS:
                v = self.vessels.get(mmsi) or {}
                size = self.size_class.get(mmsi, "Small/Unknown")
                cap = SIZE_CAPACITY_BBL.get(size, AVG_TANKER_CAPACITY_BBL)
                floating_storage.append({
                    "mmsi": mmsi,
                    "name": v.get("name") or f"MMSI {mmsi}",
                    "zone": zone,
                    "anchored_days": round(age_days, 1),
                    "size_class": size,
                    "capacity_bbl": cap,
                })
        floating_storage.sort(key=lambda x: -x["anchored_days"])

        # ---------- STS rendezvous candidates ----------
        # Pairwise distance between confirmed tankers in the same location
        # — only check pairs in the same hub/choke (keeps it O(k·n²) per
        # bucket rather than O(N²) global, still fast at the scale we run).
        sts_pairs: List[Dict] = []
        by_loc: Dict[str, List[Tuple[int, float, float]]] = {}
        for mmsi, lat, lon, loc in confirmed_tanker_positions:
            by_loc.setdefault(loc, []).append((mmsi, lat, lon))
        for loc, ships in by_loc.items():
            if loc in STS_EXCLUDE_LOCATIONS:
                # busy commercial port — vessels berth side-by-side
                # normally, so close proximity is not a STS signal
                continue
            n = len(ships)
            if n < 2:
                continue
            for i in range(n):
                mi, lai, loi = ships[i]
                vi = self.vessels.get(mi) or {}
                si = vi.get("sog")
                if si is None or si > STS_MAX_SPEED_KN:
                    continue
                for j in range(i + 1, n):
                    mj, laj, loj = ships[j]
                    vj = self.vessels.get(mj) or {}
                    sj = vj.get("sog")
                    if sj is None or sj > STS_MAX_SPEED_KN:
                        continue
                    dist_m = _haversine_meters(lai, loi, laj, loj)
                    if dist_m <= STS_DISTANCE_METERS:
                        sts_pairs.append({
                            "location": loc,
                            "distance_m": round(dist_m),
                            "vessel_a": {
                                "mmsi": mi,
                                "name": vi.get("name") or f"MMSI {mi}",
                                "size_class": self.size_class.get(mi, "?"),
                            },
                            "vessel_b": {
                                "mmsi": mj,
                                "name": vj.get("name") or f"MMSI {mj}",
                                "size_class": self.size_class.get(mj, "?"),
                            },
                        })
        sts_pairs.sort(key=lambda p: p["distance_m"])

        totals = {
            "vessels_in_zones": sum(z["total"] for z in zones.values()),
            "vessels_in_chokes": sum(c["total"] for c in chokes.values()),
            "tanker_types_known": len(self.types),
            "vessels_size_classified": len(self.size_class),
            "messages_seen": self.messages_seen,
        }
        return {
            "status": self.status,
            "connected_since": self.connected_since,
            "totals": totals,
            "zones": zones,
            "chokes": chokes,
            "floating_storage": floating_storage,
            "sts_candidates": sts_pairs,
            "avg_tanker_capacity_bbl": AVG_TANKER_CAPACITY_BBL,
        }

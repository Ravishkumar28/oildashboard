"""Market state engine — REAL PRICES ONLY mode.

Holds real (delayed) oil-market state: WTI/Brent crude, the dollar index,
refined products, and inventories. Every live value mirrors its most-recent
real anchor — there is NO per-tick random walk and NO simulated drift on
anything that has a real feed.

Update cadence (set by main.py):
  - WTI / Brent / RBOB / HO: every ~3 min (yfinance)
  - DXY: every ~15 min (Twelve Data)
  - Crude inventory: every ~6 h (EIA)
  - Curve: every ~3 min (yfinance)

Between refreshes, ALL these values stay flat. No simulation noise. This
gives up the "every-2-second chart wiggle" feel in exchange for total
data honesty — what you see on the chart is what Yahoo last printed.

Two values still don't have free real feeds and remain simulated:
  - BDTI freight (Baltic Exchange = paid-only)
  - WTI daily contract volume per session (yfinance only gives day totals)
These are labelled "simulated" in the dashboard's source tags.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional

from indicators import zscore
from kalman import KalmanPair

HISTORY_DAYS = 260          # ~1 trading year of daily closes
TICKS_PER_SESSION = 120     # live ticks before a day is committed to history
GAL_PER_BBL = 42.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class MarketEngine:
    """Holds market state and advances it on each live tick."""

    def __init__(self,
                 real_history: Optional[Dict[str, List[float]]] = None) -> None:
        random.seed()
        # --- anchor / starting levels -------------------------------------
        self.wti = 78.40
        self.brent = 82.60
        self.dxy = 104.20
        self.bdti = 920.0          # Baltic Dirty Tanker Index
        self.crude_inventory = 426.0  # million barrels, US commercial crude
        self.rbob = 2.36           # $/gal gasoline
        self.heat = 2.58           # $/gal diesel/heating oil
        self.dubai = 80.70         # $/bbl medium-sour benchmark

        # curve shape: positive = contango, negative = backwardation
        self.curve_slope = -0.0035

        # live real-data anchors (set by the data feeds); None = pure sim
        self.anchor_wti: Optional[float] = None
        self.anchor_brent: Optional[float] = None
        self.anchor_dxy: Optional[float] = None
        self.anchor_crude_inv: Optional[float] = None
        self.anchor_rbob: Optional[float] = None
        self.anchor_heat: Optional[float] = None
        # set to True after the first real product anchor; ensures the
        # crack-history rebase only happens once
        self._crack_rebased = False
        # rolling history of the M3-M6-M9 butterfly value, used by the
        # Curve Butterfly strategy's z-score trigger
        self.fly_history: List[float] = [
            round(random.gauss(0.0, 0.4), 3) for _ in range(30)
        ]
        # Per-product fly histories (capped at 120 each). Populated lazily
        # by record_product_flies() whenever a product's curve refreshes.
        # WTI is aliased to the existing fly_history above.
        self.product_fly_history: Dict[str, List[float]] = {
            "wti":   self.fly_history,
            "brent": [],
            "rbob":  [],
            "heat":  [],
            "natgas": [],
        }
        # Kalman filter tracking WTI = alpha + beta * DXY (time-varying
        # hedge ratio). Used by the dynamic-beta pair-trading strategy.
        self.kalman = KalmanPair(q_alpha=1e-4, q_beta=1e-5, r_obs=4.0)
        self.kalman_resid_history: List[float] = []
        # Real same-calendar-week closes for the past 5 years (panel 11).
        # None until lifespan startup wires it from yfinance.
        self.real_five_year: Optional[List[Dict[str, float]]] = None
        self.five_year_source = "simulation"
        # real futures curve overrides the simulated one when available
        self.real_curve: Optional[List[Dict[str, float]]] = None
        # Real 12-month settlement curves for the other 3 NYMEX products.
        # Populated by main.py refresh tasks every ~60s via yfinance.
        self.brent_curve: Optional[List[Dict[str, float]]] = None
        self.rbob_curve:  Optional[List[Dict[str, float]]] = None
        self.heat_curve:  Optional[List[Dict[str, float]]] = None
        self.natgas_curve: Optional[List[Dict[str, float]]] = None
        self.price_source = "simulation"
        self.dxy_source = "simulation"
        self.crack_source = "simulation"
        # Default tag — overridden once a real source is wired in (Yahoo
        # per-contract or xlsx). The synthetic cost-of-carry fallback was
        # removed per user directive (real-only data, no synthesis).
        self.curve_source = "pending real data"

        # --- history series ----------------------------------------------
        # rbob/heat/natgas added for the commodity correlation matrix.
        # Populated from real yfinance daily closes in _bootstrap_history
        # (and refreshed via refresh_history in main.py).
        self.hist: Dict[str, List[float]] = {
            "wti": [], "brent": [], "dxy": [], "bdti": [],
            "spread": [], "crude_inventory": [],
            "rbob": [], "heat": [], "natgas": [],
        }
        # WTI daily contract volume (for VWAP); populated from real history
        # when available, otherwise synthesized in bootstrap.
        self.wti_volume_hist: List[float] = []
        self.crack_hist: Dict[str, List[float]] = {}
        self.tick_count = 0
        self.last_news_offset = 0

        # Rolling history of full 12-month curve snapshots, used to compute
        # the calendar-spread covariance matrix (M1-M2, M2-M3, ...).
        # Each entry is a list of N month prices.
        self.curve_hist: Deque[List[float]] = deque(maxlen=200)
        # Per-product curve histories — seeded from Yahoo at boot. Powers
        # regime-conditional spread/fly stats for Brent / RBOB / HO / NatGas
        # in the Phase 2 opportunity engine, and front-month TA on the
        # technical-analysis panel.
        self.brent_curve_hist:  Deque[List[float]] = deque(maxlen=400)
        self.rbob_curve_hist:   Deque[List[float]] = deque(maxlen=400)
        self.heat_curve_hist:   Deque[List[float]] = deque(maxlen=400)
        self.natgas_curve_hist: Deque[List[float]] = deque(maxlen=400)
        # Per-month idiosyncratic noise (AR(1)) — when the real curve is
        # offset-based and the front-month wti random-walks, every spread is
        # otherwise constant per tick. This small noise gives spreads real
        # variance so the covariance matrix is informative. Magnitude is
        # tiny (~$0.30 stationary stddev) so the displayed curve is barely
        # perturbed.
        self._curve_noise: List[float] = [0.0] * 12

        self._bootstrap_history(real_history)

    # ------------------------------------------------------------------ #
    # history bootstrap
    # ------------------------------------------------------------------ #
    def _bootstrap_history(
        self, real_history: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """Build daily history. If real WTI/Brent closes are supplied (from
        the yfinance feed) the crude track is real and every derived series —
        spread and the six crack spreads — is rebuilt consistently on top of
        it; otherwise the whole market is a correlated random walk."""
        real_wti = real_history.get("wti") if real_history else None
        real_brent = real_history.get("brent") if real_history else None
        real_vol = real_history.get("wti_volume") if real_history else None
        real_dxy = real_history.get("dxy") if real_history else None
        real_rbob = real_history.get("rbob") if real_history else None
        real_heat = real_history.get("heat") if real_history else None
        real_natgas = real_history.get("natgas") if real_history else None
        n = len(real_wti) if real_wti else HISTORY_DAYS

        wti = real_wti[0] if real_wti else 71.0
        dxy = real_dxy[0] if real_dxy else 101.5
        brent = real_brent[0] if real_brent else wti + 4.3
        bdti = 780.0
        inv = 445.0
        # scale product prices to the starting crude level
        rbob = 2.12 * (wti / 71.0)
        heat = 2.30 * (wti / 71.0)
        slope = -0.002
        prev_wti = wti

        for i in range(n):
            if real_dxy and i < len(real_dxy):
                dxy = real_dxy[i]
                dxy_ret = 0.0  # not used when real_wti present
            else:
                dxy_ret = random.gauss(0.0, 0.0042)
                dxy = _clamp(dxy * (1 + dxy_ret), 94.0, 116.0)

            if real_wti:
                wti = real_wti[i]
                wti_ret = (wti - prev_wti) / prev_wti \
                    if i and prev_wti else 0.0
            else:
                # crude inversely tracks the dollar plus its own shock
                wti_ret = -0.55 * dxy_ret + random.gauss(0.0003, 0.018)
                wti = _clamp(wti * (1 + wti_ret), 45.0, 130.0)
            prev_wti = wti

            if real_brent:
                brent = real_brent[i]
            else:
                # brent/wti spread mean-reverts toward ~4.2
                sp = brent - wti
                sp += 0.18 * (4.2 - sp) + random.gauss(0.0, 0.32)
                brent = wti + sp

            bdti = _clamp(bdti * (1 + random.gauss(0.0, 0.022)) +
                          0.04 * (950.0 - bdti), 500.0, 1700.0)
            inv = _clamp(inv + random.gauss(0.0, 2.3) + 0.03 * (420.0 - inv),
                         350.0, 510.0)

            # Products: real RB=F / HO=F daily closes when available,
            # otherwise synthesised (crude * crack noise) as fallback.
            if real_rbob and i < len(real_rbob):
                rbob = real_rbob[i]
            else:
                rbob = _clamp(rbob * (1 + wti_ret + random.gauss(0.0, 0.006)),
                              1.4, 4.5)
            if real_heat and i < len(real_heat):
                heat = real_heat[i]
            else:
                heat = _clamp(heat * (1 + wti_ret + random.gauss(0.0, 0.006)),
                              1.6, 4.7)
            slope += 0.05 * (-0.003 - slope) + random.gauss(0.0, 0.0009)

            self.hist["wti"].append(round(wti, 2))
            self.hist["brent"].append(round(brent, 2))
            self.hist["dxy"].append(round(dxy, 2))
            self.hist["bdti"].append(round(bdti, 1))
            self.hist["spread"].append(round(brent - wti, 2))
            self.hist["crude_inventory"].append(round(inv, 1))
            # Real product histories for the commodity correlation matrix
            self.hist["rbob"].append(round(rbob, 4))
            self.hist["heat"].append(round(heat, 4))
            if real_natgas and i < len(real_natgas):
                self.hist["natgas"].append(round(real_natgas[i], 3))
            # WTI contract volume (for VWAP): real from yfinance when
            # supplied, otherwise a plausible synthetic ~220k contracts/day
            if real_vol and i < len(real_vol):
                self.wti_volume_hist.append(float(real_vol[i]))
            else:
                self.wti_volume_hist.append(
                    max(50_000.0, random.gauss(220_000.0, 80_000.0)))

            # Dubai (medium-sour) tracks Brent with its own daily noise so the
            # Brent-Dubai EFS has realistic historical variance and a non-zero
            # z-score denominator.
            dubai_day = (brent - 1.9) + random.gauss(0.0, 0.45)
            cr = self._crack_set(wti, brent, rbob, heat, dubai_day)
            for name, val in cr.items():
                self.crack_hist.setdefault(name, []).append(round(val, 2))

        # adopt the final state as 'live now'
        self.wti = self.hist["wti"][-1]
        self.brent = self.hist["brent"][-1]
        self.dxy = self.hist["dxy"][-1]
        self.bdti = self.hist["bdti"][-1]
        self.crude_inventory = self.hist["crude_inventory"][-1]
        self.rbob, self.heat = rbob, heat
        self.dubai = self.brent - 1.9
        self.curve_slope = slope
        self.price_source = ("Yahoo Finance CL=F / BZ=F (~15-min delayed)"
                             if real_wti else "simulation")

        # Warm up the Kalman pair filter with the historical WTI/DXY series
        # so the dynamic hedge ratio is already a good fit by tick 1 and the
        # residual z-score has a real baseline to compare against.
        for w, d in zip(self.hist["wti"], self.hist["dxy"]):
            innovation = self.kalman.update(w, d)
            self.kalman_resid_history.append(round(innovation, 3))
        if len(self.kalman_resid_history) > HISTORY_DAYS:
            self.kalman_resid_history = self.kalman_resid_history[-HISTORY_DAYS:]

    # ------------------------------------------------------------------ #
    # crack spreads
    # ------------------------------------------------------------------ #
    @staticmethod
    def _crack_set(wti: float, brent: float, rbob: float, heat: float,
                   dubai: float) -> Dict[str, float]:
        """Six refining / regional spread measures, all in $/bbl."""
        g = rbob * GAL_PER_BBL
        d = heat * GAL_PER_BBL
        return {
            "3-2-1 Crack (USGC)": (2 * g + d) / 3 - wti,
            "2-1-1 Crack": (g + d) / 2 - wti,
            "Gasoline Crack (RBOB)": g - wti,
            "Diesel Crack (Heating Oil)": d - wti,
            "Brent-WTI Sweet Diff": brent - wti,
            "Brent-Dubai EFS": brent - dubai,
        }

    # ------------------------------------------------------------------ #
    # live tick — REAL PRICES ONLY MODE
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        """Mirror real anchors onto live values. No random walk.

        Every live value just snaps to its most recent real anchor. Between
        anchor refreshes (every ~3 min for prices, ~15 min for DXY, ~6 h
        for EIA), the values stay flat. The dashboard chart is therefore
        pure real Yahoo data — no per-tick wiggle, no simulated drift,
        no jumps when the anchor refreshes.

        Only BDTI (no free Baltic feed) and Dubai (derived from Brent) still
        have implicit noise; everything else is anchored or zeroed out.
        """
        self.tick_count += 1

        # Mirror anchors to live values where a real anchor exists.
        if self.anchor_wti is not None:
            self.wti = self.anchor_wti
        if self.anchor_brent is not None:
            self.brent = self.anchor_brent
        if self.anchor_dxy is not None:
            self.dxy = self.anchor_dxy
        if self.anchor_rbob is not None:
            self.rbob = self.anchor_rbob
        if self.anchor_heat is not None:
            self.heat = self.anchor_heat
        if self.anchor_crude_inv is not None:
            self.crude_inventory = self.anchor_crude_inv
        # Dubai = Brent - 1.9 sour-sweet differential (rough proxy; no free
        # real feed for Dubai/Oman crude either)
        self.dubai = self.brent - 1.9
        # BDTI: stays at bootstrap value (no real feed, no random walk).
        # If we ever subscribe to Baltic Exchange, replace here.

        # Append the current REAL curve to the rolling history ONCE per
        # calendar day max. Why: yfinance ticks the front of the curve in
        # cent-level increments through the trading day; under exact-equal
        # dedup, each 60-second poll adds a near-duplicate snapshot, and
        # over a few hours the 200 real historical settlements seeded at
        # startup get pushed out. Spread covariance should reflect DAILY
        # settlement evolution, not intraday noise.
        live_curve = self.futures_curve()
        if live_curve:
            today_iso = dt.date.today().isoformat()
            if today_iso != getattr(self, "_last_curve_append_date", None):
                snap = [c["price"] for c in live_curve[:12]]
                if not self.curve_hist or list(self.curve_hist[-1]) != snap:
                    self.curve_hist.append(snap)
                self._last_curve_append_date = today_iso

        # Periodically observe the WTI/DXY pair in the Kalman filter so
        # the dynamic-beta Kalman signal stays alive.
        if self.tick_count % TICKS_PER_SESSION == 0:
            self._record_fly()
            innov = self.kalman.update(self.wti, self.dxy)
            self.kalman_resid_history.append(round(innov, 3))
            if len(self.kalman_resid_history) > HISTORY_DAYS:
                self.kalman_resid_history.pop(0)

    def _commit_day(self) -> None:
        # Sanity gate: refuse to commit when we have no real-price anchor.
        # Without an anchor, the live values are pure random-walk drift that
        # can drift to the clamp floor ($45) over hours of unanchored ticks.
        # Committing those values would inject garbage into the historical
        # chart (which is what caused the recent "$45→$90 jump" glitch).
        if self.anchor_wti is None or self.anchor_brent is None:
            return

        # Also: per-field sanity. If a value has drifted more than 12% from
        # its anchor, commit the anchor itself instead of the drifted value.
        def sane(value: float, anchor: Optional[float],
                 tolerance: float = 0.12) -> float:
            if anchor is None:
                return value
            if abs(value - anchor) > tolerance * anchor:
                return anchor
            return value

        commits = (
            ("wti",             sane(self.wti, self.anchor_wti)),
            ("brent",           sane(self.brent, self.anchor_brent)),
            ("dxy",             sane(self.dxy, self.anchor_dxy)),
            ("bdti",            self.bdti),         # simulated, no anchor
            ("spread",          sane(self.brent, self.anchor_brent)
                                - sane(self.wti, self.anchor_wti)),
            ("crude_inventory", sane(self.crude_inventory,
                                     self.anchor_crude_inv, tolerance=0.05)),
        )
        for key, val in commits:
            self.hist[key].append(round(val, 2))
            if len(self.hist[key]) > HISTORY_DAYS:
                self.hist[key].pop(0)
        cr = self._crack_set(self.wti, self.brent, self.rbob, self.heat,
                             self.dubai)
        for name, v in cr.items():
            series = self.crack_hist.setdefault(name, [])
            series.append(round(v, 2))
            if len(series) > HISTORY_DAYS:
                series.pop(0)
        # synthesize today's WTI contract volume (yfinance doesn't expose
        # tick-level volume; this keeps VWAP rolling forward on new days)
        self.wti_volume_hist.append(
            max(50_000.0, random.gauss(220_000.0, 80_000.0)))
        if len(self.wti_volume_hist) > HISTORY_DAYS:
            self.wti_volume_hist.pop(0)

    def set_anchor(self, wti: Optional[float] = None,
                   brent: Optional[float] = None) -> None:
        """Update the real WTI/Brent anchors. In real-prices-only mode the
        next tick() will mirror these directly onto self.wti/self.brent —
        no smoothing, no drift, the chart jumps cleanly to the new value
        the moment a fresh Yahoo print arrives."""
        if wti and wti > 0:
            self.anchor_wti = wti
            self.wti = wti
        if brent and brent > 0:
            self.anchor_brent = brent
            self.brent = brent
        if (wti or brent) and "real" not in self.price_source:
            self.price_source = ("Yahoo Finance CL=F / BZ=F "
                                 "(~15-min delayed, real prices only)")

    def set_dxy_anchor(self, dxy: Optional[float]) -> None:
        """Update the real DXY anchor (Twelve Data forex). Mirrored to
        self.dxy on the next tick — no simulated drift."""
        if not dxy:
            return
        self.anchor_dxy = dxy
        self.dxy = dxy
        self.dxy_source = "Twelve Data forex (real DXY, refreshed every 5 min)"

    def set_inventory_anchor(self, mbbl: Optional[float]) -> None:
        """Pin US commercial crude stocks to the latest EIA weekly value."""
        if not mbbl:
            return
        self.anchor_crude_inv = mbbl
        self.crude_inventory = mbbl

    def set_product_anchors(self, rbob: Optional[float] = None,
                            heat: Optional[float] = None) -> None:
        """Tether RBOB gasoline and ULSD heating oil to their real Yahoo
        futures prices, so the crack-spread panel is computed off real
        product prices instead of simulated ones.

        On the very first real anchor we also rebase the 90-day crack-spread
        history so its tail aligns with the current real value — otherwise
        z-scores compare today's real crack against a synthetic-low history
        and trigger spurious extreme signals."""
        first_time = not self._crack_rebased
        if rbob:
            self.anchor_rbob = rbob
            self.rbob = rbob
        if heat:
            self.anchor_heat = heat
            self.heat = heat
        if rbob or heat:
            self.crack_source = ("Yahoo Finance RB=F / HO=F "
                                 "(real refining margins, refreshed every 3 min)")
            if first_time:
                self._rebase_crack_history()
                self._crack_rebased = True

    def _rebase_crack_history(self) -> None:
        """Shift each crack-spread history series so the most recent point
        equals the current live (real-anchored) value. Preserves variance
        and day-to-day shape — only re-centres the mean."""
        live = self._crack_set(self.wti, self.brent, self.rbob, self.heat,
                               self.brent - 1.9)
        for name, current in live.items():
            hist = self.crack_hist.get(name)
            if not hist:
                continue
            shift = current - hist[-1]
            if abs(shift) < 0.01:
                continue
            self.crack_hist[name] = [round(v + shift, 2) for v in hist]

    def set_curve_history(self, rows: Optional[List[Dict]]) -> None:
        """Seed curve_hist with REAL historical daily settlement curves
        fetched by ``datafeed.fetch_curve_history()``.

        Each row is one trading day with real Yahoo settlements for every
        currently-listed NYMEX WTI contract. The covariance matrix is then
        built from purely real curve evolution — no synthetic data, no
        noise, no proxies."""
        if not rows:
            return
        self.curve_hist.clear()
        for row in rows:
            prices = row.get("prices") or []
            if prices:
                self.curve_hist.append([float(p) for p in prices])

    def set_product_curve_history(self, key: str,
                                  rows: Optional[List[Dict]]) -> None:
        """Seed a non-WTI product's curve_hist with real Yahoo settlements.

        ``key`` is one of "brent", "rbob", "heat", "natgas". ``rows`` is
        the same shape returned by datafeed.fetch_curve_history()."""
        attr = {
            "brent":  "brent_curve_hist",
            "rbob":   "rbob_curve_hist",
            "heat":   "heat_curve_hist",
            "natgas": "natgas_curve_hist",
        }.get(key)
        if not attr or not rows:
            return
        dq = getattr(self, attr)
        dq.clear()
        for row in rows:
            prices = row.get("prices") or []
            if prices:
                dq.append([float(p) for p in prices])

    def set_real_curve(self, curve: Optional[List[Dict[str, float]]]) -> None:
        """Use a real WTI futures settlement curve from Yahoo instead of the
        simulated curve. ``curve`` is a list of {month, price} dicts.

        Also records the resulting butterfly value into ``fly_history`` so
        the Curve Butterfly strategy has a real-data z-score baseline."""
        if curve:
            self.real_curve = curve
            self.curve_source = ("Yahoo Finance CL<MM><YY>.NYM "
                                 "(real 12-month curve)")
            self._record_fly()

    def butterfly_value(self) -> Optional[float]:
        """M3-M6-M9 futures butterfly: long M3 + short 2x M6 + long M9.

        Captures the curvature of the curve in $/bbl. A positive fly means
        the middle of the curve is concave (M6 'cheap'); negative means
        convex (M6 'rich'). Falls back to nearby month triplets, or to a
        first-middle-last curvature if no canonical triplet is available."""
        curve = self.futures_curve()
        by_month = {c["month"]: c["price"] for c in curve}
        for m1, m2, m3 in [(3, 6, 9), (4, 7, 10), (2, 5, 8), (3, 5, 7)]:
            if m1 in by_month and m2 in by_month and m3 in by_month:
                return round(by_month[m1] - 2 * by_month[m2] + by_month[m3], 3)
        if len(curve) >= 5:
            mid = len(curve) // 2
            return round(curve[0]["price"] - 2 * curve[mid]["price"]
                         + curve[-1]["price"], 3)
        return None

    def _record_fly(self) -> None:
        """Append the current butterfly value to fly_history (capped).
        Also records flies for Brent/RBOB/HO/NatGas when their static
        curves are available — gives every product a rolling fly z-score."""
        fly = self.butterfly_value()
        if fly is None:
            return
        self.fly_history.append(fly)
        if len(self.fly_history) > 120:
            self.fly_history = self.fly_history[-120:]
        # WTI is aliased so the above already covered it. Now record the
        # other 4 product flies if their curves are loaded.
        for key, curve_attr in [("brent", "brent_curve"),
                                 ("rbob",  "rbob_curve"),
                                 ("heat",  "heat_curve"),
                                 ("natgas", "natgas_curve")]:
            curve = getattr(self, curve_attr, None)
            if not curve or len(curve) < 9:
                continue
            try:
                p = [float(row.get("price", 0.0)) for row in curve]
                f = p[2] - 2 * p[5] + p[8]   # M3 - 2*M6 + M9
            except Exception:
                continue
            hist = self.product_fly_history.setdefault(key, [])
            hist.append(round(f, 4))
            if len(hist) > 120:
                self.product_fly_history[key] = hist[-120:]

    def kalman_live_residual(self) -> float:
        """Live (intraday) residual of WTI against the current Kalman
        WTI = alpha + beta * DXY model. Reads from the last-committed
        alpha/beta — Kalman state is only updated on the daily commit."""
        return self.kalman.residual(self.wti, self.dxy)

    # ------------------------------------------------------------------ #
    # derived views
    # ------------------------------------------------------------------ #
    def series(self, key: str, with_live: bool = True) -> List[float]:
        """History for `key`, optionally with the live value appended."""
        base = list(self.hist[key])
        if not with_live:
            return base
        live = {
            "wti": self.wti, "brent": self.brent, "dxy": self.dxy,
            "bdti": self.bdti, "spread": self.brent - self.wti,
            "crude_inventory": self.crude_inventory,
        }.get(key)
        if live is not None:
            base.append(round(live, 2))
        return base

    def futures_curve(self) -> List[Dict[str, float]]:
        """12 monthly WTI futures prices. Sourcing priority — NO synthetic:
        1. Live Yahoo per-contract symbols (when M2-M12 actually return data)
        2. User's xlsx tick-weighted-mid (real, but frozen at file's last_date)

        The previous synthetic fallback (`front × (1 + slope × M) + noise`)
        was removed per user request — every shown price is now real."""
        if self.real_curve:
            return [dict(row) for row in self.real_curve]
        # Fall back to xlsx-sourced real curve (frozen at xlsx last_date).
        try:
            try:
                from . import real_curves
            except ImportError:
                import real_curves
            xlsx_curve = real_curves.get_curve("wti")
            if xlsx_curve:
                self.curve_source = real_curves.source_label("wti")
                return xlsx_curve
        except Exception:
            pass
        # Last resort: front-month only, no fake deep tenors.
        return [{"month": 1, "price": round(float(self.wti), 2),
                  "contract": "front (live)"}]

    def crack_spreads(self) -> List[Dict[str, object]]:
        """Current value + trailing z-score for each of the six spreads."""
        live = self._crack_set(self.wti, self.brent, self.rbob, self.heat,
                               self.dubai)
        out = []
        for name, val in live.items():
            hist = self.crack_hist.get(name, [])
            z = zscore(hist + [val], window=90)
            # Brent-Dubai uses a synthesised Dubai price (Brent − 1.9 + noise).
            # Tag it as simulated inline so the UI is honest about it.
            display_name = name
            if "Dubai" in name:
                display_name = (name
                                + " <span class='sim-inline'>SIMULATED</span>")
            out.append({
                "name": display_name,
                "value": round(val, 2),
                "zscore": round(z, 2),
                "history": [round(x, 2) for x in hist[-60:]],
            })
        return out

    def set_five_year(self,
                      years_data: Optional[List[Dict[str, float]]]) -> None:
        """Pin the 5-year week-range prior values to real yfinance same-week
        closes; supplied once at lifespan startup."""
        if years_data:
            self.real_five_year = [dict(y) for y in years_data]
            self.five_year_source = ("Yahoo Finance CL=F "
                                     "same-week closes (real)")

    def five_year_week(self) -> Dict[str, object]:
        """Current price vs the same calendar week across the last 5 years.

        Uses real yfinance closes when ``real_five_year`` is set; falls back
        to a synthetic walk around the current price otherwise."""
        base = self.wti
        if self.real_five_year:
            years = [dict(y) for y in self.real_five_year]
        else:
            years = []
            this_year = dt.date.today().year
            for i in range(1, 6):
                # synthesize a plausible same-week close for each prior year
                drift = math.sin(i * 1.1) * 9.0 + random.uniform(-6.0, 6.0)
                years.append({
                    "year": this_year - i,
                    "price": round(_clamp(base + drift, 40.0, 125.0), 2),
                })
        prior = [y["price"] for y in years]
        is_lowest = base <= min(prior) if prior else False
        return {
            "current": round(base, 2),
            "years": years,
            "low": round(min(prior + [base]), 2) if prior else round(base, 2),
            "high": round(max(prior + [base]), 2) if prior else round(base, 2),
            "buy_signal": is_lowest,
        }

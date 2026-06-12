"""Oil Trading Desk dashboard — FastAPI backend.

Runs a market simulation, computes indicators, and streams a full dashboard
snapshot to every connected browser over a WebSocket. A REST endpoint serves
the same snapshot for the initial page load."""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import ais
import composite
import config
import cot
import datafeed
import eia
import fred
import hurricane
import real_curves
import seasonality
import sentiment
import steo
import twelvedata
import twitter_nitter
from fundamentals import Fundamentals
from indicators import bollinger_bands, correlation, covariance, \
    covariance_matrix, ema, ewma_correlation_matrix, moving_average, \
    returns, vwap, zscore
from market import MarketEngine
from news import NewsFeed
from paper import PaperBook

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

TICK_SECONDS = 2.0          # live price tick / snapshot push interval
# Yahoo (kept at proven-safe cadence — 1/min for prices is at the rate-limit
# boundary; going lower risks 429s and gives no fresher data than Yahoo's
# 15-min delayed feed already does).
PRICE_EVERY_TICKS = 30      # WTI/Brent every ~60s — Yahoo rate-limit floor
CURVE_EVERY_TICKS = 30      # 12-month futures curve every ~60s.
                            # Each fetch = 12 yfinance tickers, so this is
                            # ~12 req/min just for the curve. Plus prices
                            # = ~14 yfinance req/min total — still under
                            # Yahoo's ~100/min ceiling but approaching it.
# Everything below = aggressive minimum to catch releases as fast as possible
NEWS_EVERY_TICKS = 8        # RSS news every ~16s (was 24s)
DXY_EVERY_TICKS = 150       # DXY every ~5 min (was 15 min)
                            # Twelve Data free tier: 800 calls/day budget
EIA_EVERY_TICKS = 900       # EIA every ~30 min (was 6h)
                            # WPSR publishes Wed 10:30am ET — catches it
                            # within 30 min instead of 6h
COT_EVERY_TICKS = 900       # CFTC COT every ~30 min (was 12h)
                            # publishes Fri 3:30pm ET — captured in 30 min
STEO_EVERY_TICKS = 1800     # STEO every ~60 min (unchanged; monthly publish)
ANALYST_EVERY_TICKS = 450   # Google News analyst feeds every ~15 min
HURRICANE_EVERY_TICKS = 300 # NOAA NHC storms every ~10 min (was 15 min)
FIVE_YEAR_EVERY_TICKS = 10800   # 5y same-week closes every ~6h (was daily)
FUNDAMENTALS_EVERY_TICKS = 150  # no-op when EIA active; keep cheap
HISTORY_REFRESH_EVERY_TICKS = 10800  # refresh 1y yfinance history every ~6h
                                     # so the covariance matrix evolves with
                                     # new daily closes instead of staying
                                     # frozen at startup-snapshot values
FRED_EVERY_TICKS = 3600              # FRED manufacturing indicators every
                                     # ~2h — regional Fed surveys publish
                                     # monthly; checking 12x/day catches
                                     # the release within ~2h of publication

MONTHLY_STORAGE_COST = 0.55  # $/bbl/month, used by the storage trade signal


class Hub:
    """Owns market state and the set of connected WebSocket clients."""

    def __init__(self) -> None:
        self.market = MarketEngine()
        self.fundamentals = Fundamentals()
        self.news = NewsFeed()
        self.cot: Optional[Dict] = None    # set by refresh_cot()
        self.analyst_news: Dict = {}       # set by refresh_analyst()
        self.refinery_history: List = []   # 5y weekly utilization (EIA)
        self.seasonality: Dict = {"available": False}
        self.tankers = ais.TankerTracker()  # live AIS tanker positions
        self.storms = hurricane.StormTracker()  # live NOAA NHC storms
        self.steo: Optional[Dict] = None    # EIA STEO monthly balance
        self.manufacturing: Optional[Dict] = None  # FRED PMI proxy
        # PAPER_STATE_REPO + HF_TOKEN are set as HF Space secrets; if either
        # is missing, PaperBook quietly falls back to local-only state.
        self.paper = PaperBook(
            remote_repo=os.environ.get("PAPER_STATE_REPO", ""),
            remote_token=os.environ.get("HF_TOKEN", ""),
        )
        # PAPER_RESET_AT_UTC (ISO 8601, e.g. "2026-05-26T13:30:00+00:00") —
        # the paper book auto-resets to $100k the first tick after this moment.
        # Used to wipe holiday-period drift trades the moment real market
        # data resumes.
        reset_iso = os.environ.get("PAPER_RESET_AT_UTC", "").strip()
        if reset_iso:
            try:
                when = dt.datetime.fromisoformat(reset_iso.replace("Z", "+00:00"))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=dt.timezone.utc)
                self.paper.schedule_reset(when.timestamp())
            except Exception:
                pass
        self.clients: Set[WebSocket] = set()
        self.prev_header: Dict[str, float] = {}
        self.tick = 0

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


hub = Hub()


# ---------------------------------------------------------------------- #
# snapshot assembly
# ---------------------------------------------------------------------- #
def _trend(key: str, value: float) -> str:
    prev = hub.prev_header.get(key)
    hub.prev_header[key] = value
    if prev is None or abs(value - prev) < 1e-9:
        return "flat"
    return "up" if value > prev else "down"


def _gasoil_header_entry() -> Dict:
    """Header-strip entry for Gas Oil — REAL LGO M1 from your xlsx.

    Replaces the prior `(brent + $7) * 7.45` synthetic estimate. Frozen
    at the file's last_date; tagged so the staleness is visible."""
    curve = real_curves.get_curve("gasoil")
    last_dt = real_curves.get_last_date("gasoil")
    if not curve:
        return {"label": "Gas Oil (n/a)", "value": 0.0,
                "unit": "$/mt", "trend": "flat"}
    val = float(curve[0]["price"])
    label = (f"Gas Oil <span class='asof-tag'>(LGO M1, as of {last_dt})</span>"
             if last_dt else "Gas Oil (LGO M1)")
    return {"label": label, "value": round(val, 1),
            "unit": "$/mt", "trend": _trend("gasoil", val)}


def _build_bb_panel(price_now: float, hist: list, precision: int = 2) -> dict:
    """BB(20, 2σ) panel for any price series. Returns price/upper/middle/lower,
    position 0-100%, state label, and last-90 history for the chart."""
    mid, upper, lower = bollinger_bands(hist, window=20, k=2.0)
    bb_m = mid[-1]   if mid[-1]   is not None else price_now
    bb_u = upper[-1] if upper[-1] is not None else price_now
    bb_l = lower[-1] if lower[-1] is not None else price_now
    width = bb_u - bb_l
    pos = ((price_now - bb_l) / width * 100.0) if width > 0 else 50.0
    if pos >= 100: state = "above upper band"
    elif pos <= 0: state = "below lower band"
    elif pos >= 80: state = "near upper"
    elif pos <= 20: state = "near lower"
    else: state = "within bands"
    return {
        "price":  round(price_now, precision),
        "upper":  round(bb_u, precision),
        "middle": round(bb_m, precision),
        "lower":  round(bb_l, precision),
        "position": round(pos, 1),
        "width":  round(width, precision),
        "state":  state,
        "history": {
            "labels": list(range(len(hist)))[-90:],
            "price":  [round(x, precision) for x in hist[-90:]],
            "upper":  [round(x, precision) if x is not None else None
                       for x in upper[-90:]],
            "middle": [round(x, precision) if x is not None else None
                       for x in mid[-90:]],
            "lower":  [round(x, precision) if x is not None else None
                       for x in lower[-90:]],
        },
    }


def build_snapshot() -> dict:
    m = hub.market

    wti_hist = m.series("wti")
    brent_hist = m.series("brent")
    dxy_hist = m.series("dxy")
    bdti_hist = m.series("bdti")
    spread_hist = m.series("spread")

    # --- 02 price chart + moving averages ----------------------------- #
    # Three "fair value" lenses on price: EMA20 weights recency, MA50 is
    # the long trend, and VWAP20 weights by traded volume (big-volume days
    # dominate). EMA20 + MA50 are mirrored for Brent so the chart isn't
    # WTI-only. VWAP stays WTI-only because we only store WTI volume.
    ema20         = ema(wti_hist, 20)
    ma50          = moving_average(wti_hist, 50)
    vwap20        = vwap(wti_hist, m.wti_volume_hist, 20)
    brent_ema20   = ema(brent_hist, 20)
    brent_ma50    = moving_average(brent_hist, 50)
    show = 130                                  # points shown on the chart
    price_panel = {
        "labels":      list(range(len(wti_hist)))[-show:],
        "wti":         wti_hist[-show:],
        "brent":       brent_hist[-show:],
        "ema20":       ema20[-show:],
        "ma50":        ma50[-show:],
        "vwap20":      vwap20[-show:],
        "brent_ema20": brent_ema20[-show:],
        "brent_ma50":  brent_ma50[-show:],
    }

    # --- 03 Bollinger Bands (20, 2σ) --------------------------------- #
    # Volatility-bracketed price. More relevant for our mean-reversion
    # strategies than RSI: when a product rides the upper band, it's "rich vs
    # current volatility" — the same setup our crack-spread signals trade.
    bb_panel        = _build_bb_panel(m.wti,   wti_hist,   precision=2)
    bb_brent_panel  = _build_bb_panel(m.brent, brent_hist, precision=2)
    rbob_hist  = list(m.hist["rbob"])
    heat_hist  = list(m.hist["heat"])
    bb_rbob_panel = _build_bb_panel(m.rbob, rbob_hist, precision=4) \
        if len(rbob_hist) >= 20 else None
    bb_heat_panel = _build_bb_panel(m.heat, heat_hist, precision=4) \
        if len(heat_hist) >= 20 else None
    # Position used by the composite header tile is WTI-only (anchor product)
    bb_pos = bb_panel["position"]

    # --- 04 WTI-Dollar correlation / covariance ----------------------- #
    wti_ret = returns(wti_hist)
    dxy_ret = returns(dxy_hist)
    corr = correlation(wti_ret[-90:], dxy_ret[-90:])
    cov = covariance(wti_ret[-90:], dxy_ret[-90:])
    dxy_panel = {
        "correlation": round(corr, 3),
        "covariance": round(cov * 1e4, 3),       # scaled bp^2 for readability
        "explained": round(min(0.99, corr * corr) * 100, 1),
        "wti": [round(x, 2) for x in wti_hist[-90:]],
        "dxy": [round(x, 2) for x in dxy_hist[-90:]],
    }

    # --- 05 WTI-Brent spread ------------------------------------------ #
    sp_now = spread_hist[-1]
    sp_window = spread_hist[-120:]
    sp_mean = sum(sp_window) / len(sp_window)
    sp_sd = (sum((x - sp_mean) ** 2 for x in sp_window)
             / max(1, len(sp_window) - 1)) ** 0.5
    sp_z = (sp_now - sp_mean) / sp_sd if sp_sd else 0.0
    spread_panel = {
        "value": round(sp_now, 2),
        "mean": round(sp_mean, 2),
        "zscore": round(sp_z, 2),
        "series": [round(x, 2) for x in spread_hist[-show:]],
        "state": ("unusually wide" if sp_z > 1.3 else
                  "unusually narrow" if sp_z < -1.3 else "in normal range"),
    }

    # --- 06 futures curve --------------------------------------------- #
    curve = m.futures_curve()
    m12_spread = curve[-1]["price"] - m.wti
    structure = "Contango" if m12_spread > 0.05 else (
        "Backwardation" if m12_spread < -0.05 else "Flat")
    futures_panel = {
        "curve": curve,
        "spot": round(m.wti, 2),
        "structure": structure,
        "m12_spread": round(m12_spread, 2),
        "monthly_carry": round(m12_spread / 12, 3),
    }

    # --- 07 freight (BDTI) -------------------------------------------- #
    freight_panel = {
        "value": round(m.bdti, 1),
        "series": [round(x, 1) for x in bdti_hist[-show:]],
        "avg90": round(sum(bdti_hist[-90:]) / min(90, len(bdti_hist)), 1),
    }

    # --- Product terminals (RBOB, Heating Oil, Gas Oil) ----------- #
    # Real Yahoo prices for front-month NYMEX RB and HO. Gas Oil has no
    # free Yahoo equivalent — that tile gets a clear "paid feed only"
    # note so we don't fake it.
    products_panel = {
        "rbob": {
            "price": round(m.rbob, 4),
            "anchor": (round(m.anchor_rbob, 4)
                       if m.anchor_rbob is not None else None),
            "unit": "$/gal",
            "ticker_yahoo": "RB=F",
            "ticker_tv":    "NYMEX:RB1!",
            "label":  "RBOB Gasoline",
            "venue":  "NYMEX",
            "source": ("Yahoo Finance RB=F (~15-min delayed) "
                       "+ TradingView NYMEX:RB1! widget"),
        },
        "heat": {
            "price": round(m.heat, 4),
            "anchor": (round(m.anchor_heat, 4)
                       if m.anchor_heat is not None else None),
            "unit": "$/gal",
            "ticker_yahoo": "HO=F",
            "ticker_tv":    "NYMEX:HO1!",
            "label":  "Heating Oil / ULSD",
            "venue":  "NYMEX",
            "source": ("Yahoo Finance HO=F (~15-min delayed) "
                       "+ TradingView NYMEX:HO1! widget"),
        },
        "gasoil": {
            "price": None,        # no free real feed for ICE Gas Oil
            "anchor": None,
            "unit": "$/mt",
            "ticker_yahoo": None,
            "ticker_tv":    "ICEEUR:GAS1!",
            "label":  "Gas Oil (ICE LGO)",
            "venue":  "ICE Europe",
            "source": ("No free Yahoo feed for ICE Low Sulphur Gasoil. "
                       "TradingView widget shown only — may fall back to "
                       "delayed quote on free tier."),
        },
    }

    # --- 08 crack spreads --------------------------------------------- #
    cracks = m.crack_spreads()

    # --- 09 spread covariance matrix ---------------------------------- #
    cov_series = {
        "WTI-Brent": spread_hist[-120:],
        "3-2-1 Crack": (m.crack_hist.get("3-2-1 Crack (USGC)", []) or
                        [0])[-120:],
        "Gasoline Crk": (m.crack_hist.get("Gasoline Crack (RBOB)", []) or
                         [0])[-120:],
        "Diesel Crk": (m.crack_hist.get("Diesel Crack (Heating Oil)", []) or
                       [0])[-120:],
        # Brent-Dubai EFS removed — Dubai is synthesized as (Brent - 1.9 + noise)
        # because there's no free real Dubai feed. Per user directive: real only.
        "DXY": dxy_hist[-120:],
    }
    # EWMA correlation (lam=0.94, RiskMetrics standard) — recent observations
    # weighted exponentially more, so the matrix actually responds to current
    # tick movements instead of barely shifting under 120-day equal weights.
    cov_matrix = ewma_correlation_matrix(cov_series, lam=0.94)

    # --- 10 fundamentals ---------------------------------------------- #
    fundamentals = hub.fundamentals.cards(m.crude_inventory)

    # --- 11 trade signals --------------------------------------------- #
    signals = build_signals(m, cracks, m12_spread, sp_now, sp_z)

    # --- 11B regime-aware butterfly (Lasso per regime + LogReg classifier) #
    # Builds 5-regime curve classification (Steep Backwardation / Backwardation
    # / Flat / Contango / Steep Contango), fits a Lasso per regime to predict
    # next-tick fly value from BB/crack/DXY/inventory/return factors, and
    # surfaces a recommendation. Sparse Lasso weights tell which factors
    # actually matter inside the current regime.
    import regime_fly
    _fly = m.butterfly_value()
    if _fly is not None and len(m.fly_history) >= 8:
        _fly_z = zscore(m.fly_history + [_fly], window=60)
    else:
        _fly_z = 0.0
    _crack_obj = next((c for c in cracks
                       if c.get("name") == "3-2-1 Crack (USGC)"), None)
    _crack_val = float(_crack_obj["value"]) if _crack_obj else 0.0
    regime_butterfly = regime_fly.build_panel(m, _fly_z, _crack_val)

    # --- 11C multi-product strategy matrix (WTI/Brent/RBOB/HO/NatGas) ----- #
    import multi_product
    multi_product_matrix = multi_product.build_matrix(m)

    # --- 11D Phase 2: multi-dim regime fingerprint + historical DB + ----- #
    # multi-model regression comparison + opportunity ranking
    import regime_classifier
    import regime_history
    import multi_model
    import opportunity_engine
    import multi_product_engine
    import real_data_engine
    import best_per_product_engine
    import paper_strats_engine
    import percentile_regime_engine
    _phase2_curve_prices = [float(row.get("price", 0.0))
                            for row in (m.real_curve or [])]
    phase2_regime = regime_classifier.classify(
        price_hist=wti_hist,
        curve_prices=_phase2_curve_prices,
        inv_hist=list(m.hist.get("crude_inventory", [])),
        dxy_hist=dxy_hist,
    )
    phase2_history = regime_history.build_database(m)
    phase2_models  = multi_model.build_panel(
        m,
        current_slope=(_phase2_curve_prices[-1] - _phase2_curve_prices[0])
        if len(_phase2_curve_prices) >= 9 else 0.0,
    )
    phase2_opportunities = opportunity_engine.build_opportunities(
        m, db=phase2_history, regime=phase2_regime,
    )
    # 5-product LIVE SIGNAL panel (LONG/SHORT/FLAT from a 5y LGBM fit on
    # each product + 3 oil-desk spreads). The build_panel call hits its
    # own 6h cache and is cheap on subsequent ticks.
    try:
        live_signals = multi_product_engine.build_panel()
    except Exception as _sig_err:
        live_signals = {"available": False,
                        "reason": f"engine error: {type(_sig_err).__name__}"}

    # REAL-DATA signal panel — winners from a 60/20/20 regression backtest
    # on the 5 user-supplied curve xlsx files. Static JSON load, no compute.
    try:
        real_data_panel = real_data_engine.build_panel()
    except Exception as _rde_err:
        real_data_panel = {"available": False,
                            "reason": f"engine error: {type(_rde_err).__name__}"}

    # BEST-PER-PRODUCT panel — for each product, use ONLY the model family
    # that wins the most targets in the backtest. This narrows the signal
    # set to high-conviction trades from each product's specialist model.
    try:
        best_pp_panel = best_per_product_engine.build_panel()
    except Exception as _bpp_err:
        best_pp_panel = {"available": False,
                          "reason": f"engine error: {type(_bpp_err).__name__}"}

    # PAPER-STRATEGIES panel — PCA Curve (CL/LCO/LGO), Bertram OU (HO),
    # HMM Regime (WTCL). Backtested OOS Sharpe 0.88 to 6.02 per product,
    # combined PnL +$202k on test slice. See tools/strategy_per_product_eval.py.
    try:
        pstrat_panel = paper_strats_engine.build_panel()
    except Exception as _pst_err:
        pstrat_panel = {"available": False,
                          "reason": f"engine error: {type(_pst_err).__name__}: {_pst_err}"[:200]}

    # PERCENTILE-REGIME panel — per-product quintile labels (Q1..Q5) computed
    # from each product's own slope distribution, shown side-by-side with the
    # legacy hard-cutoff labels. NO ML, just better labeling: hard cutoffs
    # were calibrated for WTI ($/bbl) and produce 90%+ class imbalance on
    # LGO ($/mt) and HO ($/gal). Per-product percentiles fix that.
    try:
        percentile_regime_panel = percentile_regime_engine.build_panel()
    except Exception as _pct_err:
        percentile_regime_panel = {"available": False,
                                    "reason": f"engine error: {type(_pct_err).__name__}"}

    # composite_panel is built AFTER term_structure / technical / news_signals
    # have been computed (see below, just before the snapshot dict).

    # --- 11E Term Structure: user-supplied regression results --------- #
    # Pre-trained models across the full strip (M1-M14) for 5 products.
    # Generates regime-aware trade signals + lot sizing per spread/fly.
    try:
        from . import term_structure
    except ImportError:
        import term_structure
    term_structure_panel = term_structure.build_panel(m)

    # --- 11F Technical analysis: per-product TA trade signals -------- #
    # RSI/BB/EMA-cross/momentum across the 5 oil-complex front months.
    # Independent of the regression / regime engines so the user gets a
    # second-opinion trade verdict per product.
    try:
        from . import technical_signals
    except ImportError:
        import technical_signals
    technical_panel = technical_signals.build_panel(m)

    # --- 11G News-aware trade signals -------------------------------- #
    # Routes recent news to each product via keyword matching, fuses
    # FinBERT sentiment + impact bias with existing BB / fly z-scores,
    # emits per-product LONG/SHORT/HOLD verdicts with vol-banded plans.
    try:
        from . import news_signals
    except ImportError:
        import news_signals
    _ns_news = []
    for it in hub.news.snapshot():
        _ns_news.append({**it, "ts": round(float(it["ts"]))})
    news_signals_panel = news_signals.build_panel(
        m, _ns_news, bb_pos=bb_pos, fly_z=_fly_z,
    )

    # --- COMPOSITE multi-factor strategy (must run BEFORE auto_trader) -- #
    try:
        try:
            from . import composite_strategy
        except ImportError:
            import composite_strategy
        composite_panel = composite_strategy.build_panel(
            market=m,
            live_signals=live_signals,
            real_data_panel=real_data_panel,
            technical=technical_panel,
            news_signals_panel=news_signals_panel,
            regime_butterfly=regime_butterfly,
            term_structure=term_structure_panel,
        )
    except Exception as _comp_err:
        composite_panel = {"available": False,
                            "reason": f"engine error: {type(_comp_err).__name__}: {_comp_err}"[:200]}

    # --- 14 paper trading: react to signal state changes -------------- #
    # Original 4 surviving strategies + auto-trader signals from the new
    # term-structure regression engine (P3B) and technical-analysis engine
    # (P3E). Auto-trader signals carry their own asset_key so the paper
    # book can mark them to market independently.
    assets = {
        "wti_spot": m.wti,
        "butterfly_fly": m.butterfly_value(),
        "brent_wti_spread": sp_now,
        "kalman_residual": m.kalman_live_residual(),
    }
    try:
        from . import auto_trader
    except ImportError:
        import auto_trader
    auto_signals, auto_assets = auto_trader.build(
        m, term_structure_panel, technical_panel,
        real_data_panel=real_data_panel,
        composite_panel=composite_panel,
        best_pp_panel=best_pp_panel,
        pstrat_panel=pstrat_panel,
        top_n=10, real_top_n=12,
    )
    assets.update(auto_assets)
    merged_signals = list(signals) + auto_signals

    # don't open new positions until the simulation has had time to converge
    # to its real-data anchors (otherwise we'd trade against bootstrap noise)
    warmed = hub.tick > 30 and m.anchor_wti is not None
    hub.paper.update(merged_signals, assets, live=warmed)
    paper_snapshot = hub.paper.snapshot(assets)

    # --- 12 five-year week range -------------------------------------- #
    five_year = m.five_year_week()

    # --- 13 news ------------------------------------------------------ #
    news_items = []
    for it in hub.news.snapshot():
        news_items.append({**it, "ts": round(float(it["ts"]))})

    # --- 01 header strip (9 key numbers) ------------------------------ #
    crack321 = next(c["value"] for c in cracks
                    if c["name"] == "3-2-1 Crack (USGC)")
    header = [
        {"label": "WTI Crude", "value": round(m.wti, 2), "unit": "$/bbl",
         "trend": _trend("wti", m.wti)},
        {"label": "Brent Crude", "value": round(m.brent, 2), "unit": "$/bbl",
         "trend": _trend("brent", m.brent)},
        {"label": "WTI-Brent", "value": round(sp_now, 2), "unit": "$",
         "trend": _trend("spread", sp_now)},
        # Refined products — RBOB + HO real Yahoo (NYMEX RB=F / HO=F).
        # Gas Oil derived: ICE LGO ≈ Brent + $7 distillate crack, converted
        # to $/mt via 7.45 bbl/mt for middle distillate (~840 kg/m³).
        # No free real feed for ICE Gas Oil — tagged "est" to be honest.
        {"label": "RBOB Gasoline", "value": round(m.rbob, 3), "unit": "$/gal",
         "trend": _trend("rbob", m.rbob)},
        {"label": "Heating Oil", "value": round(m.heat, 3), "unit": "$/gal",
         "trend": _trend("heat", m.heat)},
        _gasoil_header_entry(),
        {"label": "Dollar Index", "value": round(m.dxy, 2), "unit": "DXY",
         "trend": _trend("dxy", m.dxy)},
        # Freight BDTI removed — was SIMULATED (Baltic Exchange data is paid-
        # only, no free real feed). Honest absence beats fake number.
        {"label": "Crude Stocks", "value": round(m.crude_inventory, 1),
         "unit": "M bbl", "trend": _trend("inv", m.crude_inventory)},
        {"label": "BB Position", "value": round(bb_pos, 1), "unit": "%",
         "trend": _trend("bb_pos", bb_pos)},
        {"label": "3-2-1 Crack", "value": round(crack321, 2), "unit": "$/bbl",
         "trend": _trend("crack", crack321)},
        {"label": "Curve", "value": structure, "unit": "",
         "trend": "flat"},
    ]

    snap = {
        "ts": round(time.time()),
        "tick": hub.tick,
        "sources": {"news": hub.news.source,
                    "fundamentals": hub.fundamentals.source,
                    "prices": hub.market.price_source,
                    "dollar": hub.market.dxy_source,
                    "cracks": hub.market.crack_source,
                    "curve": hub.market.curve_source,
                    # freight source removed — BDTI was the only field that
                    # needed it, and BDTI was removed from the header (paid
                    # Baltic Exchange data, no free real feed available).
                    "rigs": "EIA rotary rigs (oil+gas, monthly, ~3mo lag)",
                    "five_year": hub.market.five_year_source,
                    "cot": ((hub.cot or {}).get("source") or
                            "CFTC (pending first fetch)"),
                    "paper": ("HF Dataset (persistent across redeploys)"
                              if hub.paper.remote_active
                              else "local (resets on redeploy)")},
        "header": header,
        "price": price_panel,
        "bb": bb_panel,
        "bb_brent": bb_brent_panel,
        "bb_rbob":  bb_rbob_panel,
        "bb_heat":  bb_heat_panel,
        "dxy": dxy_panel,
        "spread": spread_panel,
        "futures": futures_panel,
        "freight": freight_panel,
        "cracks": cracks,
        "covmatrix": cov_matrix,
        "fundamentals": fundamentals,
        "signals": signals,
        "regime_butterfly": regime_butterfly,
        "multi_product_matrix": multi_product_matrix,
        "phase2_regime":        phase2_regime,
        "phase2_history":       phase2_history,
        "phase2_models":        phase2_models,
        "phase2_opportunities": phase2_opportunities,
        "live_signals":         live_signals,
        "real_data_panel":      real_data_panel,
        "best_per_product":     best_pp_panel,
        "paper_strats":         pstrat_panel,
        "percentile_regime":    percentile_regime_panel,
        "term_structure":       term_structure_panel,
        "technical_signals":    technical_panel,
        "news_signals":         news_signals_panel,
        "composite_panel":      composite_panel,
        "fiveyear": five_year,
        "news": news_items,
        "news_regions": hub.news.by_region(),
        "news_sentiment": hub.news.sentiment_summary(),
        "analyst_news": hub.analyst_news,
        "curve_matrix": build_curve_matrix(curve, m.wti),
        "spread_covmatrix": build_spread_covariance(m.curve_hist),
        "seasonality": hub.seasonality,
        "finbert": sentiment.finbert_status(),
        "paper": paper_snapshot,
        "cot": hub.cot,
        "tankers": hub.tankers.snapshot(),
        "storms": hub.storms.snapshot(),
        "steo": hub.steo,
        "manufacturing": hub.manufacturing,
        "products": products_panel,
        # All 4 real 12-month curves side by side for the new panel.
        "product_curves": {
            "wti":    m.real_curve,
            "brent":  m.brent_curve,
            "rbob":   m.rbob_curve,
            "heat":   m.heat_curve,
            "natgas": m.natgas_curve,
        },
        "commodity_correlation": _build_commodity_correlation(m),
    }
    # Composite signal needs the assembled snapshot; compute last and attach.
    snap["composite_signal"] = composite.compute(snap)
    return snap


def build_signals(m: MarketEngine, cracks: List[dict],
                  m12_spread: float, sp_now: float, sp_z: float) -> List[dict]:
    """Four data-driven trade ideas:
        1. Diesel refining arbitrage (mean-reversion at z>=1σ)
        2. Crude storage cash-and-carry (contango > storage cost)
        3. Curve butterfly mean-reversion (M3-M6-M9 fly, z>=1.5σ)
        4. Brent-WTI spread mean-reversion (extreme z>=2σ)
    """
    diesel = next(c for c in cracks
                  if c["name"] == "Diesel Crack (Heating Oil)")
    dz = float(diesel["zscore"])
    if dz <= -1.0:
        d_status, d_dir = "ACTIVE", "LONG diesel crack"
        d_text = ("Diesel crack is {:.1f}σ cheap vs its 90-day mean — buy "
                  "heating oil / sell crude to capture mean reversion.")
    elif dz >= 1.0:
        d_status, d_dir = "ACTIVE", "SHORT diesel crack"
        d_text = ("Diesel crack is {:.1f}σ rich vs its 90-day mean — sell "
                  "heating oil / buy crude as refining margin normalises.")
    else:
        d_status, d_dir = "WATCHING", "Diesel crack arbitrage"
        d_text = ("Diesel crack is {:.1f}σ from its mean — inside the normal "
                  "band, no edge yet. Trigger at ±1.0σ.")

    carry_total = MONTHLY_STORAGE_COST * 12
    carry_edge = m12_spread - carry_total
    if carry_edge > 0.3:
        s_status, s_dir = "ACTIVE", "LONG storage (cash & carry)"
        s_text = ("12-month contango of ${:.2f} beats ${:.2f} all-in storage "
                  "cost — buy spot, sell the 12M future, store the barrels.")
    elif m12_spread > 0:
        s_status, s_dir = "WATCHING", "Storage play"
        s_text = ("Contango is ${:.2f} but storage costs ${:.2f} — carry is "
                  "not yet profitable. Watching for the curve to steepen.")
    else:
        s_status, s_dir = "WATCHING", "Storage play"
        s_text = ("Curve is in backwardation (${:.2f}) — storage economics "
                  "are negative. No storage trade while the curve is inverted.")

    # --- Strategy 5: 3-2-1 USGC Crack mean-reversion ---------------- #
    crack321 = next(c for c in cracks if c["name"] == "3-2-1 Crack (USGC)")
    c321_z = float(crack321["zscore"])
    c321_val = float(crack321["value"])
    if c321_z <= -1.25:
        c_status, c_dir = "ACTIVE", "LONG 3-2-1 crack"
        c_text = ("3-2-1 USGC crack is {:.1f}σ cheap (currently ${:.2f}/bbl) "
                  "— buy refined products / sell crude to capture margin "
                  "expansion.")
    elif c321_z >= 1.25:
        c_status, c_dir = "ACTIVE", "SHORT 3-2-1 crack"
        c_text = ("3-2-1 USGC crack is {:.1f}σ rich (currently ${:.2f}/bbl) "
                  "— sell refined products / buy crude as refining margins "
                  "normalise.")
    else:
        c_status, c_dir = "WATCHING", "3-2-1 crack mean-revert"
        c_text = ("3-2-1 crack is {:.1f}σ from mean (${:.2f}/bbl) — inside "
                  "±1.25σ band, no edge yet.")

    # --- Strategy 7: Kalman WTI-DXY (dynamic hedge ratio) ----------- #
    # Mean-reverts the residual of  WTI = alpha_t + beta_t * DXY  where
    # alpha and beta are tracked online by a Kalman filter. Backtested
    # Sharpe ≈ 1.5, 100% win in 6/6 walk-forward windows.
    k_resid_hist = m.kalman_resid_history
    k_resid_live = m.kalman_live_residual()
    k_alpha, k_beta = m.kalman.snapshot()
    if len(k_resid_hist) >= 20:
        k_z = zscore(k_resid_hist + [k_resid_live], window=90)
        if k_z >= 2.0:
            k_status, k_dir = "ACTIVE", "SHORT residual (WTI rich vs DXY)"
            k_text = ("Kalman residual is {:.1f}σ above mean (β_t={:+.2f}). "
                      "WTI overshot its current dollar relationship — short "
                      "WTI / long DXY for mean reversion.")
        elif k_z <= -2.0:
            k_status, k_dir = "ACTIVE", "LONG residual (WTI cheap vs DXY)"
            k_text = ("Kalman residual is {:.1f}σ below mean (β_t={:+.2f}). "
                      "WTI undershot its current dollar relationship — long "
                      "WTI / short DXY for catch-up.")
        else:
            k_status, k_dir = "WATCHING", "Kalman pair (dynamic β)"
            k_text = ("Kalman residual is {:.1f}σ from 0 (β_t={:+.2f}) — "
                      "inside ±2σ band, no decoupling.")
    else:
        k_status, k_dir = "WATCHING", "Kalman pair (dynamic β)"
        k_text = "Warming up Kalman state."
        k_z = 0.0

    # --- Strategy 6: WTI-DXY Pair Trade ----------------------------- #
    wti_hist_pair = m.hist["wti"][-90:]
    dxy_hist_pair = m.hist["dxy"][-90:]
    if min(len(wti_hist_pair), len(dxy_hist_pair)) >= 20:
        # crude and dollar are inversely correlated; their SUM has low
        # variance when that relationship holds. Z-score of the sum flags
        # decoupling episodes.
        combo_hist = [w + 0.5 * d
                      for w, d in zip(wti_hist_pair, dxy_hist_pair)]
        combo_now = m.wti + 0.5 * m.dxy
        pair_z = zscore(combo_hist + [combo_now], window=60)
        if pair_z >= 2.0:
            p_status = "ACTIVE"
            p_dir = "SHORT WTI (vs DXY)"
            p_text = ("WTI+DXY combo is {:.1f}σ above its mean — crude has "
                      "decoupled from the dollar's inverse relationship. "
                      "Short WTI as a mean-reversion bet.")
        elif pair_z <= -2.0:
            p_status = "ACTIVE"
            p_dir = "LONG WTI (vs DXY)"
            p_text = ("WTI+DXY combo is {:.1f}σ below mean — crude lagged "
                      "the dollar move. Long WTI to capture the catch-up.")
        else:
            p_status, p_dir = "WATCHING", "WTI-DXY pair trade"
            p_text = ("WTI-DXY combo is {:.1f}σ from mean — within ±2σ band, "
                      "the inverse relationship is holding normally.")
    else:
        p_status, p_dir = "WATCHING", "WTI-DXY pair trade"
        p_text = "Building combo history."
        pair_z = 0.0

    # --- Strategy 3: Curve Butterfly (M3-M6-M9 fly) ----------------- #
    fly = m.butterfly_value()
    fly_hist = m.fly_history
    if fly is None or len(fly_hist) < 8:
        f_status, f_dir = "WATCHING", "Curve butterfly"
        f_text = "Building fly history — need more curve observations."
        f_z = 0.0
    else:
        f_z = zscore(fly_hist + [fly], window=60)
        if f_z <= -1.5:
            f_status = "ACTIVE"
            f_dir = "LONG butterfly (M6 cheap)"
            f_text = ("M3-M6-M9 fly is {:.1f}σ below mean (currently ${:+.2f}) "
                      "— middle of the curve is rich. Buy fly: long M3, short "
                      "2× M6, long M9.")
        elif f_z >= 1.5:
            f_status = "ACTIVE"
            f_dir = "SHORT butterfly (M6 rich)"
            f_text = ("M3-M6-M9 fly is {:.1f}σ above mean (currently ${:+.2f}) "
                      "— middle of the curve is cheap. Sell fly: short M3, "
                      "long 2× M6, short M9.")
        else:
            f_status, f_dir = "WATCHING", "Curve butterfly mean-revert"
            f_text = ("Fly is {:.1f}σ from mean (currently ${:+.2f}) — inside "
                      "normal band, trigger at ±1.5σ.")

    # --- Strategy 4: Brent-WTI Spread (extreme z >= 2σ) -------------- #
    if sp_z >= 2.0:
        b_status = "ACTIVE"
        b_dir = "SHORT Brent-WTI spread"
        b_text = ("Brent-WTI is {:.1f}σ above its 120d mean (${:.2f}) — spread "
                  "is unusually wide. Short Brent / long WTI to capture "
                  "convergence.")
    elif sp_z <= -2.0:
        b_status = "ACTIVE"
        b_dir = "LONG Brent-WTI spread"
        b_text = ("Brent-WTI is {:.1f}σ below its 120d mean (${:.2f}) — spread "
                  "is unusually narrow. Long Brent / short WTI for "
                  "normalisation.")
    else:
        b_status, b_dir = "WATCHING", "Brent-WTI extreme z-trade"
        b_text = ("Brent-WTI is {:.1f}σ from mean (${:.2f}) — within ±2σ band, "
                  "no extreme dislocation to trade.")

    # Strategy roster after the backtest_results.json prune. Dropped
    # strategies (Diesel Refining Arbitrage, 3-2-1 USGC Crack, WTI-DXY
    # Pair fixed-beta) all had negative out-of-sample Sharpe ratios — the
    # in-sample edge didn't generalize. The 4 keepers all have positive
    # Sharpe in both windows, or are structural plays (carry / curvature
    # mean-reversion) that don't rely on time-series backtest validation.
    return [
        {
            "title": "Crude Storage Carry",
            "status": s_status,
            "direction": s_dir,
            "rationale": s_text.format(m12_spread, carry_total),
            "metric": "12M contango vs storage cost",
            "metric_value": "${:+.2f}".format(carry_edge),
            "size_bbl": 1000,
        },
        {
            "title": "Curve Butterfly",
            "status": f_status,
            "direction": f_dir,
            "rationale": f_text.format(f_z, fly or 0.0),
            "metric": "M3-M6-M9 fly z-score",
            "metric_value": "{:+.2f}σ".format(f_z),
            "size_bbl": 10000,   # fly $ moves are small, scale up notional
        },
        {
            "title": "Brent-WTI Spread",
            "status": b_status,
            "direction": b_dir,
            "rationale": b_text.format(sp_z, sp_now),
            "metric": "WTI-Brent spread z-score",
            "metric_value": "{:+.2f}σ".format(sp_z),
            "size_bbl": 1000,
        },
        {
            "title": "Kalman Pair (dyn β)",
            "status": k_status,
            "direction": k_dir,
            "rationale": k_text.format(k_z, k_beta),
            "metric": "Kalman residual z-score",
            "metric_value": "{:+.2f}σ".format(k_z),
            "size_bbl": 1000,
        },
    ]


# ---------------------------------------------------------------------- #
# background simulation loop
# ---------------------------------------------------------------------- #
async def refresh_real_prices() -> None:
    """Pull the latest delayed Yahoo Finance prices and re-anchor the engine
    (WTI, Brent, plus RBOB gasoline and ULSD heating oil for real cracks)."""
    loop = asyncio.get_running_loop()
    wti, brent = await asyncio.wait_for(
        loop.run_in_executor(None, datafeed.fetch_latest), timeout=20)
    if wti or brent:
        hub.market.set_anchor(wti, brent)
    rbob, heat = await asyncio.wait_for(
        loop.run_in_executor(None, datafeed.fetch_products), timeout=20)
    if rbob or heat:
        hub.market.set_product_anchors(rbob, heat)


async def refresh_product_curves() -> None:
    """Pull real 12-month settlement curves for Brent, RBOB, Heating Oil
    and Natural Gas. All four use the same NYMEX symbol-composition
    pattern as WTI — same yfinance batched download path."""
    loop = asyncio.get_running_loop()
    m = hub.market
    # Each fetch is its own thread-pool call so a slow one doesn't block.
    fetchers = [
        ("brent_curve",  datafeed.fetch_brent_curve,  "brent"),
        ("rbob_curve",   datafeed.fetch_rbob_curve,   None),    # no xlsx for RBOB
        ("heat_curve",   datafeed.fetch_heat_curve,   "heat"),
        ("natgas_curve", datafeed.fetch_natgas_curve, None),    # no xlsx for NG
    ]
    # Real-curves xlsx fallback (used when Yahoo per-contract symbols return
    # NaN — common for back-month tenors). NO synthetic anywhere.
    try:
        try:
            from . import real_curves
        except ImportError:
            import real_curves
    except Exception:
        real_curves = None
    for attr, fn, xlsx_key in fetchers:
        with contextlib.suppress(Exception):
            curve = await asyncio.wait_for(
                loop.run_in_executor(None, fn), timeout=30)
            if curve:
                setattr(m, attr, curve)
                continue
            if real_curves is not None and xlsx_key:
                xc = real_curves.get_curve(xlsx_key)
                if xc:
                    setattr(m, attr, xc)


async def refresh_curve() -> None:
    """Pull the real 12-month WTI futures settlement curve from Yahoo."""
    loop = asyncio.get_running_loop()
    curve = await asyncio.wait_for(
        loop.run_in_executor(None, datafeed.fetch_curve), timeout=30)
    if curve:
        hub.market.set_real_curve(curve)


async def refresh_five_year() -> None:
    """Pull real WTI same-week closes for each of the past 5 years."""
    loop = asyncio.get_running_loop()
    years = await asyncio.wait_for(
        loop.run_in_executor(None, datafeed.fetch_5y_same_week), timeout=30)
    if years:
        hub.market.set_five_year(years)


async def refresh_cot() -> None:
    """Pull the latest weekly CFTC Commitment of Traders report for NYMEX WTI."""
    data = await cot.fetch_cot()
    if data:
        hub.cot = data


async def refresh_analyst() -> None:
    """Pull REAL tweets from the three tracked analysts (Amena Bakr,
    Javier Blas, Trump) via Nitter RSS — Twitter's own API is paid-only
    since 2023, but Nitter exposes public timelines as RSS for free.

    Per-analyst fallback: if Nitter returns nothing for a specific
    analyst (rate-limited, common for Trump's account), substitute
    Google News mention coverage just for that analyst. Bakr/Blas
    almost always come through as real tweets; Trump frequently falls
    back to Google News because every Nitter instance rate-limits his
    handle."""
    try:
        tweets = await twitter_nitter.fetch_analyst_tweets()
    except Exception:
        tweets = {}
    empty_analysts = [a for a, p in tweets.items()
                      if not (p or {}).get("items")]
    if empty_analysts:
        try:
            gn_data = await hub.news.fetch_analyst_news()
        except Exception:
            gn_data = {}
        for analyst in empty_analysts:
            fallback = (gn_data or {}).get(analyst)
            if fallback and fallback.get("items"):
                tweets[analyst] = fallback
    if tweets:
        hub.analyst_news = tweets


async def refresh_manufacturing() -> None:
    """Pull FRED manufacturing-health indicators (PMI proxy)."""
    data = await fred.fetch_manufacturing(config.FRED_API_KEY)
    if data:
        hub.manufacturing = data


async def refresh_history() -> None:
    """Pull the latest 1y of yfinance daily closes and update the rolling
    history series so the spread / crack / DXY covariance matrix evolves
    as new daily closes are published. Runs every ~6h.

    Replaces hist['wti', 'brent', 'dxy', 'spread'] and the crack_hist
    series with freshly computed values from the new daily closes.
    Live anchor values (set by refresh_real_prices etc.) are untouched."""
    loop = asyncio.get_running_loop()
    try:
        hist = await asyncio.wait_for(
            loop.run_in_executor(None, datafeed.fetch_history),
            timeout=45)
    except Exception:
        return
    if not hist or not hist.get("wti"):
        return
    m = hub.market
    wti_series = [round(float(x), 2) for x in hist["wti"]]
    brent_series = [round(float(x), 2) for x in hist["brent"]]
    m.hist["wti"] = wti_series
    m.hist["brent"] = brent_series
    m.hist["spread"] = [round(b - w, 2)
                        for w, b in zip(wti_series, brent_series)]
    if "dxy" in hist:
        m.hist["dxy"] = [round(float(x), 2) for x in hist["dxy"]]
    if "wti_volume" in hist:
        m.wti_volume_hist = [float(x) for x in hist["wti_volume"]]
    # Recompute crack-spread histories from the refreshed real series so
    # the EWMA correlation matrix sees the new daily closes too.
    real_rbob = hist.get("rbob")
    real_heat = hist.get("heat")
    n = len(wti_series)
    new_crack_hist: dict = {}
    for i in range(n):
        w = wti_series[i]
        b = brent_series[i]
        rbob_i = (real_rbob[i] if real_rbob and i < len(real_rbob)
                  else m.rbob)
        heat_i = (real_heat[i] if real_heat and i < len(real_heat)
                  else m.heat)
        dubai_i = b - 1.9
        cr = m._crack_set(w, b, rbob_i, heat_i, dubai_i)
        for name, val in cr.items():
            new_crack_hist.setdefault(name, []).append(round(val, 2))
    m.crack_hist = new_crack_hist


async def refresh_steo() -> None:
    """Pull EIA Short-Term Energy Outlook — global oil supply/demand balance
    with ~18 months of forward forecast."""
    data = await steo.fetch_steo(config.EIA_API_KEY)
    if data:
        hub.steo = data


async def refresh_hurricane() -> None:
    """Pull active Atlantic storms from NOAA NHC and overlay onto US Gulf
    refineries / offshore production zone."""
    await hub.storms.refresh()


async def refresh_seasonality() -> None:
    """Pull 5y of EIA weekly refinery utilization and rebuild the seasonal
    pattern (week-of-year average + current week's deviation from norm)."""
    hist = await eia.fetch_refinery_history(config.EIA_API_KEY)
    if hist:
        hub.refinery_history = hist
        hub.seasonality = seasonality.build_summary(
            hist, latest_value=hub.fundamentals.utilization)


def _build_commodity_correlation(m: MarketEngine) -> Dict:
    """Correlation matrix between the 6 commodities we have ≥1y of real
    daily history for: WTI, Brent, RBOB, Heating Oil, DXY, Natural Gas.

    Uses Pearson correlation of daily RETURNS (so the magnitude reflects
    co-movement of % moves, not level co-movement). 6×6 symmetric matrix
    with 1.0 on the diagonal. Returns None until the histories are seeded
    so the panel cleanly says 'building...'."""
    series_map = {
        "WTI":    m.hist.get("wti")   or [],
        "Brent":  m.hist.get("brent") or [],
        "DXY":    m.hist.get("dxy")   or [],
        "RBOB":   m.hist.get("rbob")  or [],
        "HO":     m.hist.get("heat")  or [],
        "NatGas": m.hist.get("natgas") or [],
    }
    # Only include series that actually have ≥60 daily closes
    series = {k: v for k, v in series_map.items() if len(v) >= 60}
    if len(series) < 3:
        return {"labels": [], "matrix": [], "n_days": 0,
                "note": "warming up — fetching daily history"}
    # Align on a common minimum length (newest-aligned).
    # NOTE: covariance_matrix() internally takes RETURNS of whatever you
    # pass in, so we pass aligned PRICES, not pre-computed returns —
    # otherwise we'd be correlating returns-of-returns.
    n = min(len(v) for v in series.values())
    aligned = {k: v[-n:] for k, v in series.items()}
    cov = covariance_matrix(aligned)
    return {
        "labels": cov["labels"],
        "matrix": cov["matrix"],
        "n_days": n,
        "note": (f"Pearson correlation of daily returns over the last "
                 f"{n} trading days. Diagonal = 1.0 (a series perfectly "
                 "correlates with itself)."),
    }


def build_spread_covariance(curve_hist) -> Dict:
    """Plain sample covariance matrix between consecutive calendar spreads
    (M1-M2, M2-M3, ..., M11-M12), in $²/bbl² units.

    Diagonal = sample variance of each spread (= spread vol²).
    Off-diagonal = pairwise covariance.

    Equal-weighted (every snapshot counts the same), Bessel-corrected
    (divide by n-1) — the literal covariance matrix in the statistical
    sense, not an EWMA estimate.

    Useful for fly-trade variance:
        var(fly = M1 − 2*M2 + M3) = var(M1-M2) + var(M2-M3)
                                    − 2 * cov(M1-M2, M2-M3)
    Adjacent spreads share a common month and are anti-correlated, so
    cov(M1-M2, M2-M3) is typically negative — which makes the fly's
    variance LARGER than the sum of leg variances. That's the math
    behind why fly trades carry more risk than they look like.
    """
    snapshots = list(curve_hist)
    if len(snapshots) < 30:
        return {
            "warming_up": True,
            "n_snapshots": len(snapshots),
            "labels": [], "matrix": [], "max_abs": 0.0,
        }
    n_months = min(len(snapshots[-1]), 12)
    if n_months < 3:
        return {"warming_up": True, "n_snapshots": len(snapshots),
                "labels": [], "matrix": [], "max_abs": 0.0}

    # Build each calendar-spread time series
    spread_labels: List[str] = []
    spread_series: List[List[float]] = []
    for i in range(n_months - 1):
        spread_labels.append(f"M{i+1}-M{i+2}")
        spread_series.append([snap[i+1] - snap[i]
                              for snap in snapshots
                              if len(snap) > i + 1])

    # Plain equal-weighted sample covariance matrix
    matrix: List[List[float]] = []
    max_abs = 0.0
    for a in spread_series:
        row: List[float] = []
        for b in spread_series:
            cov = covariance(a, b)
            row.append(round(cov, 5))
            if abs(cov) > max_abs:
                max_abs = abs(cov)
        matrix.append(row)

    return {
        "warming_up": False,
        "n_snapshots": len(snapshots),
        "labels": spread_labels,
        "matrix": matrix,
        "max_abs": round(max_abs, 5),    # frontend uses this for colour scale
        "units": "$²/bbl²",
    }


def build_curve_matrix(curve: List[Dict[str, float]], spot: float) -> Dict:
    """Calendar-spread matrix:  M[i,j] = price(month_j) − price(month_i).
    Diagonal is zero. Includes 'Spot' as row/col 0. Real covariance would
    need months of historical curve snapshots we don't have — this is the
    structural information from the current curve only."""
    labels = ["Spot"] + [f"M{c['month']}" for c in curve]
    prices = [spot] + [c["price"] for c in curve]
    n = len(labels)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(round(prices[j] - prices[i], 2))
        matrix.append(row)
    return {"labels": labels, "prices": [round(p, 2) for p in prices],
            "matrix": matrix}


async def refresh_dxy() -> None:
    """Rebuild the real Dollar Index from Twelve Data forex pairs."""
    dxy = await twelvedata.fetch_dxy(config.TWELVE_DATA_API_KEY)
    if dxy:
        hub.market.set_dxy_anchor(dxy)


async def refresh_eia() -> None:
    """Pull real US crude stocks, Cushing, refinery use and OPEC supply."""
    data = await eia.fetch_fundamentals(config.EIA_API_KEY)
    if not data:
        return
    if "us_crude_stocks_mbbl" in data:
        hub.market.set_inventory_anchor(data["us_crude_stocks_mbbl"])
    hub.fundamentals.apply_eia(data)


async def simulation_loop() -> None:
    while True:
        hub.tick += 1
        hub.market.tick()

        if hub.tick % NEWS_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await hub.news.refresh()

        if hub.tick % PRICE_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_real_prices()

        if hub.tick % CURVE_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_curve()
            with contextlib.suppress(Exception):
                await refresh_product_curves()

        if hub.tick % DXY_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_dxy()

        if hub.tick % EIA_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_eia()

        if hub.tick % FIVE_YEAR_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_five_year()
            with contextlib.suppress(Exception):
                await refresh_seasonality()

        if hub.tick % COT_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_cot()

        if hub.tick % ANALYST_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_analyst()

        if hub.tick % HURRICANE_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_hurricane()

        if hub.tick % STEO_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_steo()

        if hub.tick % HISTORY_REFRESH_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_history()

        if hub.tick % FRED_EVERY_TICKS == 0:
            with contextlib.suppress(Exception):
                await refresh_manufacturing()

        if hub.tick % FUNDAMENTALS_EVERY_TICKS == 0:
            hub.fundamentals.weekly_update()

        with contextlib.suppress(Exception):
            await hub.broadcast(build_snapshot())

        await asyncio.sleep(TICK_SECONDS)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    # Seed the market with one year of REAL Yahoo Finance daily history.
    # Retry up to 5 times — if yfinance times out and we fall back to a
    # synthetic random-walk seed, the chart shows fake data that diverges
    # from reality, producing visible jumps when the live anchor lands.
    # Wait long enough between retries that yfinance has time to recover.
    hist = None
    for attempt in range(5):
        try:
            hist = await asyncio.wait_for(
                loop.run_in_executor(None, datafeed.fetch_history),
                timeout=45)
            if hist and hist.get("wti"):
                break
        except Exception:
            pass
        await asyncio.sleep(3)
    if hist and hist.get("wti"):
        hub.market = MarketEngine(real_history=hist)
    # If hist is still None after 5 tries, we fall through to the default
    # MarketEngine (synthetic seed) — the chart will be ugly but the rest
    # of the dashboard (EIA, COT, news, AIS, storms) still works.
    # anchor the live price to the latest real (delayed) quote. Retry up
    # to 3 times — a slow first yfinance call leaves the engine unanchored,
    # which causes the random-walk drift glitch (chart history shows $45
    # values that get committed before the real anchor arrives).
    for attempt in range(3):
        try:
            await refresh_real_prices()
            if hub.market.anchor_wti is not None:
                break
        except Exception:
            pass
        await asyncio.sleep(2)
    # pull the real 12-month WTI futures settlement curve
    with contextlib.suppress(Exception):
        await refresh_curve()
    # also pull Brent / RBOB / HO / NatGas 12-month settlement curves
    with contextlib.suppress(Exception):
        await refresh_product_curves()
    # pull 1y of REAL daily settlement curves for the spread covariance
    # matrix (no synthetic data, no noise — just real Yahoo settlements
    # per NYMEX contract joined on common trading days)
    # Yahoo's dated NYMEX symbols (CL<MM><YY>.NYM) only have ~1y of history
    # per contract — they list ~12 months before delivery. So requesting more
    # than 1y here does NOT give us more curve snapshots; the intersection of
    # common dates is capped by the shortest-lived listed contract.
    with contextlib.suppress(Exception):
        rows = await asyncio.wait_for(
            loop.run_in_executor(None, datafeed.fetch_curve_history),
            timeout=45)
        if rows:
            hub.market.set_curve_history(rows)
    # one-shot seed: pull 1y of daily settlement curves for the other 4
    # NYMEX products (Brent/RBOB/HO/NatGas) so the Phase 2 opportunity
    # engine has a per-product historical distribution to compute
    # regime-conditional spread/fly means against. Without this, P2C
    # shows regime mean "—" for everything except WTI.
    with contextlib.suppress(Exception):
        all_hist = await asyncio.wait_for(
            loop.run_in_executor(None, datafeed.fetch_all_curve_histories),
            timeout=90)
        if all_hist:
            for key in ("brent", "rbob", "heat", "natgas"):
                hub.market.set_product_curve_history(key, all_hist.get(key))
    # pull real same-week WTI closes for the 5-year week-range panel
    with contextlib.suppress(Exception):
        await refresh_five_year()
    # build 5y refinery utilization seasonal pattern. Retry up to 3 times
    # because EIA API frequently has 20-30s slowness windows on weekends/
    # holidays — a single timed-out fetch would leave the panel empty.
    for attempt in range(3):
        try:
            await refresh_seasonality()
            if hub.seasonality.get("available"):
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    # pull weekly CFTC Commitment of Traders positioning
    with contextlib.suppress(Exception):
        await refresh_cot()
    # pull Google News for tracked oil analysts
    with contextlib.suppress(Exception):
        await refresh_analyst()
    # pull active Atlantic storms from NOAA NHC (oil-asset overlay)
    with contextlib.suppress(Exception):
        await refresh_hurricane()
    # pull EIA STEO global oil balance (monthly, ~18mo forecast horizon).
    # Retry up to 3 times — see seasonality comment above.
    for attempt in range(3):
        try:
            await refresh_steo()
            if hub.steo is not None:
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    # pull FRED manufacturing-health indicators (PMI proxy). Retry up to
    # 4 times — FRED occasionally 504s mid-batch, leaving only some
    # series populated. Keep retrying until we get at least 6 of the
    # ~10 series. A partial fetch with 1-2 cards is worse than retrying.
    for attempt in range(4):
        try:
            await refresh_manufacturing()
            cards = (hub.manufacturing or {}).get("cards") or []
            if len(cards) >= 6:
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    # Kick off FinBERT model load in a background thread (~30s, ~440MB
    # weights). Falls back to VADER-only if torch/transformers missing or
    # the download fails — sentiment.classify_finbert() returns None until
    # the model is loaded.
    with contextlib.suppress(Exception):
        sentiment.warm_finbert()
    # build the real Dollar Index from Twelve Data forex pairs
    with contextlib.suppress(Exception):
        await refresh_dxy()
    # pull real EIA fundamentals (inventories, Cushing, refinery use, OPEC).
    # Retry up to 3 times — see seasonality comment above.
    for attempt in range(3):
        try:
            await refresh_eia()
            if "EIA" in hub.fundamentals.source:
                break
        except Exception:
            pass
        await asyncio.sleep(5)
    # warm the news feed once before serving
    with contextlib.suppress(Exception):
        await hub.news.refresh()
    # Open the AIS WebSocket as a long-running background task. It reconnects
    # on its own with backoff, so this never raises into the lifespan and
    # the dashboard still works if aisstream.io is down.
    ais_task = asyncio.create_task(hub.tankers.run(config.AIS_API_KEY))
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    hub.tankers.stop()
    ais_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await ais_task


app = FastAPI(title="Oil Trading Desk", lifespan=lifespan)


@app.get("/api/snapshot")
async def api_snapshot() -> dict:
    return build_snapshot()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    with contextlib.suppress(Exception):
        await ws.send_json(build_snapshot())   # immediate first paint
    try:
        while True:
            await ws.receive_text()            # keep-alive / client pings
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


@app.get("/")
async def index() -> FileResponse:
    # no-cache on index.html so the ?v= query-strings inside it always reach
    # the browser. The referenced JS/CSS can stay cached because the version
    # string changes whenever we ship a meaningful frontend update.
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")

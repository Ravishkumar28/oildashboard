"""Real price feed via yfinance (Yahoo Finance).

WTI = CL=F, Brent = BZ=F crude futures. Yahoo data is roughly 15 minutes
delayed and the source is unofficial, so every call is best-effort: any
failure returns None and the caller falls back to the simulation.

yfinance is a soft dependency — if it (or pandas) is missing the module just
reports itself unavailable rather than crashing the app."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

WTI_TICKER = "CL=F"
BRENT_TICKER = "BZ=F"
RBOB_TICKER = "RB=F"          # NYMEX RBOB gasoline futures, $/gal
HEAT_TICKER = "HO=F"          # NYMEX heating oil (ULSD) futures, $/gal
DXY_TICKER = "DX-Y.NYB"       # ICE U.S. Dollar Index, daily closes
NATGAS_TICKER = "NG=F"        # NYMEX Henry Hub natural gas, $/MMBtu

# NYMEX month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun
#                    N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
_MONTH_CODES = "FGHJKMNQUVXZ"

try:
    import pandas as pd
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


def _close_series(ticker: str, period: str):
    """Daily close series indexed by plain date (so two tickers on different
    exchange calendars/timezones still align cleanly)."""
    hist = yf.Ticker(ticker).history(period=period)
    series = hist["Close"].dropna()
    series.index = [idx.date() for idx in series.index]
    return series


def fetch_history(period: str = "1y") -> Optional[Dict[str, List[float]]]:
    """One year of real daily WTI/Brent/DXY closes + WTI daily volume,
    aligned on common dates. Volume is needed for VWAP.

    DXY is included so the WTI-Dollar correlation panel runs on real ICE
    Dollar Index closes (DX-Y.NYB) rather than synthetic bootstrap values.
    If DXY history fails or is sparse we still return WTI+Brent so the
    rest of the dashboard keeps working."""
    if not YF_AVAILABLE:
        return None
    try:
        wti_hist = yf.Ticker(WTI_TICKER).history(period=period)
        wti_close = wti_hist["Close"].dropna()
        wti_close.index = [d.date() for d in wti_close.index]
        wti_vol = wti_hist["Volume"].dropna()
        wti_vol.index = [d.date() for d in wti_vol.index]

        brent = _close_series(BRENT_TICKER, period).rename("brent")

        # ICE Dollar Index history — best-effort, optional column
        try:
            dxy = _close_series(DXY_TICKER, period).rename("dxy")
        except Exception:
            dxy = None

        # RBOB + Heat futures history — used to compute REAL crack-spread
        # history. Without these, the crack series are synthetic and the
        # covariance matrix uses fake correlations between cracks/crude.
        try:
            rbob = _close_series(RBOB_TICKER, period).rename("rbob")
        except Exception:
            rbob = None
        try:
            heat = _close_series(HEAT_TICKER, period).rename("heat")
        except Exception:
            heat = None

        # Natural gas — only non-crude energy series we pull for the
        # commodity correlation matrix (intentionally NOT pulling gold /
        # copper / treasuries; this dashboard stays oil-focused).
        try:
            natgas = _close_series(NATGAS_TICKER, period).rename("natgas")
        except Exception:
            natgas = None

        frames = [wti_close.rename("wti"), brent, wti_vol.rename("wti_volume")]
        if dxy is not None:    frames.append(dxy)
        if rbob is not None:   frames.append(rbob)
        if heat is not None:   frames.append(heat)
        if natgas is not None: frames.append(natgas)
        df = pd.concat(frames, axis=1).dropna()
        if len(df) < 60:
            return None
        out: Dict[str, List[float]] = {
            "wti":        [round(float(x), 2) for x in df["wti"].tolist()],
            "brent":      [round(float(x), 2) for x in df["brent"].tolist()],
            "wti_volume": [float(x) for x in df["wti_volume"].tolist()],
        }
        if "dxy" in df.columns:
            out["dxy"] = [round(float(x), 2) for x in df["dxy"].tolist()]
        if "rbob" in df.columns:
            out["rbob"] = [round(float(x), 4) for x in df["rbob"].tolist()]
        if "heat" in df.columns:
            out["heat"] = [round(float(x), 4) for x in df["heat"].tolist()]
        if "natgas" in df.columns:
            out["natgas"] = [round(float(x), 3) for x in df["natgas"].tolist()]
        return out
    except Exception:
        return None


def _last_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="15m")
        series = hist["Close"].dropna()
        if series.empty:
            hist = yf.Ticker(ticker).history(period="5d")
            series = hist["Close"].dropna()
        if series.empty:
            return None
        return round(float(series.iloc[-1]), 2)
    except Exception:
        return None


def fetch_latest() -> Tuple[Optional[float], Optional[float]]:
    """Most recent (~15-min delayed) WTI and Brent futures prices."""
    if not YF_AVAILABLE:
        return None, None
    return _last_price(WTI_TICKER), _last_price(BRENT_TICKER)


def fetch_products() -> Tuple[Optional[float], Optional[float]]:
    """Most recent RBOB gasoline and ULSD heating oil prices ($/gallon).

    Feeding these into the crack-spread calculation makes the entire
    refining-margin panel real instead of simulated."""
    if not YF_AVAILABLE:
        return None, None
    return _last_price(RBOB_TICKER), _last_price(HEAT_TICKER)


def _curve_symbols(start_date: dt.date,
                    prefix: str = "CL") -> List[str]:
    """NYMEX symbols for the next 12 monthly contracts after start_date.

    Defaults to WTI (`CL`). Pass `prefix="BZ"` for Brent (BZ NYMEX),
    `"RB"` for RBOB Gasoline, `"HO"` for Heating Oil / ULSD,
    `"NG"` for Natural Gas. All five products use the same NYMEX month-code
    suffix convention so the same composition works."""
    out: List[str] = []
    year, month = start_date.year, start_date.month
    for _ in range(12):
        month += 1
        if month > 12:
            month = 1
            year += 1
        out.append(f"{prefix}{_MONTH_CODES[month - 1]}{str(year)[-2:]}.NYM")
    return out


def _fetch_curve_for_prefix(prefix: str) -> Optional[List[Dict[str, float]]]:
    """Real 12-month settlement curve for any NYMEX product prefix."""
    if not YF_AVAILABLE:
        return None
    symbols = _curve_symbols(dt.date.today(), prefix=prefix)
    curve: List[Dict[str, float]] = []
    try:
        df = yf.download(symbols, period="5d", progress=False,
                         auto_adjust=True, threads=True)
    except Exception:
        return None
    try:
        block = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    except Exception:
        block = df
    for i, sym in enumerate(symbols, start=1):
        try:
            series = block[sym] if sym in block.columns else None
            if series is None:
                continue
            cleaned = series.dropna()
            if cleaned.empty:
                continue
            curve.append({"month": i,
                          "price": round(float(cleaned.iloc[-1]), 4)})
        except Exception:
            continue
    return curve or None


def fetch_brent_curve() -> Optional[List[Dict[str, float]]]:
    """Real Brent 12-month settlement curve via NYMEX BZ contract series."""
    return _fetch_curve_for_prefix("BZ")


def fetch_rbob_curve() -> Optional[List[Dict[str, float]]]:
    """Real RBOB Gasoline 12-month settlement curve via NYMEX RB series."""
    return _fetch_curve_for_prefix("RB")


def fetch_heat_curve() -> Optional[List[Dict[str, float]]]:
    """Real Heating Oil / ULSD 12-month settlement curve via NYMEX HO series."""
    return _fetch_curve_for_prefix("HO")


def fetch_natgas_curve() -> Optional[List[Dict[str, float]]]:
    """Real Natural Gas 12-month settlement curve via NYMEX NG series."""
    return _fetch_curve_for_prefix("NG")


def fetch_5y_same_week() -> Optional[List[Dict[str, float]]]:
    """Real WTI closing prices for the same calendar week in each of the
    past 5 years. Used to replace the synthesized 5-year week-range.

    Returns ``[{year: 2025, price: 78.40}, ..., {year: 2021, price: 65.11}]``
    newest-first, or ``None`` if too few years can be matched (e.g. yfinance
    hasn't returned enough history yet)."""
    if not YF_AVAILABLE:
        return None
    try:
        today = dt.date.today()
        # pull 5 years + 2 months buffer so the 5-year-back lookup always lands
        start = today - dt.timedelta(days=365 * 5 + 60)
        hist = yf.Ticker(WTI_TICKER).history(
            start=str(start), end=str(today))["Close"].dropna()
        idx_dates = [d.date() for d in hist.index]
        out: List[Dict[str, float]] = []
        for years_back in range(1, 6):
            target = today - dt.timedelta(days=365 * years_back)
            best_i, best_gap = None, 999
            for i, d in enumerate(idx_dates):
                gap = abs((d - target).days)
                if gap < best_gap:
                    best_i, best_gap = i, gap
                if d > target + dt.timedelta(days=7):
                    break
            if best_i is not None and best_gap <= 7:
                out.append({
                    "year": today.year - years_back,
                    "price": round(float(hist.iloc[best_i]), 2),
                })
        return out if len(out) >= 3 else None
    except Exception:
        return None


def fetch_curve_history(period: str = "1y",
                        prefix: str = "CL") -> Optional[List[Dict]]:
    """REAL daily settlement history for the 12-month futures curve.

    Args:
        period: yfinance period string ("1y", "2y", etc.).
        prefix: NYMEX root code. "CL" = WTI (default), "BZ" = Brent,
                "RB" = RBOB, "HO" = Heating Oil / ULSD, "NG" = Natural Gas.

    Returns a list of curve snapshots — one per trading day — sorted
    oldest-first:

        [{"date": "2025-05-30", "prices": [88.5, 87.2, ..., 79.3]}, ...]

    Each row is a single day's full curve, with ``prices`` aligned in
    M1..M12 order (relative to the NYMEX contracts trading TODAY).
    Used to seed the spread covariance matrix with real variance from
    real settlements — no synthetic data, no noise.

    Returns ``None`` if too few aligned days resolve. Tolerates missing
    front-month contracts (already expired) by skipping them — the
    resulting curve is shorter but still real."""
    if not YF_AVAILABLE:
        return None
    symbols = _curve_symbols(dt.date.today(), prefix=prefix)
    try:
        df = yf.download(symbols, period=period, progress=False,
                         auto_adjust=True, threads=True)
    except Exception:
        return None
    try:
        block = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    except Exception:
        block = df

    # Build a per-symbol date->price map and find the symbols that
    # actually have data. Skip empty symbols (expired / not-yet-listed).
    series_by_sym: Dict[str, Dict] = {}
    active_syms: List[str] = []
    for sym in symbols:
        try:
            s = block[sym] if sym in block.columns else None
            if s is None:
                continue
            cleaned = s.dropna()
            if cleaned.empty:
                continue
            series_by_sym[sym] = {
                d.date(): round(float(v), 2)
                for d, v in cleaned.items()
            }
            active_syms.append(sym)
        except Exception:
            continue
    if len(active_syms) < 6:
        return None

    # Intersect dates so every row has all active contracts populated.
    common_dates = None
    for sym in active_syms:
        these = set(series_by_sym[sym].keys())
        common_dates = these if common_dates is None else common_dates & these
    if not common_dates or len(common_dates) < 30:
        return None

    rows: List[Dict] = []
    for d in sorted(common_dates):
        prices = [series_by_sym[sym][d] for sym in active_syms]
        rows.append({"date": str(d), "prices": prices})
    return rows


def fetch_all_curve_histories(period: str = "1y") -> Dict[str, Optional[List[Dict]]]:
    """One-shot fetch of the daily settlement history for all 5 NYMEX products.

    Returns a dict keyed by product name:
        {"wti": [...], "brent": [...], "rbob": [...], "heat": [...], "natgas": [...]}

    Each value is either a list of {"date": ..., "prices": [...]} rows
    (same format as fetch_curve_history()) or None if Yahoo refused that
    product. Brent/RBOB/HO/NG curves on NYMEX are thinner than WTI, so it's
    normal to get back shorter strips (5-8 contracts vs 12) for them.
    """
    out: Dict[str, Optional[List[Dict]]] = {}
    for key, prefix in (("wti", "CL"), ("brent", "BZ"), ("rbob", "RB"),
                        ("heat", "HO"), ("natgas", "NG")):
        try:
            out[key] = fetch_curve_history(period=period, prefix=prefix)
        except Exception:
            out[key] = None
    return out


def fetch_curve() -> Optional[List[Dict[str, float]]]:
    """Real WTI futures settlement curve for the next 12 months. Returns
    [{month: 1..12, price: float}, ...]; None if too few contracts resolve."""
    if not YF_AVAILABLE:
        return None
    symbols = _curve_symbols(dt.date.today())
    curve: List[Dict[str, float]] = []
    try:
        df = yf.download(symbols, period="5d", progress=False,
                         auto_adjust=True, threads=True)
    except Exception:
        return None

    # multi-ticker download returns a multi-index column frame; single ticker
    # returns a single-level frame. Normalize to a dict of price-by-symbol.
    closes: Dict[str, float] = {}
    try:
        block = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    except Exception:
        block = df
    for i, sym in enumerate(symbols, start=1):
        try:
            series = block[sym] if sym in block.columns else None
            if series is None:
                continue
            cleaned = series.dropna()
            if cleaned.empty:
                continue
            closes[sym] = round(float(cleaned.iloc[-1]), 2)
            curve.append({"month": i, "price": closes[sym]})
        except Exception:
            continue
    return curve if len(curve) >= 6 else None

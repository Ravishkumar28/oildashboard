"""Historical backtest of the dashboard's z-score strategies.

Pulls 5 years of daily closes for WTI, Brent, RBOB, heating oil and the
Dollar Index from Yahoo Finance, replays each strategy day-by-day using
only data that was available at that date (no look-ahead bias), and reports
realistic performance stats — win rate, Sharpe, max drawdown, total P&L.

Run from the backend/ directory:
    python backtest.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf


# --- assumptions -------------------------------------------------------- #
STARTING_EQUITY = 100_000.0
TRADE_SIZE_BBL = 1_000
COST_PER_BBL = 0.10                # bid-ask + slippage assumption, per side
LOOKBACK_DAYS = 90                 # rolling window for z-score
HISTORY_PERIOD = "5y"
OUT_OF_SAMPLE_DAYS = 90


# --- data --------------------------------------------------------------- #
def _close(ticker: str) -> pd.Series:
    """Daily close series indexed by date (tz-stripped)."""
    s = yf.Ticker(ticker).history(period=HISTORY_PERIOD)["Close"].dropna()
    s.index = [d.date() for d in s.index]
    return s


def fetch_history() -> pd.DataFrame:
    """Real daily closes for everything the strategies need."""
    series = {
        "wti":   _close("CL=F"),
        "brent": _close("BZ=F"),
        "rbob":  _close("RB=F"),
        "ho":    _close("HO=F"),
        "dxy":   _close("DX-Y.NYB"),
    }
    df = pd.concat(series, axis=1).dropna()
    # derived series
    df["brent_wti_spread"] = df["brent"] - df["wti"]
    df["gasoline_crack"]   = df["rbob"] * 42.0 - df["wti"]
    df["diesel_crack"]     = df["ho"]   * 42.0 - df["wti"]
    df["crack_321"]        = (2 * df["gasoline_crack"] + df["diesel_crack"]) / 3
    df["wti_dxy_combo"]    = df["wti"] + 0.5 * df["dxy"]
    return df


# --- backtest engine ---------------------------------------------------- #
def backtest_mr(series: pd.Series, *,
                z_in: float, z_out: float = 0.3,
                lookback: int = LOOKBACK_DAYS,
                size: int = TRADE_SIZE_BBL,
                cost_per_bbl: float = COST_PER_BBL,
                stop_sigma: Optional[float] = None,
                max_hold_days: Optional[int] = None) -> Dict:
    """Walk-forward mean-reversion backtest on a 1-D price/spread series.

    Rules:
      - enter SHORT at z >= +z_in, LONG at z <= -z_in
      - close on |z| <= z_out (mean reverted)
      - stop_sigma:    if loss vs entry exceeds stop_sigma × σ_at_entry, exit
      - max_hold_days: if position older than this many bars, force exit

    Each round-trip pays 2x cost_per_bbl in transaction friction. Returns
    summary stats; each trade also records its ``exit_reason``."""
    values = series.values
    dates  = list(series.index)
    n = len(values)
    if n <= lookback:
        return {"n_trades": 0, "trades": [], "equity_curve": []}

    trades: List[Dict] = []
    pos: Optional[Dict] = None
    realized = 0.0
    equity_curve = []

    for i in range(lookback, n):
        window = values[i - lookback:i]
        std = float(window.std(ddof=1))
        if std == 0:
            continue
        z = (values[i] - float(window.mean())) / std
        price = float(values[i])

        if pos is None:
            if z >= z_in:
                pos = {"dir": "SHORT", "entry": price, "entry_i": i,
                       "entry_sigma": std, "entry_date": str(dates[i])}
            elif z <= -z_in:
                pos = {"dir": "LONG",  "entry": price, "entry_i": i,
                       "entry_sigma": std, "entry_date": str(dates[i])}
        else:
            sign = 1 if pos["dir"] == "LONG" else -1
            adverse_move = -(price - pos["entry"]) * sign   # +ve = loss/bbl
            held_days = i - pos["entry_i"]

            exit_reason: Optional[str] = None
            if abs(z) <= z_out:
                exit_reason = "mean_revert"
            elif (stop_sigma is not None
                  and adverse_move >= stop_sigma * pos["entry_sigma"]):
                exit_reason = "stop_loss"
            elif (max_hold_days is not None
                  and held_days >= max_hold_days):
                exit_reason = "max_hold"

            if exit_reason:
                pnl = (price - pos["entry"]) * sign * size
                pnl -= 2 * cost_per_bbl * size      # in + out cost
                realized += pnl
                trades.append({
                    "dir": pos["dir"],
                    "entry": round(pos["entry"], 3),
                    "exit":  round(price, 3),
                    "pnl":   round(pnl, 2),
                    "hold_days": held_days,
                    "entry_date": pos["entry_date"],
                    "exit_date":  str(dates[i]),
                    "exit_reason": exit_reason,
                })
                pos = None

        # mark equity (realized + unrealized MTM)
        unreal = 0.0
        if pos:
            sign = 1 if pos["dir"] == "LONG" else -1
            unreal = (price - pos["entry"]) * sign * size
        equity_curve.append({"date": str(dates[i]),
                             "equity": round(STARTING_EQUITY + realized + unreal, 2)})

    return summarise(trades, equity_curve)


_MONTH_CODES = "FGHJKMNQUVXZ"   # NYMEX month codes Jan..Dec


def _contract_symbol(year: int, month: int) -> str:
    return f"CL{_MONTH_CODES[month - 1]}{str(year)[-2:]}.NYM"


def _contract_for_offset(base_date: dt.date, offset_months: int) -> str:
    y, m = base_date.year, base_date.month + offset_months
    while m > 12:
        m -= 12
        y += 1
    return _contract_symbol(y, m)


def fetch_eia_continuous_futures(years_back: int = 5
                                 ) -> Optional[pd.DataFrame]:
    """EIA continuous WTI futures C1-C4 daily settlements.

    NOTE: EIA discontinued this series in April 2024 — data ends there.
    Still ~3+ years of usable history before that. Returns a DataFrame
    indexed by date with columns C1, C2, C3, C4, or None on failure."""
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        try:
            for line in (Path(__file__).resolve().parent.parent
                         / ".env").read_text().splitlines():
                if line.strip().startswith("EIA_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
        except Exception:
            return None
    if not api_key:
        return None

    today = dt.date.today()
    start = (today - dt.timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    rows_by_date: Dict[dt.date, Dict[int, float]] = {}
    for n in (1, 2, 3, 4):
        try:
            r = httpx.get(
                f"https://api.eia.gov/v2/seriesid/PET.RCLC{n}.D"
                f"?api_key={api_key}&start={start}&length=5000",
                timeout=20).json()
            for row in r.get("response", {}).get("data", []):
                d = dt.datetime.strptime(row["period"], "%Y-%m-%d").date()
                rows_by_date.setdefault(d, {})[n] = float(row["value"])
        except Exception:
            return None
    full = {d: v for d, v in rows_by_date.items() if len(v) == 4}
    if len(full) < 200:
        return None
    dates = sorted(full.keys())
    return pd.DataFrame({
        "C1": [full[d][1] for d in dates],
        "C2": [full[d][2] for d in dates],
        "C3": [full[d][3] for d in dates],
        "C4": [full[d][4] for d in dates],
    }, index=dates)


def fetch_butterfly_history(years: int = 5) -> Optional[pd.Series]:
    """Historical M3-M6-M9 butterfly fly value for each trading day.

    For each date t, the fly is computed from the three CL contracts that
    expire 3, 6, and 9 months after t. Requires downloading every CL monthly
    contract that traded during the window — roughly 80-100 symbols.
    Returns a pd.Series of fly values indexed by date, or None on failure."""
    today = dt.date.today()
    syms: List[str] = []
    for y in range(today.year - years, today.year + 2):
        for m in range(1, 13):
            syms.append(_contract_symbol(y, m))

    print(f"  fetching {len(syms)} CL monthly contracts ...")
    df = yf.download(syms, period=f"{years + 1}y", progress=False,
                     auto_adjust=True, threads=True, group_by="ticker")

    closes: Dict[str, pd.Series] = {}
    for s in syms:
        try:
            col = df[s]["Close"] if (s, "Close") in df.columns else df[s]
            cleaned = col.dropna()
            if not cleaned.empty:
                cleaned.index = [d.date() for d in cleaned.index]
                closes[s] = cleaned
        except Exception:
            continue
    print(f"  {len(closes)} contracts returned data")

    all_dates = sorted({d for s in closes.values() for d in s.index})
    flys: Dict[dt.date, float] = {}
    for d in all_dates:
        m3 = _contract_for_offset(d, 3)
        m6 = _contract_for_offset(d, 6)
        m9 = _contract_for_offset(d, 9)
        if all(s in closes and d in closes[s].index for s in (m3, m6, m9)):
            p3 = float(closes[m3].loc[d])
            p6 = float(closes[m6].loc[d])
            p9 = float(closes[m9].loc[d])
            flys[d] = p3 - 2 * p6 + p9

    if len(flys) < 200:
        print(f"  WARNING only {len(flys)} valid fly observations")
        return None

    sorted_dates = sorted(flys.keys())
    return pd.Series([flys[d] for d in sorted_dates], index=sorted_dates,
                     name="fly_value")


def kalman_pair(y_series: pd.Series, x_series: pd.Series,
                q_alpha: float = 1e-4,
                q_beta: float = 1e-5,
                r_obs: float = 2.0) -> pd.Series:
    """Time-varying linear regression  y_t = alpha_t + beta_t * x_t + noise,
    estimated via a 2-state Kalman filter where alpha and beta drift as a
    random walk.

    Used to track the slowly-changing WTI vs DXY relationship. The output
    residual = y_t - (alpha_t + beta_t * x_t) is what we z-score and trade.

    Tuning:
      q_alpha / q_beta — how much we let alpha/beta drift per day (small =
          stable estimates, large = chase noise). Beta is set 10x more stable
          than alpha because the hedge ratio shifts on regime timescales.
      r_obs           — observation noise; roughly the daily $-scale of WTI
          deviations not explained by the linear DXY model.
    """
    y = y_series.values
    x = x_series.values
    n = len(y)

    state = np.zeros(2)          # [alpha, beta]
    P = np.eye(2) * 1.0          # posterior covariance (start uncertain)
    Q = np.diag([q_alpha, q_beta])
    I2 = np.eye(2)

    residuals = np.zeros(n)
    for t in range(n):
        # ---- predict (random-walk transition: state stays put) -------- #
        P_pred = P + Q

        # ---- observe -------------------------------------------------- #
        H = np.array([1.0, x[t]])               # 1x2 observation row
        y_pred = state[0] + state[1] * x[t]
        innovation = y[t] - y_pred
        residuals[t] = innovation
        S = float(H @ P_pred @ H + r_obs)        # innovation variance
        K = (P_pred @ H) / S                     # Kalman gain (2-vector)

        # ---- update --------------------------------------------------- #
        state = state + K * innovation
        P = (I2 - np.outer(K, H)) @ P_pred

    return pd.Series(residuals, index=y_series.index, name="kalman_residual")


def summarise(trades: List[Dict], equity_curve: List[Dict]) -> Dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "avg_hold_days": 0.0, "max_dd_pct": 0.0,
                "sharpe": 0.0, "final_equity": STARTING_EQUITY,
                "return_pct": 0.0, "trades": [], "equity_curve": equity_curve}

    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / n
    avg_hold = sum(t["hold_days"] for t in trades) / n

    # max drawdown from peak
    peak = STARTING_EQUITY
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt["equity"])
        dd = (pt["equity"] - peak) / peak if peak else 0
        max_dd = min(max_dd, dd)

    # annualised Sharpe from daily equity returns
    eqs = [pt["equity"] for pt in equity_curve]
    rets = []
    for i in range(1, len(eqs)):
        if eqs[i - 1]:
            rets.append(eqs[i] / eqs[i - 1] - 1)
    sharpe = 0.0
    if rets:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            sharpe = (mean_r / sd) * math.sqrt(252)

    final_eq = equity_curve[-1]["equity"] if equity_curve else STARTING_EQUITY
    return {
        "n_trades": n,
        "win_rate_pct": round(100 * len(wins) / n, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "avg_hold_days": round(avg_hold, 1),
        "max_dd_pct": round(100 * max_dd, 2),
        "sharpe": round(sharpe, 2),
        "final_equity": round(final_eq, 2),
        "return_pct": round(100 * (final_eq / STARTING_EQUITY - 1), 2),
        "trades": trades,
        "equity_curve": equity_curve,
    }


# --- strategy catalog --------------------------------------------------- #
STRATEGIES = [
    # (name, series column, z_in entry, z_out exit)
    ("Diesel Crack MR (z>=1.0)",          "diesel_crack",     1.00, 0.30),
    ("3-2-1 USGC Crack MR (z>=1.25)",     "crack_321",        1.25, 0.30),
    ("Brent-WTI Extreme (z>=2.0)",        "brent_wti_spread", 2.00, 0.50),
    ("WTI-DXY Pair (fixed beta, z>=2.0)", "wti_dxy_combo",    2.00, 0.50),
]


def _print_row(fmt: str, name: str, r: Dict) -> None:
    print(fmt.format(
        name, r["n_trades"],
        f"{r['win_rate_pct']:.1f}",
        f"${r['total_pnl']:,.0f}",
        f"${r['final_equity']:,.0f}",
        f"{r['return_pct']:+.2f}",
        f"{r['max_dd_pct']:+.2f}",
        f"{r['sharpe']:+.2f}",
    ))


def run_period(df: pd.DataFrame, label: str,
               stop_sigma: Optional[float] = None,
               max_hold_days: Optional[int] = None) -> List[Dict]:
    tag = (f" + stops (stop={stop_sigma}σ, max_hold={max_hold_days}d)"
           if stop_sigma else "")
    print(f"\n=== {label}{tag} "
          f"({df.index[0]} → {df.index[-1]}, {len(df)} bars) ===")
    fmt = ("{:38s} {:>8} {:>10} {:>14} {:>12} {:>10} {:>9} {:>8}")
    print(fmt.format("Strategy", "Trades", "Win%", "Total P&L",
                     "Final Eq", "Return%", "Max DD%", "Sharpe"))
    print("-" * 120)
    rows = []
    for name, col, z_in, z_out in STRATEGIES:
        result = backtest_mr(df[col], z_in=z_in, z_out=z_out,
                             stop_sigma=stop_sigma,
                             max_hold_days=max_hold_days)
        _print_row(fmt, name, result)
        rows.append({"name": name, "label": label + tag,
                     **{k: v for k, v in result.items()
                        if k not in ("trades", "equity_curve")}})

    kalman_resid = kalman_pair(df["wti"], df["dxy"])
    result = backtest_mr(kalman_resid, z_in=2.0, z_out=0.5,
                         stop_sigma=stop_sigma,
                         max_hold_days=max_hold_days)
    _print_row(fmt, "WTI-DXY Pair (Kalman beta, z>=2.0)", result)
    rows.append({"name": "WTI-DXY Pair (Kalman beta, z>=2.0)",
                 "label": label + tag,
                 **{k: v for k, v in result.items()
                    if k not in ("trades", "equity_curve")}})
    return rows


def grid_search_kalman(df: pd.DataFrame,
                       q_alpha_grid=(1e-5, 1e-4, 1e-3),
                       q_beta_grid=(1e-6, 1e-5, 1e-4),
                       r_obs_grid=(1.0, 2.0, 4.0, 8.0)) -> List[Dict]:
    """Sweep Kalman noise parameters; reports stability of the result.

    If Sharpe stays positive across most of the grid, the original result
    was robust. If it only worked at one specific (Q, R) the parameters
    were probably overfit."""
    print("\n=== KALMAN PARAMETER GRID SEARCH (WTI vs DXY) ===")
    rows = []
    for qa in q_alpha_grid:
        for qb in q_beta_grid:
            for r in r_obs_grid:
                resid = kalman_pair(df["wti"], df["dxy"],
                                    q_alpha=qa, q_beta=qb, r_obs=r)
                bt = backtest_mr(resid, z_in=2.0, z_out=0.5)
                rows.append({
                    "q_alpha": qa, "q_beta": qb, "r_obs": r,
                    "n_trades": bt["n_trades"],
                    "win_rate": bt["win_rate_pct"],
                    "total_pnl": bt["total_pnl"],
                    "sharpe": bt["sharpe"],
                    "max_dd_pct": bt["max_dd_pct"],
                })
    sharpes = [r["sharpe"] for r in rows]
    pnls = [r["total_pnl"] for r in rows]
    pos = sum(1 for s in sharpes if s > 0.5)
    print(f"Sweep over {len(rows)} (q_alpha, q_beta, r_obs) combinations")
    print(f"  Sharpe   min={min(sharpes):+.2f}  "
          f"median={sorted(sharpes)[len(sharpes)//2]:+.2f}  "
          f"max={max(sharpes):+.2f}")
    print(f"  Total $   min=${min(pnls):>+10,.0f}   "
          f"max=${max(pnls):>+12,.0f}")
    print(f"  Combos with Sharpe > 0.5: {pos} / {len(rows)} "
          f"({100*pos/len(rows):.0f}%)")
    print(f"  Top 5 (q_alpha, q_beta, r_obs) by Sharpe:")
    for r in sorted(rows, key=lambda x: -x["sharpe"])[:5]:
        print(f"    qa={r['q_alpha']:.0e}  qb={r['q_beta']:.0e}  "
              f"r={r['r_obs']:.1f}  ->  Sharpe {r['sharpe']:+.2f}  "
              f"trades {r['n_trades']:>3}  P&L ${r['total_pnl']:>+11,.0f}")
    return rows


def walk_forward_kalman(df: pd.DataFrame, chunk_days: int = 180) -> List[Dict]:
    """Run the Kalman pair strategy on non-overlapping forward windows.

    If the strategy works in 2021 but fails in 2024, that tells us the
    regime changed. Stable performance across chunks = real edge."""
    print(f"\n=== WALK-FORWARD ANALYSIS (Kalman WTI-DXY, {chunk_days}d chunks) ===")
    fmt = "{:25s}  {:>6}  {:>7}  {:>11}  {:>8}  {:>8}"
    print(fmt.format("Window", "Trades", "Win%", "Total P&L", "Sharpe", "DD%"))
    print("-" * 78)
    chunks = []
    step = chunk_days
    for start in range(0, len(df) - chunk_days, step):
        chunk = df.iloc[start:start + chunk_days + LOOKBACK_DAYS]
        if len(chunk) < LOOKBACK_DAYS + 30:
            continue
        resid = kalman_pair(chunk["wti"], chunk["dxy"])
        bt = backtest_mr(resid, z_in=2.0, z_out=0.5)
        label = f"{chunk.index[LOOKBACK_DAYS]} → {chunk.index[-1]}"
        print(fmt.format(label, bt["n_trades"],
                         f"{bt['win_rate_pct']:.0f}",
                         f"${bt['total_pnl']:,.0f}",
                         f"{bt['sharpe']:+.2f}",
                         f"{bt['max_dd_pct']:+.2f}"))
        chunks.append({"window": label, **{k: v for k, v in bt.items()
                       if k not in ("trades", "equity_curve")}})
    sharpes = [c["sharpe"] for c in chunks]
    pos = sum(1 for s in sharpes if s > 0)
    print(f"  Windows positive Sharpe: {pos} / {len(chunks)}")
    return chunks


def run_extra_kalman_pairs(df: pd.DataFrame) -> List[Dict]:
    """Apply Kalman dynamic-beta hedge to two more pairs:
       Brent vs WTI (replacing the fixed 1-to-1 spread) and
       Heating-oil vs WTI (replacing the fixed 42x diesel crack)."""
    print("\n=== EXTENDING KALMAN TO OTHER PAIRS ===")
    fmt = "{:42s} {:>8} {:>10} {:>14} {:>10} {:>9} {:>8}"
    print(fmt.format("Strategy", "Trades", "Win%", "Total P&L",
                     "Return%", "DD%", "Sharpe"))
    print("-" * 105)
    rows = []
    for name, y_col, x_col, z_in in [
        ("Brent-WTI (Kalman beta, z>=2.0)",     "brent", "wti", 2.0),
        ("HO-WTI Diesel Crack (Kalman, z>=1.5)", "ho",    "wti", 1.5),
    ]:
        resid = kalman_pair(df[y_col], df[x_col])
        bt = backtest_mr(resid, z_in=z_in, z_out=0.3)
        print(fmt.format(name, bt["n_trades"],
                         f"{bt['win_rate_pct']:.1f}",
                         f"${bt['total_pnl']:,.0f}",
                         f"{bt['return_pct']:+.2f}",
                         f"{bt['max_dd_pct']:+.2f}",
                         f"{bt['sharpe']:+.2f}"))
        rows.append({"name": name, **{k: v for k, v in bt.items()
                     if k not in ("trades", "equity_curve")}})
    return rows


def main() -> None:
    print("Pulling 5y of daily closes from Yahoo for WTI/Brent/RBOB/HO/DXY ...")
    df = fetch_history()
    print(f"Got {len(df)} aligned trading days ({df.index[0]} → {df.index[-1]})")
    print(f"Assumptions: ${COST_PER_BBL:.2f}/bbl cost per side, "
          f"{TRADE_SIZE_BBL:,} bbl per trade, ${STARTING_EQUITY:,.0f} starting equity")

    all_rows = []
    # Baseline: no risk controls
    all_rows += run_period(df, "FULL PERIOD")
    # With risk controls: 3σ stop-loss + 60-day max hold
    all_rows += run_period(df, "FULL PERIOD",
                           stop_sigma=3.0, max_hold_days=60)
    if len(df) > OUT_OF_SAMPLE_DAYS + LOOKBACK_DAYS:
        oos = df.iloc[-(OUT_OF_SAMPLE_DAYS + LOOKBACK_DAYS):]
        all_rows += run_period(oos, f"OUT-OF-SAMPLE last {OUT_OF_SAMPLE_DAYS}d")
        all_rows += run_period(oos, f"OUT-OF-SAMPLE last {OUT_OF_SAMPLE_DAYS}d",
                               stop_sigma=3.0, max_hold_days=60)

    # --- robustness / extension analyses ------------------------------ #
    grid = grid_search_kalman(df)
    walk = walk_forward_kalman(df, chunk_days=180)
    extra = run_extra_kalman_pairs(df)

    # --- butterfly backtest -------------------------------------------- #
    # The live strategy uses M3-M6-M9; that needs per-day historical prices
    # for ~80 expired CL contracts which yfinance doesn't retain. Substitute
    # is EIA's continuous C1-C4 series, which lets us backtest the SAME
    # curvature concept on a 1-2-3 fly. EIA discontinued the series after
    # April 2024, so this backtest covers ~3 years ending then — different
    # data window, same mathematical setup.
    print("\n=== CURVE BUTTERFLY BACKTEST (EIA C1-C2-C3 fly proxy) ===")
    butterfly_rows: List[Dict] = []
    ef = fetch_eia_continuous_futures(years_back=5)
    if ef is not None and len(ef) > LOOKBACK_DAYS + 30:
        fly_123 = ef["C1"] - 2 * ef["C2"] + ef["C3"]
        fly_234 = ef["C2"] - 2 * ef["C3"] + ef["C4"]
        print(f"  fly history: {len(fly_123)} days  "
              f"{ef.index[0]} → {ef.index[-1]}")
        print(f"  1-2-3 fly  range  ${fly_123.min():+.3f} → ${fly_123.max():+.3f}  "
              f"mean ${fly_123.mean():+.3f}")
        configs = [
            ("1-2-3 fly  z>=1.5σ  (10k bbl)", fly_123, 1.5, 10_000),
            ("1-2-3 fly  z>=2.0σ  (10k bbl)", fly_123, 2.0, 10_000),
            ("1-2-3 fly  z>=1.0σ  (10k bbl)", fly_123, 1.0, 10_000),
            ("2-3-4 fly  z>=1.5σ  (10k bbl)", fly_234, 1.5, 10_000),
        ]
        for stop_label, stop_sigma in [("baseline", None),
                                        ("stop=3σ + hold≤60d", 3.0)]:
            print(f"\n  [{stop_label}]")
            for name, series, z_in, size in configs:
                bt = backtest_mr(series, z_in=z_in, z_out=0.3,
                                 lookback=LOOKBACK_DAYS, size=size,
                                 cost_per_bbl=0.05,
                                 stop_sigma=stop_sigma,
                                 max_hold_days=60 if stop_sigma else None)
                print(f"  {name:34s}  trades={bt['n_trades']:>3}  "
                      f"win%={bt['win_rate_pct']:>5.1f}  "
                      f"P&L=${bt['total_pnl']:>+10,.0f}  "
                      f"DD={bt['max_dd_pct']:>+6.2f}%  "
                      f"Sharpe={bt['sharpe']:+.2f}")
                butterfly_rows.append({
                    "name": f"{name} [{stop_label}]",
                    **{k: v for k, v in bt.items()
                       if k not in ("trades", "equity_curve")}})
    else:
        print("  EIA C1-C4 series unavailable — skipping butterfly backtest")

    # persist a JSON summary alongside the script for later inspection
    out_path = Path(__file__).parent / "backtest_results.json"
    out_path.write_text(json.dumps({
        "strategies": all_rows,
        "kalman_grid_search": grid,
        "kalman_walk_forward": walk,
        "extra_kalman_pairs": extra,
        "butterfly": butterfly_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()

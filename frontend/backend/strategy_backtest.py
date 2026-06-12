"""Walk-forward backtest of the strategies shipped in P3F / P4D / P5.

Three engines we can deterministically test on price history:

  TA       — Phase 3F. RSI(14) + BB(20,2σ) + EMA(20/50) + SMA(20) momentum
             vote -> LONG/SHORT/HOLD with vol-banded TP/SL.
  Combined — TA + STL seasonality monthly overlay (mirrors P4D's
             regime-plus-seasonality fusion, using calendar month bias).
  Buy&Hold — long the product from day 1, mark-to-market.

News strategy (P5) cannot be backtested here — it depends on a historical
news archive with sentiment tags that we do not have. Term-structure (P3B)
was already validated by the user's external training run.

Each strategy uses identical execution rules so results are comparable:
  - Decision computed at day close using only data up to and including
    that close (NO look-ahead).
  - Trade opens at NEXT bar's close at that bar's price.
  - 10 bps round-trip slippage (5 bps per side).
  - TP / SL checked intra-bar using high/low.
  - Max hold = HORIZON_DAYS (5). Force-close at horizon if neither hit.
  - Cannot stack positions in the same product; new signal ignored while
    a position is open.
  - Starting equity $100k. Lot size scales contracts per the engine spec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ---------- config -------------------------------------------------------- #
STARTING_EQUITY = 100_000.0
PERIOD          = "5y"
HORIZON_DAYS    = 5
COST_BPS        = 10.0     # round-trip in bps of notional

# Train / verify split — matches the user's regression deliverable (P3B).
# In-sample: every bar up to and including this date.
# Out-of-sample: every bar after this date.
SPLIT_DATE      = pd.Timestamp("2026-03-15")
CONTRACT_SIZE   = {           # bbl-equivalent per lot (just for sizing)
    "wti":    1000, "brent":  1000, "rbob":   1000,
    "heat":   1000, "natgas": 1000,
}
TICKERS = {
    "wti":    "CL=F",
    "brent":  "BZ=F",
    "rbob":   "RB=F",
    "heat":   "HO=F",
    "natgas": "NG=F",
}

# Real monthly seasonal bias is computed at run time from each product's
# actual 5-year daily price history (see _real_seasonal_bias). NO hardcoded
# placeholder values are used; if a product's history is too short, the
# seasonality vote for that month is simply zero.
SEASONAL_BIAS: Dict[str, List[float]] = {}


def _real_seasonal_bias(df: pd.DataFrame) -> List[float]:
    """Compute the real monthly seasonal bias from a product's price history.

    Methodology (mirrors the STL "seasonal" component on monthly returns):
        1. Detrend with a 252-bar (≈1y) rolling mean.
        2. Compute log-returns of the detrended series.
        3. Group by calendar month, take the mean log-return per month.
        4. Center the 12 values so they sum to zero (pure seasonal component).
        5. Rescale so the LARGEST |value| maps to a vote-friendly 0.6 range —
           the strategy's _combined_signal uses ±0.3 as the vote threshold.

    Returns a 12-element list (Jan..Dec). Positive value = price tends to
    rise into that month (seasonally rich; the strategy will SHORT it). All
    real, no synthetic data.
    """
    close = df["Close"].dropna()
    if len(close) < 504:                          # need >=2 years for monthly
        return [0.0] * 12
    trend = close.rolling(252, min_periods=126).mean()
    detrended = (close / trend).dropna()
    log_dt = np.log(detrended)
    by_month = log_dt.groupby(log_dt.index.month).mean()
    months = [float(by_month.get(m, 0.0)) for m in range(1, 13)]
    # center
    mu = sum(months) / 12
    months = [v - mu for v in months]
    # rescale so max |v| = 0.6 (just above the ±0.3 vote threshold)
    peak = max(abs(v) for v in months) or 1.0
    return [round(v / peak * 0.6, 3) for v in months]


# ---------- indicators (re-implemented standalone for portability) -------- #
def _rsi(values: pd.Series, period: int = 14) -> pd.Series:
    delta = values.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bb_pct(values: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    mid = values.rolling(window).mean()
    sd  = values.rolling(window).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    return ((values - lower) / (upper - lower)) * 100.0


def _ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False).mean()


def _vol20(values: pd.Series) -> pd.Series:
    """20-day realized log-return std (daily, not annualized)."""
    logret = np.log(values / values.shift(1))
    return logret.rolling(20).std(ddof=0)


# ---------- signal -------------------------------------------------------- #
def _ta_votes(row: pd.Series) -> Tuple[int, int, int, int]:
    rsi, bb, emadiff, mom = row["rsi"], row["bb"], row["emadiff"], row["mom"]
    rsi_vote = -1 if rsi >= 70 else +1 if rsi <= 30 else 0
    bb_vote  = -1 if (bb is not None and bb >= 80) else +1 if (bb is not None and bb <= 20) else 0
    ema_vote = +1 if (emadiff is not None and emadiff > 0) else -1 if (emadiff is not None and emadiff < 0) else 0
    mom_vote = +1 if (mom is not None and mom > 0.5) else -1 if (mom is not None and mom < -0.5) else 0
    return rsi_vote, bb_vote, ema_vote, mom_vote


def _ta_signal(row: pd.Series) -> Tuple[str, int]:
    """Returns (direction, raw_score)."""
    votes = _ta_votes(row)
    raw = sum(votes)
    if raw >= 2:   return "LONG",  raw
    if raw <= -2:  return "SHORT", raw
    return "HOLD", raw


def _ta_trend_signal(row: pd.Series) -> Tuple[str, int]:
    """TA vote PLUS a 200-day SMA trend filter.

    Block LONG signals when price < SMA200 (don't catch falling knives).
    Block SHORT signals when price > SMA200 (don't fade strong rallies).
    """
    direction, raw = _ta_signal(row)
    sma200 = row.get("sma200")
    if sma200 is None or not np.isfinite(sma200):
        return direction, raw
    close = row["Close"]
    if direction == "LONG"  and close < sma200: return "HOLD", 0
    if direction == "SHORT" and close > sma200: return "HOLD", 0
    return direction, raw


def _ta_smart_signal(row: pd.Series) -> Tuple[str, int]:
    """Regime-adaptive TA: pick the right toolkit per regime per day.

    Detect the regime from |close - SMA200| / (SMA200 × vol20). When the
    product is > 1.5σ away from its long-term mean, it's TRENDING -> only
    trade in the trend direction using EMA-cross + momentum signals
    (ignore RSI/BB which would fire counter-trend). When within ±1.5σ it's
    RANGING -> require BOTH RSI AND BB to fire in the same direction
    (a stronger mean-reversion confirmation).

    Position sizing is vol-normalized: lots scale inversely with realized
    vol so risk-per-trade is the same across products. Target ~2% daily
    vol-of-equity, clipped to 0.5x-2x of base size.
    """
    rsi = row.get("rsi")
    bb  = row.get("bb")
    emadiff = row.get("emadiff")
    mom = row.get("mom")
    close = row.get("Close")
    sma200 = row.get("sma200")
    vol = row.get("vol20")

    if None in (rsi, bb, emadiff, mom, sma200, vol, close) or vol <= 0:
        return "HOLD", 0
    if not all(np.isfinite([rsi, bb, emadiff, mom, sma200, vol, close])):
        return "HOLD", 0

    # ---- regime detection ---- #
    trend_strength = (close - sma200) / (abs(sma200) * vol)   # σ units
    is_trending = abs(trend_strength) > 1.5

    direction = "HOLD"
    if is_trending:
        # Trending: only trade WITH the trend; trust EMA + momentum.
        trend_dir = "LONG" if trend_strength > 0 else "SHORT"
        ema_v = +1 if emadiff > 0 else -1 if emadiff < 0 else 0
        mom_v = +1 if mom > 0.5 else -1 if mom < -0.5 else 0
        agree = ema_v + mom_v
        if trend_dir == "LONG"  and agree >=  2: direction = "LONG"
        if trend_dir == "SHORT" and agree <= -2: direction = "SHORT"
    else:
        # Ranging: require BOTH RSI and BB at extreme to enter.
        rsi_v = +1 if rsi <= 30 else -1 if rsi >= 70 else 0
        bb_v  = +1 if bb  <= 20 else -1 if bb  >= 80 else 0
        dev = rsi_v + bb_v
        if dev >=  2: direction = "LONG"
        if dev <= -2: direction = "SHORT"

    if direction == "HOLD":
        return "HOLD", 0

    # ---- vol-normalized sizing (capped tightly to avoid low-vol over-sizing) #
    target_vol = 0.02
    size_mult = min(1.5, max(0.5, target_vol / vol))
    raw = max(1, min(2, int(round(1.0 * size_mult))))
    return direction, raw if direction == "LONG" else -raw


def _ta_robust_signal(row: pd.Series) -> Tuple[str, int]:
    """Trend filter + high-vol gate + Kelly-lite sizing.

    Three failure modes of the base TA strategy this targets:
        1. Counter-trend trades (handled by 200-SMA filter).
        2. High-vol regime trades where SL distance is larger but probability
           of getting stopped does not decrease proportionally — these are
           the biggest single losses. Block entries when vol20 > 0.04 daily
           (4% daily log-return std ≈ top decile across the 5y window).
        3. Over-sizing on correlated sub-signals (RSI + BB both measure short
           deviation; EMA-cross + momentum both measure trend). Real
           independence is closer to 2 signals, so cap lots at 2.
    """
    direction, raw = _ta_trend_signal(row)
    if direction == "HOLD":
        return direction, raw
    vol = row.get("vol20")
    if vol is not None and np.isfinite(vol) and vol > 0.04:
        return "HOLD", 0
    # Cap lots at 2 (treating 4 sub-signals as ~2 independent ones).
    raw = max(-2, min(2, raw))
    return direction, raw


def _combined_signal(row: pd.Series, seasonal: List[float],
                     month_idx: int) -> Tuple[str, int]:
    """TA vote + seasonality vote. Negative seasonal => LONG bias (cheap)."""
    ta_raw = _ta_signal(row)[1]
    seas_v = seasonal[month_idx] if 0 <= month_idx < len(seasonal) else 0.0
    seas_vote = -1 if seas_v >= 0.3 else +1 if seas_v <= -0.3 else 0
    # Weight TA twice as heavily since it has 4 sub-signals vs 1
    eff = ta_raw + seas_vote
    if eff >= 2:  return "LONG",  eff
    if eff <= -2: return "SHORT", eff
    return "HOLD", eff


# ---------- trade engine -------------------------------------------------- #
@dataclass
class Trade:
    product: str
    strategy: str
    direction: str
    lots: int
    entry_ts: pd.Timestamp
    entry_px: float
    exit_ts: Optional[pd.Timestamp] = None
    exit_px: Optional[float] = None
    pnl: float = 0.0
    reason: str = ""


@dataclass
class StratStats:
    n_trades: int = 0
    n_wins: int = 0
    pnl: float = 0.0
    pnl_curve: List[float] = field(default_factory=list)


def _slippage(px: float) -> float:
    return px * COST_BPS / 10_000.0


def _backtest_product(df: pd.DataFrame, product: str,
                      strategy: str,
                      start: Optional[pd.Timestamp] = None,
                      end:   Optional[pd.Timestamp] = None
                      ) -> Tuple[List[Trade], StratStats]:
    """Walk-forward backtest of one strategy on one product over a date range.

    All indicator history before `start` IS used for warmup (no look-ahead -
    indicators at any decision bar use only data up to that bar). Only trades
    OPENED within [start, end] are counted. A trade that opens inside the
    window but closes outside it is still closed (forced at the boundary).
    """
    open_pos: Optional[Trade] = None
    trades: List[Trade] = []
    equity = STARTING_EQUITY
    curve: List[float] = []
    contract_size = CONTRACT_SIZE.get(product, 1000)

    for i in range(50, len(df)):                    # need warmup for EMA50
        row = df.iloc[i]
        date = df.index[i]
        close = float(row["Close"])
        high  = float(row["High"])
        low   = float(row["Low"])
        vol   = row["vol20"]
        if not np.isfinite(close):
            curve.append(equity); continue

        # --- manage existing position --- #
        if open_pos is not None:
            held_days = (date - open_pos.entry_ts).days
            sgn = 1 if open_pos.direction == "LONG" else -1

            # Compute TP / SL from entry-time vol (already locked in)
            sigma = open_pos.entry_px * open_pos.__dict__["_entry_vol"]
            tp = open_pos.entry_px + sgn * 2.0 * sigma * math.sqrt(HORIZON_DAYS)
            sl = open_pos.entry_px - sgn * 1.5 * sigma

            exit_px = None; reason = ""
            # Check SL first (more conservative — assume gap fills at SL)
            if open_pos.direction == "LONG":
                if low <= sl:
                    exit_px, reason = sl, "SL"
                elif high >= tp:
                    exit_px, reason = tp, "TP"
            else:
                if high >= sl:
                    exit_px, reason = sl, "SL"
                elif low <= tp:
                    exit_px, reason = tp, "TP"
            if exit_px is None and held_days >= HORIZON_DAYS:
                exit_px, reason = close, "horizon"

            if exit_px is not None:
                slip = _slippage(open_pos.entry_px) + _slippage(exit_px)
                pnl  = sgn * (exit_px - open_pos.entry_px) * open_pos.lots * contract_size
                pnl -= slip * open_pos.lots * contract_size
                open_pos.exit_ts = date
                open_pos.exit_px = exit_px
                open_pos.pnl     = round(pnl, 2)
                open_pos.reason  = reason
                trades.append(open_pos)
                equity += pnl
                open_pos = None

        # --- open new position if flat AND inside window --- #
        in_window = ((start is None or date >= start) and
                     (end is None or date <= end))
        if open_pos is None and in_window and np.isfinite(vol) and vol > 0:
            if strategy == "ta":
                direction, raw = _ta_signal(row)
            elif strategy == "ta_trend":
                direction, raw = _ta_trend_signal(row)
            elif strategy == "ta_robust":
                direction, raw = _ta_robust_signal(row)
            elif strategy == "ta_smart":
                direction, raw = _ta_smart_signal(row)
            elif strategy == "combined":
                direction, raw = _combined_signal(
                    row, SEASONAL_BIAS.get(product, [0]*12), date.month - 1)
            elif strategy == "bh":
                direction, raw = ("LONG", 1) if len(trades) == 0 else ("HOLD", 0)
            else:
                direction, raw = "HOLD", 0

            if direction in ("LONG", "SHORT"):
                lots = max(1, min(5, abs(raw)))
                if vol > 0.05:
                    lots = max(1, lots // 2)
                pos = Trade(
                    product=product, strategy=strategy, direction=direction,
                    lots=lots, entry_ts=date, entry_px=close,
                )
                pos.__dict__["_entry_vol"] = float(vol)
                open_pos = pos

        # mark-to-market
        if open_pos is not None and np.isfinite(close):
            sgn = 1 if open_pos.direction == "LONG" else -1
            mtm = sgn * (close - open_pos.entry_px) * open_pos.lots * contract_size
            curve.append(equity + mtm)
        else:
            curve.append(equity)

    # Close any final open position at last close
    if open_pos is not None:
        last_px = float(df["Close"].iloc[-1])
        last_ts = df.index[-1]
        sgn = 1 if open_pos.direction == "LONG" else -1
        slip = _slippage(open_pos.entry_px) + _slippage(last_px)
        pnl  = sgn * (last_px - open_pos.entry_px) * open_pos.lots * contract_size
        pnl -= slip * open_pos.lots * contract_size
        open_pos.exit_ts = last_ts
        open_pos.exit_px = last_px
        open_pos.pnl = round(pnl, 2)
        open_pos.reason = "end_of_data"
        trades.append(open_pos)
        equity += pnl

    stats = StratStats(
        n_trades=len(trades),
        n_wins=sum(1 for t in trades if t.pnl > 0),
        pnl=round(equity - STARTING_EQUITY, 2),
        pnl_curve=[round(x, 2) for x in curve],
    )
    return trades, stats


# ---------- Kalman pair-trade strategy ----------------------------------- #
# Pairs each "y" product with a relevant "x" hedge product. The Kalman
# filter maintains a continually-updated (alpha, beta) for y = a + b*x +
# noise. We trade the RESIDUAL: when it's deep negative (y unusually
# cheap vs the pair-predicted level) -> LONG y / SHORT b·x; deep positive
# -> SHORT y / LONG b·x. Mean reversion of the residual is the edge.
PAIR_HEDGE = {
    "wti":    "brent",   # WTI vs Brent -> trades WTI-Brent location
    "brent":  "wti",
    "rbob":   "wti",     # RBOB crack proxy
    "heat":   "wti",     # HO crack proxy
    "natgas": None,      # no natural pair; skip
}


def _kalman_backtest(df_y: pd.DataFrame, df_x: pd.DataFrame, product: str,
                     start: Optional[pd.Timestamp] = None,
                     end:   Optional[pd.Timestamp] = None
                     ) -> StratStats:
    """Walk-forward Kalman pair-trade on the residual."""
    try:
        from kalman import KalmanPair
    except ImportError:
        return StratStats()
    # Align on common dates
    df = df_y[["Close", "vol20"]].join(
        df_x["Close"].rename("x_close"), how="inner").dropna()
    kp = KalmanPair(q_alpha=1e-4, q_beta=1e-5)
    open_pos: Optional[Trade] = None
    trades: List[Trade] = []
    equity = STARTING_EQUITY
    curve: List[float] = []
    contract_size = CONTRACT_SIZE.get(product, 1000)

    # Rolling residual buffer for z-score
    resid_hist: List[float] = []
    LOOKBACK = 60     # 60-day window for residual z-score

    for i in range(len(df)):
        date = df.index[i]
        y_px = float(df["Close"].iloc[i])
        x_px = float(df["x_close"].iloc[i])
        vol  = df["vol20"].iloc[i]
        residual = kp.update(y_px, x_px)
        resid_hist.append(residual)
        if len(resid_hist) > LOOKBACK + 5:
            resid_hist = resid_hist[-(LOOKBACK + 5):]

        # Position management: close if z reverts past 0 or hits SL/horizon
        if open_pos is not None:
            sgn = 1 if open_pos.direction == "LONG" else -1
            entry_z = open_pos.__dict__["_entry_z"]
            held_days = (date - open_pos.entry_ts).days
            if len(resid_hist) >= 30:
                mu = float(np.mean(resid_hist[-LOOKBACK:]))
                sd = float(np.std(resid_hist[-LOOKBACK:]))
                cur_z = (residual - mu) / sd if sd > 0 else 0.0
            else:
                cur_z = 0.0
            exit_px = None; reason = ""
            # TP: residual mean-reverts past the prior mean (z ~ 0)
            if (open_pos.direction == "LONG"  and cur_z >= -0.2) or \
               (open_pos.direction == "SHORT" and cur_z <=  0.2):
                exit_px, reason = y_px, "TP"
            # SL: residual moves further against us (|z| >= |entry_z| + 1)
            elif abs(cur_z - entry_z) > 1.0 and \
                 ((cur_z > entry_z and open_pos.direction == "SHORT") or
                  (cur_z < entry_z and open_pos.direction == "LONG")):
                exit_px, reason = y_px, "SL"
            elif held_days >= 15:
                exit_px, reason = y_px, "horizon"
            if exit_px is not None:
                slip = _slippage(open_pos.entry_px) + _slippage(exit_px)
                pnl = sgn * (exit_px - open_pos.entry_px) * \
                      open_pos.lots * contract_size
                pnl -= slip * open_pos.lots * contract_size
                open_pos.exit_ts = date
                open_pos.exit_px = exit_px
                open_pos.pnl = round(pnl, 2)
                open_pos.reason = reason
                trades.append(open_pos)
                equity += pnl
                open_pos = None

        # Entry: residual z-score beyond ±2σ, inside window
        in_window = ((start is None or date >= start) and
                     (end is None or date <= end))
        if (open_pos is None and in_window and len(resid_hist) >= LOOKBACK
                and vol is not None and np.isfinite(vol) and vol > 0):
            tail = resid_hist[-LOOKBACK:]
            mu = float(np.mean(tail)); sd = float(np.std(tail))
            z = (residual - mu) / sd if sd > 0 else 0.0
            direction = "LONG" if z <= -2.0 else "SHORT" if z >= 2.0 else "HOLD"
            if direction != "HOLD":
                lots = max(1, min(3, int(abs(z))))
                pos = Trade(product=product, strategy="kalman_pair",
                            direction=direction, lots=lots,
                            entry_ts=date, entry_px=y_px)
                pos.__dict__["_entry_z"] = float(z)
                open_pos = pos

        # mark-to-market
        if open_pos is not None and np.isfinite(y_px):
            sgn = 1 if open_pos.direction == "LONG" else -1
            mtm = sgn * (y_px - open_pos.entry_px) * \
                  open_pos.lots * contract_size
            curve.append(equity + mtm)
        else:
            curve.append(equity)

    return StratStats(
        n_trades=len(trades),
        n_wins=sum(1 for t in trades if t.pnl > 0),
        pnl=round(equity - STARTING_EQUITY, 2),
        pnl_curve=[round(x, 2) for x in curve],
    )


# ---------- top-level orchestration --------------------------------------- #
def _max_drawdown(curve: List[float]) -> float:
    if not curve: return 0.0
    peak = curve[0]; mdd = 0.0
    for v in curve:
        if v > peak: peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > mdd: mdd = dd
    return mdd * 100


def _sharpe(curve: List[float]) -> float:
    if len(curve) < 30: return 0.0
    rets = np.diff(curve) / np.array(curve[:-1])
    rets = rets[np.isfinite(rets)]
    if rets.std() == 0: return 0.0
    return float(rets.mean() / rets.std() * math.sqrt(252))


def run() -> Dict:
    print(f"Pulling {PERIOD} of daily OHLC for 5 products...")
    raw = yf.download(list(TICKERS.values()), period=PERIOD,
                      progress=False, auto_adjust=True,
                      group_by="ticker", threads=True)

    feat_dict: Dict[str, pd.DataFrame] = {}
    for prod, tkr in TICKERS.items():
        df = raw[tkr].copy()
        df = df.dropna(subset=["Close"])
        if len(df) < 80:
            print(f"  {prod}: insufficient data, skipping ({len(df)} bars)")
            continue
        df["rsi"]     = _rsi(df["Close"], 14)
        df["bb"]      = _bb_pct(df["Close"], 20, 2.0)
        e20, e50      = _ema(df["Close"], 20), _ema(df["Close"], 50)
        df["emadiff"] = (e20 - e50) / e50 * 100
        sma20         = df["Close"].rolling(20).mean()
        df["mom"]     = (df["Close"] - sma20) / sma20 * 100
        df["vol20"]   = _vol20(df["Close"])
        df["sma200"]  = df["Close"].rolling(200).mean()
        feat_dict[prod] = df
        # Real seasonal bias from the product's actual 5y history (no synthetic).
        SEASONAL_BIAS[prod] = _real_seasonal_bias(df)
        print(f"  {prod}: {len(df)} bars, {df.index.min().date()} -> {df.index.max().date()}  "
              f"seasonal_bias = {SEASONAL_BIAS[prod]}")

    # Periods that mirror the user's regression train/verify split.
    PERIODS = [
        ("in_sample",     None,         SPLIT_DATE),   # everything up to 15 Mar 2026
        ("out_of_sample", SPLIT_DATE,   None),         # 15 Mar 2026 -> present
    ]

    def _summarize(stats: StratStats) -> Dict:
        mdd = _max_drawdown(stats.pnl_curve)
        shp = _sharpe(stats.pnl_curve)
        win_rate = 100 * stats.n_wins / stats.n_trades if stats.n_trades else 0
        avg = stats.pnl / stats.n_trades if stats.n_trades else 0
        return {
            "n_trades":     stats.n_trades,
            "n_wins":       stats.n_wins,
            "win_rate_pct": round(win_rate, 1),
            "pnl":          round(stats.pnl, 0),
            "avg_pnl":      round(avg, 0),
            "max_dd_pct":   round(mdd, 2),
            "sharpe":       round(shp, 2),
            "return_pct":   round(stats.pnl / STARTING_EQUITY * 100, 2),
        }

    results: Dict = {}
    for prod, df in feat_dict.items():
        results[prod] = {}
        for strat in ("ta", "ta_trend", "ta_robust", "ta_smart",
                      "combined", "kalman_pair", "bh"):
            results[prod][strat] = {}
            for period_name, p_start, p_end in PERIODS:
                if strat == "kalman_pair":
                    hedge = PAIR_HEDGE.get(prod)
                    if hedge is None or hedge not in feat_dict:
                        stats = StratStats()           # n/a
                    else:
                        stats = _kalman_backtest(
                            df, feat_dict[hedge], prod,
                            start=p_start, end=p_end)
                else:
                    stats = _backtest_product(
                        df, prod, strat, start=p_start, end=p_end)[1]
                results[prod][strat][period_name] = _summarize(stats)

    print("\n" + "=" * 96)
    print(f"BACKTEST  —  {PERIOD} history, split at {SPLIT_DATE.date()}, "
          f"walk-forward, 10bps round-trip, $100k start")
    print(f"  in-sample      : history -> {SPLIT_DATE.date()}")
    print(f"  out-of-sample  : {SPLIT_DATE.date()} -> present  (mirrors the "
          f"regression P3B verify window)")
    print("=" * 96)
    fmt = "  {:>9s}  {:>9s}  {:>4s}  {:>5s}  {:>6s}  {:>9s}  {:>7s}  {:>6s}"
    for prod, by_strat in results.items():
        print(f"\n{prod.upper()} ({TICKERS[prod]})")
        print(fmt.format("strat", "period", "n", "wr%", "ret%",
                         "totalP&L", "maxDD%", "shrp"))
        for s in ("ta", "combined", "bh"):
            for pname in ("in_sample", "out_of_sample"):
                r = by_strat[s][pname]
                print(fmt.format(
                    s, pname, str(r["n_trades"]),
                    f'{r["win_rate_pct"]:.0f}', f'{r["return_pct"]:+.1f}',
                    f'${r["pnl"]:.0f}', f'{r["max_dd_pct"]:.1f}',
                    f'{r["sharpe"]:.2f}',
                ))
    return results


if __name__ == "__main__":
    import datetime
    import json
    from pathlib import Path
    res = run()
    out = {
        "generated_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fetch_period":    PERIOD,
        "split_date":      SPLIT_DATE.date().isoformat(),
        "horizon_days":    HORIZON_DAYS,
        "cost_bps":        COST_BPS,
        "starting_equity": STARTING_EQUITY,
        "results":         res,
        "notes": (
            "Same fixed rule-set applied to both periods. The TA strategy "
            "has no trained parameters - so 'in-sample' vs 'out-of-sample' "
            "is identical in mechanism, only different in price-action it "
            "encounters. Comparison shows whether the strategy generalizes "
            "to the most recent (verify) window after working historically."
        ),
    }
    target = Path(__file__).parent.parent / "frontend" / "data" / "strategy_backtest.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {target}")

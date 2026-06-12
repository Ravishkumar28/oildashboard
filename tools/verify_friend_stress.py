"""Stress-test the friend's backtest by deliberately INTRODUCING common biases
one at a time, and see which one gets us to their reported numbers.

Their reported MR_zscore on CL fly M1-M2-M12:
  92 trades, $40.4 PnL, 67% wins, $22.7 max DD, Sharpe 0.44

My honest version got:
  16 trades, $0.2 PnL, 56% wins, $10.0 max DD, Sharpe 0.04

The 200x PnL gap is way too large to be implementation noise. Something is
fundamentally different. Tests:

  Variant A: No execution lag        (fill on signal day at same close)
  Variant B: No transaction cost      ($0 instead of $0.02 per turn)
  Variant C: Look-ahead z-score       (use FULL-SAMPLE mu/sd instead of train-only)
  Variant D: Continuous re-entry      (re-enter immediately if z crosses again)
  Variant E: Different fly weights    (M1 + M12 - 2*M2 vs M1 - 2*M2 + M12, etc)
  Variant F: All biases combined      (the kitchen-sink "look how good my model is" run)
"""
from __future__ import annotations
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "CL_data_trimmed_daily_close.xlsx"

TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 504, 63, 63
ENTRY_Z, EXIT_Z = 1.5, 0.3


def load_fly():
    df = pd.read_excel(XLSX)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    m1  = pd.to_numeric(df["c1||weighted_mid"],  errors="coerce")
    m2  = pd.to_numeric(df["c2||weighted_mid"],  errors="coerce")
    m12 = pd.to_numeric(df["c12||weighted_mid"], errors="coerce")
    return (m1 - 2*m2 + m12).dropna()


def run_backtest(fly, *, exec_lag=1, cost=0.02, lookahead=False,
                  continuous=False):
    """Walk-forward MR_zscore with configurable biases."""
    n = len(fly)
    pnls = []
    full_mu = float(fly.mean())   # for look-ahead variant
    full_sd = float(fly.std(ddof=1))

    start = 0
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train = fly.iloc[start : start + TRAIN_DAYS]
        test  = fly.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        if lookahead:
            mu, sd = full_mu, full_sd
        else:
            mu, sd = float(train.mean()), float(train.std(ddof=1))
        if sd <= 0:
            start += STEP_DAYS
            continue
        z = (test - mu) / sd

        position = 0
        entry_px = None
        for i in range(len(test) - exec_lag):
            zi = float(z.iloc[i])
            fill_px = float(test.iloc[i + exec_lag])
            if position == 0:
                if zi >= +ENTRY_Z:
                    position = -1
                    entry_px = fill_px
                elif zi <= -ENTRY_Z:
                    position = +1
                    entry_px = fill_px
            else:
                # Exit when |z| crosses below threshold OR (continuous) when z
                # crosses zero (so we can flip immediately)
                exit_now = (abs(zi) <= EXIT_Z)
                if continuous and not exit_now:
                    # If z flips sign, also exit (so we can re-enter opposite)
                    if (position > 0 and zi > 0) or (position < 0 and zi < 0):
                        exit_now = True
                if exit_now:
                    raw = position * (fill_px - entry_px)
                    pnls.append(raw - 2*cost)
                    position = 0
                    entry_px = None
        # Force close at window end
        if position != 0 and entry_px is not None:
            fill_px = float(test.iloc[-1])
            raw = position * (fill_px - entry_px)
            pnls.append(raw - 2*cost)
        start += STEP_DAYS

    if not pnls:
        return None
    arr = np.array(pnls)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() if len(cum) else 0.0
    if arr.std(ddof=1) > 0:
        sharpe = arr.mean() / arr.std(ddof=1) * math.sqrt(252.0 / 5.0)
    else:
        sharpe = 0.0
    return {
        "n_trades": len(arr),
        "pnl":      float(arr.sum()),
        "win_pct":  float((arr > 0).mean() * 100),
        "max_dd":   float(dd),
        "sharpe":   float(sharpe),
    }


def main():
    fly = load_fly()
    print(f"CL fly M1-2M2+M12  n={len(fly)}  range "
          f"({fly.index[0].date()} -> {fly.index[-1].date()})")
    print()
    print(f"Friend reported (MR_zscore):  92 trades, $40.4 PnL, 67%, "
          f"$22.7 DD, Sharpe 0.44")
    print()

    variants = [
        ("Honest baseline (no biases)",     dict(exec_lag=1, cost=0.02, lookahead=False, continuous=False)),
        ("A: No execution lag",             dict(exec_lag=0, cost=0.02, lookahead=False, continuous=False)),
        ("B: No transaction cost",          dict(exec_lag=1, cost=0.00, lookahead=False, continuous=False)),
        ("C: Look-ahead z-score",           dict(exec_lag=1, cost=0.02, lookahead=True,  continuous=False)),
        ("D: Continuous re-entry",          dict(exec_lag=1, cost=0.02, lookahead=False, continuous=True)),
        ("F: A+B+C+D (kitchen sink)",       dict(exec_lag=0, cost=0.00, lookahead=True,  continuous=True)),
    ]
    print(f"  {'Variant':<32}{'Trades':>8}{'PnL$':>10}{'Win%':>8}{'MaxDD$':>10}{'Sharpe':>8}")
    print("  " + "-"*76)
    for name, kw in variants:
        r = run_backtest(fly, **kw)
        if r is None:
            print(f"  {name:<32}{'-':>8}")
            continue
        print(f"  {name:<32}{r['n_trades']:>8d}{r['pnl']:>+10.2f}"
              f"{r['win_pct']:>7.1f}%{r['max_dd']:>10.2f}{r['sharpe']:>+8.2f}")

    print()
    print("If any variant matches the friend's 92 trades + $40 PnL closely,")
    print("that explains their methodology. If NOTHING matches, the result")
    print("is inflated by something not captured here (e.g. different lookbacks,")
    print("different walk-forward, or scaled position sizing).")


if __name__ == "__main__":
    main()

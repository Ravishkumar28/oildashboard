"""Verify a friend's reported walk-forward backtest results on CL fly M1-M2-M12.

Reported setup (from screenshot):
  Instrument:  CL fly M1-M2-M12  (= M1 - 2*M2 + M12)
  Walk-forward: train=504d / test=63d / step=63d
  Execution lag: 1 day (signal -> trade next day)
  Costs: $0.02/unit turn (bid-ask + slippage proxy)
  Entry: |z| >= 1.5
  Exit:  |z| <= 0.3

Reported top-3 strategies on this instrument:
  ★ MR_bollinger:   PnL=$40.7  Sharpe=0.70  MaxDD=-9.0   Win%=62.9  Trades=70
    MR_zscore:      PnL=$40.4  Sharpe=0.44  MaxDD=-22.7  Win%=67.4  Trades=92
    MR_ou:          PnL=$44.7  Sharpe=0.55  MaxDD=-17.7  Win%=55.1  Trades=234

This script independently implements MR_zscore (the simplest and most replicable
of the three) on the actual CL xlsx data and reports its walk-forward stats.
Then we compare to the friend's numbers to flag any inflation.
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

# Reported parameters (verbatim from screenshot)
TRAIN_DAYS  = 504
TEST_DAYS   = 63
STEP_DAYS   = 63
ENTRY_Z     = 1.5
EXIT_Z      = 0.3
COST_PER_UNIT_TURN = 0.02   # $ per fly unit per round-trip
EXEC_LAG    = 1             # signal day -> trade day+1


def load_cl_fly_M1_M2_M12() -> pd.Series:
    df = pd.read_excel(XLSX)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    m1  = pd.to_numeric(df["c1||weighted_mid"],  errors="coerce")
    m2  = pd.to_numeric(df["c2||weighted_mid"],  errors="coerce")
    m12 = pd.to_numeric(df["c12||weighted_mid"], errors="coerce")
    fly = m1 - 2*m2 + m12
    return fly.dropna()


def walk_forward_zscore(fly: pd.Series) -> dict:
    """Walk-forward MR_zscore: in each TEST window, use TRAIN window's mean/std
    for the z-score, then trade |z| >= ENTRY_Z (mean-revert: short if z>+1.5,
    long if z<-1.5), exit when |z| <= EXIT_Z. 1-day execution lag."""
    n = len(fly)
    if n < TRAIN_DAYS + TEST_DAYS:
        return {"error": f"not enough data ({n} < {TRAIN_DAYS+TEST_DAYS})"}

    all_pnls   = []
    all_trades = []  # for diagnostics
    daily_pnls = []  # for Sharpe
    daily_dates = []
    equity = 0.0
    peak_equity = 0.0
    max_dd = 0.0

    # Walk-forward windows
    start = 0
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train = fly.iloc[start : start + TRAIN_DAYS]
        test  = fly.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        mu  = float(train.mean())
        sd  = float(train.std(ddof=1))
        if sd <= 0:
            start += STEP_DAYS
            continue

        # z-score series on the test slice using TRAIN mu/sd
        z = (test - mu) / sd

        # State machine: walk through test days with 1-day exec lag
        position = 0     # +1 long fly, -1 short fly, 0 flat
        entry_px = None
        for i in range(len(test) - EXEC_LAG):
            zi    = float(z.iloc[i])
            # PnL settles when we ENTER or EXIT (using next-day price as fill)
            fill_px = float(test.iloc[i + EXEC_LAG])
            if position == 0:
                # Look for entry
                if zi >= +ENTRY_Z:
                    position = -1   # fly is rich -> short it
                    entry_px = fill_px
                elif zi <= -ENTRY_Z:
                    position = +1   # fly is cheap -> long it
                    entry_px = fill_px
            else:
                # Already in a position - check exit
                if abs(zi) <= EXIT_Z:
                    raw_pnl = position * (fill_px - entry_px)
                    # 2x cost: entry + exit
                    pnl = raw_pnl - 2 * COST_PER_UNIT_TURN
                    all_pnls.append(pnl)
                    all_trades.append({
                        "entry_date": test.index[i - 1] if i > 0 else test.index[0],
                        "exit_date":  test.index[i + EXEC_LAG],
                        "direction":  position,
                        "entry_px":   entry_px,
                        "exit_px":    fill_px,
                        "pnl":        pnl,
                    })
                    daily_pnls.append(pnl)
                    daily_dates.append(test.index[i + EXEC_LAG])
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    dd = peak_equity - equity
                    if dd > max_dd:
                        max_dd = dd
                    position = 0
                    entry_px = None
        # Force-close at window end (so we don't carry positions across windows)
        if position != 0 and entry_px is not None:
            fill_px = float(test.iloc[-1])
            raw_pnl = position * (fill_px - entry_px)
            pnl = raw_pnl - 2 * COST_PER_UNIT_TURN
            all_pnls.append(pnl)
            daily_pnls.append(pnl)
            equity += pnl
            peak_equity = max(peak_equity, equity)
            dd = peak_equity - equity
            if dd > max_dd:
                max_dd = dd

        start += STEP_DAYS

    if not all_pnls:
        return {"error": "no trades generated"}

    arr = np.array(all_pnls)
    wins = (arr > 0).sum()
    losses = (arr < 0).sum()
    win_rate = wins / len(arr) * 100

    # Annualized Sharpe on the per-trade PnL stream
    # Friend's spec says daily $ P&L, so we use a daily approximation
    if arr.std(ddof=1) > 0:
        sharpe = arr.mean() / arr.std(ddof=1) * math.sqrt(252.0 / 5.0)  # ~5d hold
    else:
        sharpe = 0.0
    # Sortino - downside std only
    neg = arr[arr < 0]
    sortino = (arr.mean() / neg.std(ddof=1) * math.sqrt(252.0 / 5.0)) if len(neg) > 1 and neg.std() > 0 else 0.0
    calmar = (arr.sum() / max_dd) if max_dd > 0 else 0.0

    return {
        "n_trades": int(len(arr)),
        "wins":     int(wins),
        "losses":   int(losses),
        "total_pnl_dollars": float(arr.sum()),
        "mean_pnl":         float(arr.mean()),
        "win_rate":         float(win_rate),
        "max_dd_dollars":   float(max_dd),
        "sharpe":           float(sharpe),
        "sortino":          float(sortino),
        "calmar":           float(calmar),
    }


def main():
    fly = load_cl_fly_M1_M2_M12()
    print(f"Loaded CL fly M1-2M2+M12: {len(fly)} days  "
          f"({fly.index[0].date()} -> {fly.index[-1].date()})")
    print(f"Fly summary: mean={fly.mean():+.3f}  std={fly.std():.3f}  "
          f"min={fly.min():+.3f}  max={fly.max():+.3f}")
    print()

    # Friend's reported MR_zscore numbers
    REPORTED = {
        "n_trades": 92,
        "total_pnl_dollars": 40.4,
        "win_rate": 67.4,
        "max_dd_dollars": 22.7,
        "sharpe": 0.44,
        "sortino": 0.33,
        "calmar": 1.78,
    }

    result = walk_forward_zscore(fly)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("=" * 75)
    print("WALK-FORWARD MR_ZSCORE on CL fly M1-2M2+M12")
    print("=" * 75)
    print(f"Setup: train={TRAIN_DAYS}d / test={TEST_DAYS}d / step={STEP_DAYS}d / "
          f"entry|z|>={ENTRY_Z} / exit|z|<={EXIT_Z} / cost=${COST_PER_UNIT_TURN}/turn / "
          f"exec_lag={EXEC_LAG}d")
    print()
    print(f"  {'Metric':<22}{'Friend reported':>18}{'My calc':>18}{'diff':>15}")
    print("  " + "-"*70)
    metrics = [
        ("n_trades", "{:>18d}", lambda x: x),
        ("total_pnl_dollars", "${:>16.1f}", lambda x: x),
        ("win_rate", "{:>17.1f}%", lambda x: x),
        ("max_dd_dollars", "${:>16.1f}", lambda x: x),
        ("sharpe", "{:>18.2f}", lambda x: x),
        ("sortino", "{:>18.2f}", lambda x: x),
        ("calmar", "{:>18.2f}", lambda x: x),
    ]
    for name, fmt, _ in metrics:
        r = REPORTED[name]
        m = result[name]
        r_str = fmt.format(r)
        m_str = fmt.format(m)
        if isinstance(r, (int, float)) and isinstance(m, (int, float)) and r != 0:
            delta_pct = (m - r) / abs(r) * 100
            delta_str = f"{delta_pct:>+13.1f}%"
        else:
            delta_str = "n/a"
        print(f"  {name:<22}{r_str}{m_str}{delta_str}")

    print()
    print("=" * 75)
    print("INTERPRETATION")
    print("=" * 75)
    n_diff = abs(result["n_trades"] - REPORTED["n_trades"])
    pnl_diff = abs(result["total_pnl_dollars"] - REPORTED["total_pnl_dollars"])
    if n_diff <= 5 and pnl_diff <= 5:
        print("  Numbers are CLOSE to the friend's report - results plausible")
    elif n_diff <= 15 and pnl_diff <= 15:
        print("  Numbers are in the same ballpark, some differences (likely")
        print("  implementation details: how partial trades handled, exact lag,")
        print("  exact cost model)")
    else:
        print("  Numbers are MATERIALLY DIFFERENT from the friend's report.")
        print("  Either their implementation differs (likely look-ahead leakage,")
        print("  different cost assumptions, or different fly definition) or the")
        print("  reported numbers are inflated.")

if __name__ == "__main__":
    main()

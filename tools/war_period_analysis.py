"""3-month war-period analysis: VaR, PnL, Sharpe, drawdown for every strategy.

Uses the LAST 63 trading days of the xlsx data as proxy for the war-onset
window. With the CL data ending 2026-05-22, this covers approximately
2026-02-20 to 2026-05-22 — the period when Israel-Iran tensions escalated
into open conflict and the WTCL spread hit historic widths.

For each strategy implemented on the dashboard, we run it on ONLY this
window (signals fit on training history, executed on the 63-day window)
and compute:

  - Total PnL ($ per 1,000-bbl unit)
  - Mean / std per trade
  - 95% VaR  (loss exceeded 5% of the time)
  - 99% VaR  (loss exceeded 1% of the time)
  - 95% CVaR (expected loss when 95% VaR is breached)
  - Max drawdown
  - Win rate
  - Sharpe (annualized)
  - n_trades

Strategies tested:
  S1 PCA_Curve_CL    PC3 fly mean reversion on CL
  S2 PCA_Curve_LCO   PC3 fly on LCO (Brent)
  S3 PCA_Curve_LGO   PC3 fly on LGO (Gasoil)
  S4 Bertram_HO      OU calendar spreads on HO
  S5 HMM_WTCL        Regime-switching on WTI-Brent
  S6 MR_zscore_CL    Classic z-score MR on CL fly M1-2M2+M12
  S7 MR_boll_CL      Bollinger band MR on CL fly
  S8 BuyHold_M1      Naive long M1 per product (benchmark)
"""
from __future__ import annotations
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_FILES = {
    "CL":   ROOT / "CL_data_trimmed_daily_close.xlsx",
    "LCO":  ROOT / "LCO_data_trimmed_daily_close.xlsx",
    "LGO":  ROOT / "LGO_data_trimmed_daily_close.xlsx",
    "HO":   ROOT / "HO_data_trimmed_daily_close.xlsx",
    "WTCL": ROOT / "wtcl_lco_outrights_1min_trimmed_daily_close.xlsx",
}
WAR_WINDOW_DAYS = 63    # last ~3 months
COST_BPS = 5.0          # 5 bps round-trip
NOTIONAL = 1000         # bbl per contract


def load_curve(path):
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=df.index)
    for i in range(1, 15):
        c = f"c{i}||weighted_mid"
        if c in df.columns:
            out[f"m{i}"] = pd.to_numeric(df[c], errors="coerce")
    return out


def settle(direction, entry_px, exit_px):
    gross = direction * (exit_px - entry_px) * NOTIONAL
    cost = (COST_BPS / 10000.0) * abs(entry_px) * NOTIONAL
    return gross - cost


def stats(pnls, label):
    if not pnls:
        return {"strategy": label, "n_trades": 0,
                "total_pnl": 0, "var95": 0, "var99": 0,
                "cvar95": 0, "max_dd": 0, "win_rate": 0, "sharpe": 0,
                "mean_pnl": 0, "std_pnl": 0, "max_win": 0, "max_loss": 0}
    if len(pnls) == 1:
        # Single trade: report it directly; VaR/Sharpe undefined
        return {"strategy": label, "n_trades": 1,
                "total_pnl": float(pnls[0]),
                "mean_pnl": float(pnls[0]), "std_pnl": 0,
                "win_rate": 100.0 if pnls[0] > 0 else 0.0,
                "var95": float(max(0, -pnls[0])),   # if it lost
                "var99": float(max(0, -pnls[0])),
                "cvar95": float(max(0, -pnls[0])),
                "max_dd": float(max(0, -pnls[0])),
                "sharpe": 0,   # undefined with n=1
                "max_win": float(max(0, pnls[0])),
                "max_loss": float(min(0, pnls[0])),
                "single_trade": True}
    arr = np.array(pnls)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) if len(cum) else 0
    win_rate = float((arr > 0).mean() * 100)
    losses = -arr   # convert to losses (positive = bad)
    var95 = float(np.percentile(losses, 95))
    var99 = float(np.percentile(losses, 99))
    tail = losses[losses >= var95]
    cvar95 = float(tail.mean()) if len(tail) else var95
    if arr.std(ddof=1) > 0:
        # 3-month window: annualize with sqrt(252/63)
        sharpe = arr.mean() / arr.std(ddof=1) * math.sqrt(252.0 / 63.0)
    else:
        sharpe = 0
    return {
        "strategy":  label,
        "n_trades":  int(len(arr)),
        "total_pnl": float(arr.sum()),
        "mean_pnl":  float(arr.mean()),
        "std_pnl":   float(arr.std(ddof=1)),
        "var95":     var95,
        "var99":     var99,
        "cvar95":    cvar95,
        "max_dd":    dd,
        "win_rate":  win_rate,
        "sharpe":    sharpe,
        "max_win":   float(arr.max()),
        "max_loss":  float(arr.min()),
    }


# ----- Strategy implementations (war-window only) ------------------------ #
def pca_curve(curve, war_start_idx):
    """Fit PCA on history before war window; trade PC3 fly on war window."""
    cols = [f"m{i}" for i in range(1, 13) if f"m{i}" in curve.columns]
    if len(cols) < 8:
        return []
    prices = curve[cols].dropna()
    if len(prices) < 250 or war_start_idx >= len(prices) - 5:
        return []
    log_ret = np.log(prices / prices.shift(1)).dropna()
    train_idx = max(0, war_start_idx - 252 - 1)
    train_X = log_ret.iloc[train_idx:war_start_idx].values
    pca = PCA(n_components=3); pca.fit(train_X)
    pc3_load = pca.components_[2]
    sorted_loads = np.argsort(np.abs(pc3_load))[::-1]
    leg_indices = sorted_loads[:3]
    leg_signs = np.sign(pc3_load[leg_indices])
    leg_cols = [cols[i] for i in leg_indices]
    test_log_ret = log_ret.iloc[war_start_idx:]
    scores = pca.transform(test_log_ret.values)
    pc3 = scores[:, 2]
    # rolling z of PC3
    s = pd.Series(pc3)
    z = (s - s.rolling(20).mean()) / s.rolling(20).std()
    test_prices = prices.loc[test_log_ret.index, leg_cols]
    fly = (test_prices * leg_signs).sum(axis=1)

    pnls = []
    position = 0; entry_i = None
    for i in range(len(z)):
        zi = z.iloc[i]
        if not np.isfinite(zi):
            continue
        if position == 0:
            if zi >= 2.0: position = -1; entry_i = i
            elif zi <= -2.0: position = 1; entry_i = i
        else:
            hold = i - entry_i
            if abs(zi) <= 0.5 or hold >= 15:
                if entry_i < len(fly) and i < len(fly):
                    pnls.append(settle(position, fly.iloc[entry_i], fly.iloc[i]))
                position = 0; entry_i = None
    return pnls


def bertram_ho(curve, war_start_idx):
    """OU mean reversion on HO M1-M2 through M5-M6 calendar spreads."""
    pnls = []
    for a, b in [(1,2),(2,3),(3,4),(4,5),(5,6)]:
        ka, kb = f"m{a}", f"m{b}"
        if ka not in curve.columns or kb not in curve.columns:
            continue
        s_raw = (curve[ka] - curve[kb]).dropna()
        if war_start_idx >= len(s_raw) - 5:
            continue
        train = s_raw.iloc[:war_start_idx]
        if len(train) < 200:
            continue
        mu = float(train.mean())
        ds = train.diff().dropna()
        slope, _ = np.polyfit(train.shift(1).dropna().reindex(ds.index).values,
                              ds.values, 1)
        alpha = -slope; eta = float(ds.std(ddof=1))
        if alpha <= 0 or eta <= 0:
            continue
        sigma_eq = eta / math.sqrt(2 * alpha)
        a_star = 1.5 * sigma_eq
        test = s_raw.iloc[war_start_idx:]
        position = 0; entry_i = None
        for i in range(len(test)):
            s_now = float(test.iloc[i])
            dev = s_now - mu
            if position == 0:
                if dev >= a_star: position = -1; entry_i = i
                elif dev <= -a_star: position = 1; entry_i = i
            else:
                hold = i - entry_i
                if (position > 0 and dev >= 0) or \
                   (position < 0 and dev <= 0) or hold >= 30:
                    pnls.append(settle(position, test.iloc[entry_i], test.iloc[i]))
                    position = 0; entry_i = None
    return pnls


def mr_zscore_cl_fly(curve, war_start_idx, entry_z=1.5, exit_z=0.3):
    """MR_zscore on CL fly M1-2M2+M12 (the friend's tested instrument)."""
    if not all(c in curve.columns for c in ("m1","m2","m12")):
        return []
    fly = (curve["m1"] - 2*curve["m2"] + curve["m12"]).dropna()
    if war_start_idx >= len(fly) - 5:
        return []
    train = fly.iloc[:war_start_idx]
    if len(train) < 200:
        return []
    mu = float(train.mean()); sd = float(train.std(ddof=1))
    if sd == 0:
        return []
    test = fly.iloc[war_start_idx:]
    z = (test - mu) / sd
    pnls = []
    position = 0; entry_i = None
    for i in range(len(test)):
        zi = float(z.iloc[i])
        if position == 0:
            if zi >= entry_z: position = -1; entry_i = i
            elif zi <= -entry_z: position = 1; entry_i = i
        else:
            if abs(zi) <= exit_z:
                pnls.append(settle(position, test.iloc[entry_i], test.iloc[i]))
                position = 0; entry_i = None
    # Force-close at end
    if position != 0:
        pnls.append(settle(position, test.iloc[entry_i], test.iloc[-1]))
    return pnls


def mr_bollinger_cl_fly(curve, war_start_idx, window=20, n_std=2.0):
    """Bollinger band MR on CL fly M1-2M2+M12."""
    if not all(c in curve.columns for c in ("m1","m2","m12")):
        return []
    fly = (curve["m1"] - 2*curve["m2"] + curve["m12"]).dropna()
    if war_start_idx >= len(fly) - 5:
        return []
    # Compute Bollinger on TRAINING portion only (avoid leakage)
    train = fly.iloc[:war_start_idx]
    if len(train) < window:
        return []
    test = fly.iloc[war_start_idx:]
    # Use train's std but rolling mean (causal)
    rolling_mean = fly.rolling(window).mean()
    rolling_std  = fly.rolling(window).std()
    bb_upper = rolling_mean + n_std * rolling_std
    bb_lower = rolling_mean - n_std * rolling_std
    bb_mid   = rolling_mean
    test_upper = bb_upper.reindex(test.index)
    test_lower = bb_lower.reindex(test.index)
    test_mid   = bb_mid.reindex(test.index)
    pnls = []
    position = 0; entry_i = None
    for i in range(len(test)):
        px = float(test.iloc[i])
        if not np.isfinite(test_upper.iloc[i]):
            continue
        if position == 0:
            if px >= test_upper.iloc[i]: position = -1; entry_i = i
            elif px <= test_lower.iloc[i]: position = 1; entry_i = i
        else:
            cross_mid = (position > 0 and px >= test_mid.iloc[i]) or \
                        (position < 0 and px <= test_mid.iloc[i])
            if cross_mid:
                pnls.append(settle(position, test.iloc[entry_i], test.iloc[i]))
                position = 0; entry_i = None
    if position != 0:
        pnls.append(settle(position, test.iloc[entry_i], test.iloc[-1]))
    return pnls


def buy_hold(curve, war_start_idx, tenor="m1"):
    """Naive long-M1 over the war window."""
    if tenor not in curve.columns:
        return []
    s = curve[tenor].dropna()
    if war_start_idx >= len(s) - 5:
        return []
    test = s.iloc[war_start_idx:]
    return [settle(1, test.iloc[0], test.iloc[-1])]


def main():
    print(f"Loading 5 products...")
    curves = {p: load_curve(path) for p, path in PRODUCT_FILES.items() if path.exists()}
    # Find a common war_start cutoff - take the most recent 63 days from each
    # product's own end date so we hit the same calendar period
    print()
    print("=" * 92)
    print(f"WAR-PERIOD ANALYSIS (last {WAR_WINDOW_DAYS} trading days per product)")
    print("=" * 92)

    for prod, curve in curves.items():
        n = len(curve)
        war_start = n - WAR_WINDOW_DAYS
        if war_start < 250:
            continue
        date_from = curve.index[war_start].date()
        date_to   = curve.index[-1].date()
        print(f"\n[{prod}]  window: {date_from} -> {date_to}  ({WAR_WINDOW_DAYS}d)")

    # Per-strategy run
    print()
    print("=" * 105)
    print("PER-STRATEGY RESULTS — 3-month window, 5bps cost, 1,000-bbl notional")
    print("=" * 105)
    print(f"  {'Strategy':<24}{'#Trades':>9}{'PnL$':>10}{'WinR%':>7}{'95%VaR':>10}{'99%VaR':>10}"
          f"{'CVaR95':>10}{'MaxDD':>10}{'Sharpe':>8}")
    print("  " + "-" * 100)

    results = []

    for prod in ("CL", "LCO", "LGO"):
        if prod in curves:
            curve = curves[prod]
            war_start = len(curve) - WAR_WINDOW_DAYS
            if war_start >= 250:
                pnls = pca_curve(curve, war_start)
                results.append(stats(pnls, f"PCA_Curve_{prod}"))

    if "HO" in curves:
        curve = curves["HO"]
        war_start = len(curve) - WAR_WINDOW_DAYS
        if war_start >= 250:
            pnls = bertram_ho(curve, war_start)
            results.append(stats(pnls, "Bertram_HO"))

    if "CL" in curves:
        curve = curves["CL"]
        war_start = len(curve) - WAR_WINDOW_DAYS
        if war_start >= 200:
            pnls = mr_zscore_cl_fly(curve, war_start)
            results.append(stats(pnls, "MR_zscore_CL_fly"))
            pnls_bb = mr_bollinger_cl_fly(curve, war_start)
            results.append(stats(pnls_bb, "MR_bollinger_CL_fly"))

    for prod in ("CL", "LCO", "HO"):
        if prod in curves:
            curve = curves[prod]
            war_start = len(curve) - WAR_WINDOW_DAYS
            if war_start >= 1:
                pnls = buy_hold(curve, war_start)
                results.append(stats(pnls, f"BuyHold_{prod}_M1"))

    for r in sorted(results, key=lambda x: -x.get("total_pnl", 0)):
        print(f"  {r['strategy']:<24}{r['n_trades']:>9}{r['total_pnl']:>+10,.0f}"
              f"{r['win_rate']:>6.0f}%"
              f"{r['var95']:>+10,.0f}{r['var99']:>+10,.0f}"
              f"{r['cvar95']:>+10,.0f}{r['max_dd']:>+10,.0f}"
              f"{r['sharpe']:>+8.2f}")

    print()
    print("=" * 105)
    print("INTERPRETATION")
    print("=" * 105)
    profitable = [r for r in results if r["total_pnl"] > 0]
    losers = [r for r in results if r["total_pnl"] < 0]
    print(f"  Profitable: {len(profitable)}    Losers: {len(losers)}")
    if profitable:
        best = max(profitable, key=lambda r: r["total_pnl"])
        print(f"  Best PnL: {best['strategy']} (${best['total_pnl']:+,.0f})")
        if best["var95"] > 0:
            print(f"    For every $1 of 95% VaR risk, this strategy made ${best['total_pnl']/best['var95']:+,.1f}")
    if losers:
        worst = min(losers, key=lambda r: r["total_pnl"])
        print(f"  Worst PnL: {worst['strategy']} (${worst['total_pnl']:+,.0f})")
    # Risk reward
    print()
    print("  PnL / 95%VaR ratio (higher = better risk-adjusted):")
    rated = [(r, r["total_pnl"] / r["var95"] if r["var95"] > 0 else 0) for r in results
             if r["n_trades"] > 0]
    for r, ratio in sorted(rated, key=lambda x: -x[1]):
        marker = "★" if ratio > 2 else ("." if ratio > 0 else "")
        print(f"    {r['strategy']:<24}{ratio:>+8.2f}  {marker}")


if __name__ == "__main__":
    main()

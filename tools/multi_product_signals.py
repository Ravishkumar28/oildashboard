"""Multi-product LGBM signal generator + strategy backtest.

For each of the 5 dashboard products (WTI, Brent, RBOB, HO, NatGas):
  1. Pull 5y daily Close via yfinance.
  2. Build 8 features (returns, vol, BB, DXY, RSI).
  3. Train LGBM_dash (conservative, the per-regime dashboard config) and
     linear-family models on first 80% of bars.
  4. Generate per-day directional signals on the held-out 20%:
        prediction > +THRESHOLD %  →  LONG
        prediction < -THRESHOLD %  →  SHORT
        otherwise                  →  FLAT
  5. Backtest each signal: PnL = sum of (signal_t * realized_return_t+5).
     Report Sharpe, win rate, max drawdown, trade count.

Plus 3 cross-product SPREAD strategies (real desk patterns):
  - WTI-Brent (TI/Brent transatlantic arb)
  - 321 Crack proxy: 2*RBOB + HO - 3*WTI (refining margin)
  - RBOB-HO (gasoline/distillate ratio)

Each spread is fit with the same LGBM+linear stack and backtested.
"""
from __future__ import annotations
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                   ElasticNet, HuberRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

PERIOD  = "5y"
HORIZON = 5
SIG_THRESHOLD = 0.5    # % — predicted return magnitude required to take a trade
COST_BPS = 5.0         # 5bps round-trip frictional cost per trade

PRODUCTS = {
    "WTI":    "CL=F",
    "Brent":  "BZ=F",
    "RBOB":   "RB=F",
    "HO":     "HO=F",
    "NatGas": "NG=F",
}

FEAT_NAMES = ["ret_1d", "ret_5d", "ret_20d",
              "dxy_5d_ret", "dxy_20d_ret",
              "bb_pos", "vol_z", "rsi_14n"]


def _bb_pos(close, w=20):
    m = close.rolling(w).mean()
    s = close.rolling(w).std()
    return ((close - (m - 2*s)) / (4*s) * 100).clip(0, 100)


def _rsi(close, p=14):
    d = close.diff()
    up = d.where(d > 0, 0.0).rolling(p).mean()
    dn = (-d.where(d < 0, 0.0)).rolling(p).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _vol_z(close):
    lr = np.log(close / close.shift(1))
    v20 = lr.rolling(20).std() * math.sqrt(252) * 100
    return (v20 - v20.rolling(60).mean()) / v20.rolling(60).std()


def build_features(target_close, dxy_close):
    df = pd.DataFrame(index=target_close.index)
    df["ret_1d"]      = target_close.pct_change(1)  * 100
    df["ret_5d"]      = target_close.pct_change(5)  * 100
    df["ret_20d"]     = target_close.pct_change(20) * 100
    df["dxy_5d_ret"]  = dxy_close.pct_change(5)  * 100
    df["dxy_20d_ret"] = dxy_close.pct_change(20) * 100
    df["bb_pos"]      = _bb_pos(target_close, 20)
    df["vol_z"]       = _vol_z(target_close)
    df["rsi_14n"]     = (_rsi(target_close, 14) - 50) / 50
    y = (target_close.shift(-HORIZON) / target_close - 1) * 100
    df = df.assign(target=y).dropna()
    return df[FEAT_NAMES], df["target"], df.index


def lgbm_dash():
    return LGBMRegressor(
        n_estimators=30, max_depth=2, num_leaves=4,
        learning_rate=0.03, min_child_samples=8,
        min_data_in_bin=2, min_split_gain=0.0,
        reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        random_state=42, verbosity=-1, force_col_wise=True,
    )


def fit_predict(X_tr, y_tr, X_te, y_te):
    models = {
        "Linear":     LinearRegression(),
        "Ridge":      Ridge(alpha=1.0),
        "Lasso":      Lasso(alpha=0.05, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "Huber":      HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=400),
        "LGBM":       lgbm_dash(),
    }
    out = {}
    for n, m in models.items():
        m.fit(X_tr, y_tr)
        pred_tr = m.predict(X_tr)
        pred_te = m.predict(X_te)
        out[n] = {
            "train_r2": r2_score(y_tr, pred_tr),
            "test_r2":  r2_score(y_te, pred_te),
            "test_mae": mean_absolute_error(y_te, pred_te),
            "pred_te":  pred_te,
        }
    return out


def backtest_signals(pred_te, realized_te, threshold=SIG_THRESHOLD, cost_bps=COST_BPS):
    """Convert predictions to LONG/SHORT/FLAT signals, compute PnL stats."""
    signals = np.where(pred_te > threshold, 1,
                np.where(pred_te < -threshold, -1, 0))
    # Each trade lasts HORIZON days; assume non-overlapping by stepping HORIZON.
    pnl_per_trade = []
    sig_taken = []
    for i in range(0, len(signals), HORIZON):
        s = signals[i]
        if s == 0:
            continue
        sig_taken.append(s)
        pnl_per_trade.append(s * realized_te[i] - cost_bps/100.0)
    if not pnl_per_trade:
        return {
            "n_trades": 0, "pnl_total": 0.0, "win_rate": 0.0,
            "sharpe":   0.0, "max_dd": 0.0,
            "avg_win":  0.0, "avg_loss": 0.0,
            "long_pct": 0.0, "short_pct": 0.0,
        }
    arr = np.array(pnl_per_trade)
    sigs = np.array(sig_taken)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq).max() if len(eq) > 0 else 0.0
    sharpe = (arr.mean() / arr.std() * math.sqrt(252.0 / HORIZON)
              if arr.std() > 0 else 0.0)
    return {
        "n_trades": int(len(arr)),
        "pnl_total": float(arr.sum()),
        "win_rate": float((arr > 0).mean() * 100),
        "sharpe":   float(sharpe),
        "max_dd":   float(dd),
        "avg_win":  float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
        "long_pct": float((sigs == 1).mean() * 100),
        "short_pct": float((sigs == -1).mean() * 100),
    }


def analyze_series(label, target_close, dxy_close):
    X_df, y_ser, idx = build_features(target_close, dxy_close)
    if len(X_df) < 200:
        return None
    X = X_df.values.astype(float)
    y = y_ser.values.astype(float)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    fits = fit_predict(X_tr_s, y_tr, X_te_s, y_te)
    # Use LGBM-Dash for the signal — winner of the 5yr cross-product test.
    sig_stats = backtest_signals(fits["LGBM"]["pred_te"], y_te)
    return {
        "label": label,
        "n_train": len(X_tr),
        "n_test":  len(X_te),
        "fits":    fits,
        "signals": sig_stats,
    }


def fmt_row_fits(label, fits, families):
    parts = [f"  {label:<22}"]
    for f in families:
        r = fits[f]
        parts.append(f"{r['test_r2']:>+7.3f}")
    winner = max(families, key=lambda f: fits[f]["test_r2"])
    parts.append(f"  -> {winner}")
    return "  ".join(parts)


def main():
    print(f"Pulling {PERIOD} of daily Close...")
    tickers = list(PRODUCTS.values()) + ["DX-Y.NYB"]
    raw = yf.download(tickers, period=PERIOD,
                      progress=False, auto_adjust=True, threads=True)
    close = raw["Close"]
    dxy = close["DX-Y.NYB"]

    # ----- PART 1: per-product fits + signals --------------------------------
    results = {}
    for p, tkr in PRODUCTS.items():
        ts = close[tkr].dropna()
        dxy_a = dxy.reindex(ts.index, method="ffill")
        results[p] = analyze_series(p, ts, dxy_a)

    families = ["Linear", "Ridge", "Lasso", "ElasticNet", "Huber", "LGBM"]
    print()
    print("=" * 80)
    print("PART 1 — Per-product directional models (next 5d return, 80/20 split)")
    print("=" * 80)
    print(f"  {'Product':<22}" + "  ".join(f"{f:>7}" for f in families) + "    winner")
    print("  " + "-" * 78)
    for p in PRODUCTS:
        r = results[p]
        if r is None: continue
        print(fmt_row_fits(p, r["fits"], families))

    print()
    print("=" * 80)
    print("PART 2 — Per-product LGBM-Dash strategy backtest (signal = pred ± "
          f"{SIG_THRESHOLD}%, {COST_BPS}bps cost)")
    print("=" * 80)
    print(f"  {'Product':<10}{'Trades':>8}{'PnL%':>9}{'Win%':>8}{'Sharpe':>9}"
          f"{'MaxDD':>9}{'AvgWin':>9}{'AvgLoss':>9}{'L/S':>10}")
    print("  " + "-" * 70)
    for p in PRODUCTS:
        r = results[p]
        if r is None: continue
        s = r["signals"]
        print(f"  {p:<10}{s['n_trades']:>8}{s['pnl_total']:>+9.2f}"
              f"{s['win_rate']:>7.1f}%{s['sharpe']:>+9.2f}"
              f"{s['max_dd']:>9.2f}{s['avg_win']:>+9.2f}{s['avg_loss']:>+9.2f}"
              f"  {s['long_pct']:.0f}/{s['short_pct']:.0f}")

    # ----- PART 3: spread strategies -----------------------------------------
    spreads = {}

    # WTI / Brent — trans-atlantic spread
    wti = close["CL=F"].dropna()
    brent = close["BZ=F"].reindex(wti.index, method="ffill")
    wti_brent = (wti - brent).dropna()
    spreads["WTI-Brent"] = wti_brent

    # 321 Crack proxy — 2 RBOB + HO - 3 WTI (refining margin)
    # convert RB ($/gal) and HO ($/gal) to $/bbl by * 42
    rb_bbl = close["RB=F"].reindex(wti.index, method="ffill") * 42
    ho_bbl = close["HO=F"].reindex(wti.index, method="ffill") * 42
    crack321 = (2*rb_bbl + ho_bbl - 3*wti).dropna()
    spreads["321Crack"] = crack321

    # RBOB / HO — gasoline vs distillate
    rb_ho = (rb_bbl - ho_bbl).dropna()
    spreads["RBOB-HO"] = rb_ho

    print()
    print("=" * 80)
    print("PART 3 - Cross-product spread strategies (predict 5d delta-spread)")
    print("=" * 80)
    print(f"  {'Spread':<22}" + "  ".join(f"{f:>7}" for f in families) + "    winner")
    print("  " + "-" * 78)
    spread_results = {}
    for name, ser in spreads.items():
        # For spreads: predict ABSOLUTE 5d change, not %
        df = pd.DataFrame(index=ser.index)
        df["ret_1d"]   = ser.diff(1)
        df["ret_5d"]   = ser.diff(5)
        df["ret_20d"]  = ser.diff(20)
        df["dxy_5d_ret"]  = dxy.reindex(ser.index, method="ffill").pct_change(5) * 100
        df["dxy_20d_ret"] = dxy.reindex(ser.index, method="ffill").pct_change(20) * 100
        df["bb_pos"]   = _bb_pos(ser, 20)
        df["vol_z"]    = _vol_z(ser)
        df["rsi_14n"]  = (_rsi(ser, 14) - 50) / 50
        df["target"]   = ser.shift(-HORIZON) - ser
        df = df.dropna()
        X = df[FEAT_NAMES].values.astype(float)
        y = df["target"].values.astype(float)
        split = int(len(X) * 0.8)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[:split])
        X_te_s = scaler.transform(X[split:])
        fits = fit_predict(X_tr_s, y[:split], X_te_s, y[split:])
        sig_stats = backtest_signals(fits["LGBM"]["pred_te"],
                                       y[split:],
                                       threshold=ser.diff().abs().median())
        spread_results[name] = {"fits": fits, "signals": sig_stats, "n_test": len(X) - split}
        print(fmt_row_fits(name, fits, families))

    print()
    print(f"  {'Spread':<10}{'Trades':>8}{'PnL':>10}{'Win%':>8}{'Sharpe':>9}"
          f"{'MaxDD':>10}")
    print("  " + "-" * 60)
    for name, r in spread_results.items():
        s = r["signals"]
        print(f"  {name:<10}{s['n_trades']:>8}{s['pnl_total']:>+10.3f}"
              f"{s['win_rate']:>7.1f}%{s['sharpe']:>+9.2f}"
              f"{s['max_dd']:>10.3f}")

    # ----- Final synthesis ---------------------------------------------------
    print()
    print("=" * 80)
    print("FINAL: which strategies are tradable?")
    print("=" * 80)
    print("  Criteria: Sharpe > 0.5 AND Win% > 50 AND n_trades >= 10")
    print()
    tradable = []
    for p in PRODUCTS:
        if results[p] is None: continue
        s = results[p]["signals"]
        if s["sharpe"] > 0.5 and s["win_rate"] > 50 and s["n_trades"] >= 10:
            tradable.append((p, s["sharpe"], s["pnl_total"], s["win_rate"]))
    for name, r in spread_results.items():
        s = r["signals"]
        if s["sharpe"] > 0.5 and s["win_rate"] > 50 and s["n_trades"] >= 10:
            tradable.append((name, s["sharpe"], s["pnl_total"], s["win_rate"]))
    if tradable:
        for name, sh, pnl, wr in sorted(tradable, key=lambda x: -x[1]):
            print(f"  * {name:<14}  Sharpe={sh:+.2f}  PnL={pnl:+.2f}  WinRate={wr:.1f}%")
    else:
        print("  (none — every strategy fails the threshold; signal is too noisy)")


if __name__ == "__main__":
    main()

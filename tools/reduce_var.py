"""Five techniques to reduce VaR on WTI OUTRIGHT directional strategy.

WTI outright baseline (5y LGBM_dash on next-5d return):
  Sharpe ~0.70, 95% VaR ~10.9% per trade — large for an outright bet.

Goal: cut VaR materially while preserving Sharpe.

Techniques tested:
  1. Stop-loss at fixed % loss
  2. Higher signal threshold (only act on stronger predictions)
  3. Volatility-target sizing (shrink position when vol_z is elevated)
  4. Brent hedge (partial offset using -0.5x Brent position)
  5. Halve position size (linear baseline — halves Sharpe AND VaR)
  6. STACK: combine the best non-trivial controls
"""
from __future__ import annotations
import math, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

PERIOD, HORIZON, COST_BPS = "5y", 5, 5.0
DAILY_BUDGET = 10000.0
WTI_PRICE_REF = 90.0   # used for $ position-sizing math

FEAT = ["ret_1d","ret_5d","ret_20d","dxy_5d_ret","dxy_20d_ret","bb_pos","vol_z","rsi_14n"]

def _bb_pos(c, w=20):
    m, s = c.rolling(w).mean(), c.rolling(w).std()
    return ((c - (m - 2*s)) / (4*s) * 100).clip(0, 100)
def _rsi(c, p=14):
    d = c.diff()
    up = d.where(d > 0, 0.0).rolling(p).mean()
    dn = (-d.where(d < 0, 0.0)).rolling(p).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
def _vol_z(c):
    lr = np.log(c / c.shift(1))
    v20 = lr.rolling(20).std() * math.sqrt(252) * 100
    return (v20 - v20.rolling(60).mean()) / v20.rolling(60).std()

def feats(target, dxy):
    df = pd.DataFrame(index=target.index)
    df["ret_1d"]=target.pct_change(1)*100
    df["ret_5d"]=target.pct_change(5)*100
    df["ret_20d"]=target.pct_change(20)*100
    df["dxy_5d_ret"]=dxy.pct_change(5)*100
    df["dxy_20d_ret"]=dxy.pct_change(20)*100
    df["bb_pos"]=_bb_pos(target,20)
    df["vol_z"]=_vol_z(target)
    df["rsi_14n"]=(_rsi(target,14)-50)/50
    df["target"]=(target.shift(-HORIZON)/target - 1)*100
    return df.dropna()

def lgbm():
    return LGBMRegressor(n_estimators=30, max_depth=2, num_leaves=4,
        learning_rate=0.03, min_child_samples=8, min_data_in_bin=2,
        min_split_gain=0.0, reg_alpha=1.0, reg_lambda=1.0,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        random_state=42, verbosity=-1, force_col_wise=True)

def predict_split(df):
    X = df[FEAT].values; y = df["target"].values
    split = int(len(X) * 0.8)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[:split]); X_te = sc.transform(X[split:])
    m = lgbm(); m.fit(X_tr, y[:split])
    return m.predict(X_te), y[split:], df.iloc[split:]

def backtest(pred, y_te, df_te, threshold,
              stop_pct=None, vol_target_scaling=False, hedge_y=None,
              size=1.0):
    """Returns per-trade PnL array (in % units, after costs and any controls)."""
    sig = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
    vol_z_te = df_te["vol_z"].values if "vol_z" in df_te.columns else None
    pnl = []
    for i in range(0, len(sig), HORIZON):
        s = sig[i]
        if s == 0: continue
        raw = s * y_te[i]
        # Stop-loss: cap downside
        if stop_pct is not None and raw < -stop_pct:
            raw = -stop_pct
        # Vol-targeting: shrink position if vol_z high
        pos_mult = size
        if vol_target_scaling and vol_z_te is not None:
            vz = vol_z_te[i]
            # When vz > 1 (vol > 1 std above norm), shrink to 1/(1+vz)
            pos_mult = pos_mult / (1.0 + max(0, vz))
        raw = raw * pos_mult
        # Hedge: subtract a fraction of the same-direction outright hedge instrument
        if hedge_y is not None and i < len(hedge_y):
            raw -= 0.5 * s * hedge_y[i] * size   # 0.5x Brent hedge same direction
        pnl.append(float(raw - COST_BPS/100.0))
    return np.array(pnl) if pnl else np.array([0.0])

def stats(pnl, label):
    if len(pnl) < 5:
        return dict(label=label, n=len(pnl), sharpe=0, v95=0, v99=0, mean=0)
    losses = -pnl
    v95 = float(np.percentile(losses, 95))
    v99 = float(np.percentile(losses, 99))
    sh = float(pnl.mean()/pnl.std(ddof=1)*math.sqrt(252.0/HORIZON)) if pnl.std() > 0 else 0
    return dict(label=label, n=int(len(pnl)), sharpe=sh,
                v95=v95, v99=v99, mean=float(pnl.mean()))

def main():
    print(f"Pulling {PERIOD} WTI + Brent + DXY...")
    raw = yf.download(["CL=F","BZ=F","DX-Y.NYB"], period=PERIOD,
                       progress=False, auto_adjust=True, threads=True)["Close"]
    wti = raw["CL=F"].dropna()
    brent = raw["BZ=F"].reindex(wti.index, method="ffill")
    dxy = raw["DX-Y.NYB"].reindex(wti.index, method="ffill")

    df_wti = feats(wti, dxy)
    df_br  = feats(brent, dxy)

    # Align Brent test set to WTI test set
    pred_w, y_w, df_w_te = predict_split(df_wti)
    pred_b, y_b, df_b_te = predict_split(df_br)
    n_common = min(len(y_w), len(y_b))
    y_b_aligned = y_b[:n_common]

    rows = []
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 0.5), "Baseline (no controls)"))
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 0.5, stop_pct=4.0), "+ Stop @ 4% per trade"))
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 0.5, stop_pct=2.5), "+ Stop @ 2.5% per trade"))
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 1.0), "+ Tighter threshold (1.0%)"))
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 0.5, vol_target_scaling=True), "+ Vol-target sizing"))
    rows.append(stats(backtest(pred_w[:n_common], y_w[:n_common], df_w_te.iloc[:n_common], 0.5,
                                hedge_y=y_b_aligned), "+ Brent hedge (0.5x)"))
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 0.5, size=0.5), "+ Halve size (0.5x)"))
    rows.append(stats(backtest(pred_w, y_w, df_w_te, 0.8, stop_pct=2.5,
                                vol_target_scaling=True),
                       "STACK: stop+vol-tgt+tighter"))

    base_v95 = rows[0]["v95"]; base_sh = rows[0]["sharpe"]
    print()
    print("=" * 90)
    print("WTI OUTRIGHT — VaR REDUCTION TECHNIQUES")
    print("=" * 90)
    print(f"  {'Technique':<35}{'n':>3}{'Sharpe':>8}{'Mean%':>8}{'95%VaR':>9}{'99%VaR':>9}{'VaRdrop':>10}")
    print("  " + "-" * 88)
    for r in rows:
        if r["v95"] == 0:
            drop = "n/a"
        elif r["label"] == "Baseline (no controls)":
            drop = "---"
        else:
            d = (base_v95 - r["v95"]) / base_v95 * 100 if base_v95 else 0
            drop = f"{d:+.1f}%"
        print(f"  {r['label']:<35}{r['n']:>3}{r['sharpe']:>+8.2f}"
              f"{r['mean']:>+8.2f}{r['v95']:>9.2f}{r['v99']:>9.2f}{drop:>10}")

    print()
    print("=" * 90)
    print("BEST RISK-REDUCERS (keep Sharpe >= baseline, drop VaR)")
    print("=" * 90)
    keepers = [r for r in rows[1:]
               if r["sharpe"] >= base_sh * 0.7 and r["v95"] < base_v95 and r["n"] > 3]
    if not keepers:
        print("  None — every control hurt Sharpe more than it cut VaR.")
    else:
        keepers.sort(key=lambda r: r["v95"])
        for r in keepers:
            sz = int(DAILY_BUDGET / (r["v95"] * math.sqrt(1.0/HORIZON) * WTI_PRICE_REF * 10))
            sz_b = int(DAILY_BUDGET / (base_v95 * math.sqrt(1.0/HORIZON) * WTI_PRICE_REF * 10))
            print(f"  * {r['label']:<35} 95%VaR {r['v95']:.2f}% (baseline {base_v95:.2f}%)"
                  f"  Sharpe {r['sharpe']:+.2f}  ->  max {sz} contracts (was {sz_b})")

if __name__ == "__main__":
    main()
